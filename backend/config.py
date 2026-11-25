"""Configurações do backend MuJoCo, especificações da garra e servidor WebSocket 60 FPS."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ASSETS_DIR = PROJECT_ROOT / "assets"
DEFAULT_MJCF_PATH = BACKEND_DIR / "models" / "gripper.mjcf"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class PhysicsConfig:
    """Parâmetros do motor físico MuJoCo e cadência de sincronização."""

    mjcf_path: Path = DEFAULT_MJCF_PATH
    timestep: float = 0.002  # 500 Hz integração física interna
    broadcast_fps: int = 60  # 60 FPS streaming de estado para o frontend
    substeps_per_frame: int = 8  # 8 * 0.002s = 16ms (~60 Hz tempo real)


@dataclass(frozen=True)
class GripperConfig:
    """Limites operacionais, velocidade, força e curso mecânico da garra."""

    workspace_x: Tuple[float, float] = (-0.55, 0.55)
    workspace_y: Tuple[float, float] = (-0.52, 0.70)
    workspace_z: Tuple[float, float] = (0.022, 0.55)
    # Compatibilidade com o pórtico antigo (fallback cinemático)
    carriage_home_z: float = 0.30
    tcp_z_offset: float = -0.062
    slide_z_limits: Tuple[float, float] = (0.03, 0.55)

    default_speed: float = 0.50  # m/s (escala 0.05 .. 1.50)
    min_speed: float = 0.05
    max_speed: float = 1.50

    default_force: float = 50.0  # N (escala 1.0 .. 100.0)
    min_force: float = 1.0
    max_force: float = 100.0

    default_step_duration: float = 0.16  # s por pulso discreto de comando
    max_aperture: float = 0.080  # 8.0 cm aberta
    min_aperture: float = 0.000  # 0.0 cm fechada


@dataclass(frozen=True)
class ServerConfig:
    """Parâmetros de rede HTTP/WebSocket para desenvolvimento e produção."""

    host: str = field(default_factory=lambda: os.getenv("SIM_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("SIM_PORT", "8000")))
    ws_path: str = "/ws"
    cors_origins: str = field(default_factory=lambda: os.getenv("SIM_CORS_ORIGIN", "same-origin"))
    client_timeout_s: float = 30.0
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


@dataclass(frozen=True)
class VisionConfig:
    enabled: bool = False
    model_path: Path = PROJECT_ROOT / "models/vision/odontologia.pt"
    calibration_path: Path = PROJECT_ROOT / "models/vision/calibration.json"
    camera_name: str = "vision_top"
    width: int = 640
    height: int = 480
    imgsz: int = 320
    device: str = "cpu"
    cpu_threads: int = 2
    target_fps: float = 10.0
    confidence_threshold: float = 0.60
    max_age_ms: float = 300.0
    stable_frames: int = 3
    stability_m: float = 0.008
    enable_pick: bool = False
    pick_max_attempts: int = 2
    allowed_classes: Tuple[str, ...] = ("dental_tool", "floss_holder")

    def __post_init__(self) -> None:
        if not (64 <= self.width <= 1920 and 64 <= self.height <= 1080):
            raise ValueError("Resolução de visão deve estar entre 64x64 e 1920x1080.")
        if not (64 <= self.imgsz <= 1920):
            raise ValueError("imgsz inválido.")
        for name in ("target_fps", "confidence_threshold", "max_age_ms", "stability_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Configuração de visão inválida: {name}")
        if self.target_fps > 60 or self.confidence_threshold > 1 or self.stable_frames < 1 or not 1 <= self.cpu_threads <= 32:
            raise ValueError("FPS, confiança ou estabilidade de visão inválidos.")
        if (isinstance(self.pick_max_attempts, bool)
                or not isinstance(self.pick_max_attempts, int)
                or self.pick_max_attempts < 1):
            raise ValueError("pick_max_attempts deve ser pelo menos 1.")


@dataclass(frozen=True)
class BackendConfig:
    """Configuração consolidada do backend."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    frontend_dir: Path = FRONTEND_DIR
    assets_dir: Path = ASSETS_DIR
    logs_dir: Path = LOGS_DIR


def load_backend_config(
    *,
    host: str | None = None,
    port: int | None = None,
    fps: int | None = None,
    mjcf_path: str | Path | None = None,
    log_level: str | None = None,
    vision: VisionConfig | None = None,
) -> BackendConfig:
    """Cria `BackendConfig` aplicando sobrescritas de CLI ou variáveis de ambiente."""
    phys_default = PhysicsConfig()
    srv_default = ServerConfig()

    resolved_fps = int(fps) if fps is not None else phys_default.broadcast_fps
    substeps = max(1, round((1.0 / max(1, resolved_fps)) / phys_default.timestep))

    physics = PhysicsConfig(
        mjcf_path=Path(mjcf_path) if mjcf_path is not None else phys_default.mjcf_path,
        timestep=phys_default.timestep,
        broadcast_fps=resolved_fps,
        substeps_per_frame=substeps,
    )
    server = ServerConfig(
        host=host if host is not None else srv_default.host,
        port=int(port) if port is not None else srv_default.port,
        log_level=(log_level or srv_default.log_level).upper(),
    )
    return BackendConfig(physics=physics, gripper=GripperConfig(), server=server,
                         vision=vision or VisionConfig())


def configure_backend_logger(
    name: str = "gripper_backend",
    level: str = "INFO",
    logs_dir: Path = LOGS_DIR,
) -> logging.Logger:
    """Configura logger estruturado em console e arquivo rotativo."""
    logger = logging.getLogger(name)
    numeric = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setLevel(numeric)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            logs_dir / "backend_server.log",
            maxBytes=3 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass

    logger.propagate = False
    return logger
