#!/usr/bin/env python3
"""Ponto de entrada do backend MuJoCo + WebSocket (`python backend/main.py --port 8000`)."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import VisionConfig, configure_backend_logger, load_backend_config  # noqa: E402
from backend.websocket_server import GripperWebSocketServer  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gripper-backend",
        description="Servidor de simulação MuJoCo 60 FPS com WebSocket e interface 3D web.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="IP de escuta (padrao 127.0.0.1; pode ser definido em SIM_HOST).",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Porta TCP para HTTP e WebSocket (padrão: 8000).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Taxa de atualização do stream WebSocket em quadros por segundo (padrão: 60).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Caminho customizado para o arquivo .mjcf da garra.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nível de log do servidor (padrão: INFO).",
    )
    parser.add_argument("--vision", action="store_true", help="Ativar câmera e detector YOLO opcional.")
    parser.add_argument("--vision-model", type=Path, default=VisionConfig().model_path)
    parser.add_argument("--vision-calibration", type=Path, default=VisionConfig().calibration_path)
    parser.add_argument("--vision-device", default="cpu")
    parser.add_argument("--vision-imgsz", type=int, default=VisionConfig().imgsz)
    parser.add_argument("--vision-threads", type=int, default=VisionConfig().cpu_threads)
    parser.add_argument("--vision-fps", type=float, default=10.0)
    parser.add_argument("--vision-confidence", type=float, default=0.60)
    parser.add_argument("--vision-max-age-ms", type=float, default=300.0)
    parser.add_argument("--vision-pick-max-attempts", type=int, default=2,
                        help="Número máximo de tentativas para captura visual (padrão: 2).")
    parser.add_argument("--vision-enable-pick", action="store_true",
                        help="Habilitar captura; exige calibração e manifesto validados.")
    return parser


async def _run_async_server(args: argparse.Namespace) -> int:
    cfg = load_backend_config(
        host=args.host,
        port=args.port,
        fps=args.fps,
        mjcf_path=args.model,
        log_level=args.log_level,
        vision=VisionConfig(enabled=args.vision, model_path=args.vision_model,
                            calibration_path=args.vision_calibration,
                            device=args.vision_device, target_fps=args.vision_fps,
                            imgsz=args.vision_imgsz,
                            cpu_threads=args.vision_threads,
                            confidence_threshold=args.vision_confidence,
                            max_age_ms=args.vision_max_age_ms,
                            enable_pick=args.vision_enable_pick,
                            pick_max_attempts=args.vision_pick_max_attempts),
    )
    logger = configure_backend_logger("BackendMain", level=cfg.server.log_level)
    server = GripperWebSocketServer(config=cfg, logger=logger)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    await server.start()
    print(
        f"\n[SimuladorGarra 3D] Backend MuJoCo em execução!\n"
        f"  • Interface Web 3D : http://localhost:{server.bound_port}\n"
        f"  • WebSocket 60 FPS : ws://localhost:{server.bound_port}{cfg.server.ws_path}\n"
        f"  • Healthcheck REST : http://localhost:{server.bound_port}/health\n"
        f"  • Motor Físico     : {server.env.engine_name}\n"
        f"Pressione Ctrl+C para encerrar.\n",
        flush=True,
    )

    try:
        await stop_event.wait()
    finally:
        await server.stop()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run_async_server(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
