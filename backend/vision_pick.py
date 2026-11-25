"""Orquestração assíncrona de captura; não consulta posições reais dos objetos."""
from __future__ import annotations

import asyncio
import math
import logging
import time
import uuid


class RecoverablePickFailure(Exception):
    """Falha de observação ou preensão que permite uma nova tentativa segura."""


class VisionPick:
    def __init__(self, env, controller, vision):
        self.env, self.controller, self.vision = env, controller, vision
        self.logger = getattr(env, "logger", logging.getLogger(__name__))
        self.task = None
        self._cancel_requested = False
        max_attempts = getattr(getattr(vision, "config", None), "pick_max_attempts", 1)
        self.status = {"type": "vision_pick_status", "stage": "idle", "operation_id": None,
                       "attempt": 0, "max_attempts": max_attempts, "last_failure": None,
                       "recovered": False}
        self.revision = None
        self.motion_started = False
        self._cancel_requested = False

    @property
    def active(self):
        return self.task is not None and not self.task.done()

    def start(self, payload):
        if self.active:
            raise ValueError("Operação de visão já está em andamento.")
        class_name = payload.get("class_name")
        if class_name not in self.vision.config.allowed_classes:
            raise ValueError("Informe class_name com a ferramenta que o braço deve pegar.")
        if not self.vision.config.enabled or not self.vision.config.enable_pick:
            raise ValueError("Habilite a visão e a operação de pegar ferramenta.")
        self.motion_started = False
        self.status = {"type": "vision_pick_status", "stage": "detecting",
                       "operation_id": uuid.uuid4().hex, "error": None,
                       "class_name": class_name, "target_world_m": None, "plan": [],
                       "attempt": 1, "max_attempts": self.vision.config.pick_max_attempts,
                       "last_failure": None, "recovered": False}
        self.logger.info("Captura visual iniciada: %s", self.status)
        self.task = asyncio.create_task(self._run(class_name), name="VisionPick")

    def cancel(self):
        if self.active:
            self._cancel_requested = True
            self.task.cancel()
            if self.vision is not None:
                self.vision.cancel_detection()
            if self.motion_started:
                self.env.hold_position()
            self.status.update(stage="cancelled", error="Operação cancelada; dedos mantidos.")

    def _stage(self, value):
        self.status["stage"] = value
        self.logger.info("Captura visual %s: %s", self.status["operation_id"], value)

    async def _wait(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.revision is not None and self.env.scene_revision != self.revision:
                raise ValueError("Cena alterada durante a captura.")
            if predicate(self.env.get_state_snapshot()):
                return
            await asyncio.sleep(0.04)
        raise ValueError("Timeout aguardando conclusão física da etapa.")

    async def _move(self, point, stage):
        cfg = self.env.config.gripper
        if len(point) != 3 or not all(math.isfinite(v) for v in point):
            raise ValueError("Alvo inválido.")
        for value, bounds in zip(point, (cfg.workspace_x, cfg.workspace_y, cfg.workspace_z)):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError("Alvo fora do workspace.")
        if math.dist(self.env._clamp_cartesian_xyz(*point), point) > 1e-6:
            raise ValueError("Alvo fora do envelope alcançável do braço.")
        self._stage(stage)
        self.controller.execute_payload({"command": "move_to", "target": dict(zip(("x", "y", "z"), point)),
                                         "speed": 0.25, "force": 50})
        # O campo velocity legado contém velocidades articulares no UR5e.
        # Medir repouso cartesiano pelo deslocamento real do TCP entre amostras.
        previous, settled = None, 0
        def reached(state):
            nonlocal previous, settled
            position = state["pos"]
            slow = previous is not None and math.dist(previous, position) < .0015
            settled = settled + 1 if slow and math.dist(position, point) < .006 else 0
            previous = position
            return settled >= 3
        await self._wait(reached)

    def _plan(self, point):
        plan = [{"command":"open"},
                {"command":"move_to","target":[point[0],point[1]+.12,.20]},
                {"command":"revalidate"},
                {"command":"move_to","target":[point[0],point[1],.20]},
                {"command":"move_to","target":list(point)},
                {"command":"close"},
                {"command":"move_to","target":[point[0],point[1],.22]},
                {"command":"move_to","target":[-.30,.26,.22]},
                {"command":"move_to","target":[-.30,.26,.042]},
                {"command":"open"},
                {"command":"move_to","target":[-.30,.26,.22]}]
        for step in plan:
            target = step.get("target")
            if target is not None and (not all(math.isfinite(v) for v in target)
                                      or math.dist(self.env._clamp_cartesian_xyz(*target),target)>1e-6):
                raise ValueError("Plano contém alvo fora do envelope alcançável.")
        return plan

    def _raise_detection_failure(self, exc, context):
        """Repete apenas falhas conhecidas de observação, nunca erros de simulação."""
        message = str(exc)
        normalized = message.casefold()
        recoverable_markers = (
            "timeout", "expirad", "instável", "instavel", "ambígu", "ambigu",
            "não encontrado", "nao encontrado", "não detect", "nao detect",
            "sem observação recente", "sem observacao recente", "candidato inválido",
        )
        if isinstance(exc, TimeoutError) or any(marker in normalized for marker in recoverable_markers):
            raise RecoverablePickFailure(f"{context}: {message}") from exc
        raise exc

    async def _detect_candidate(self, class_name):
        try:
            packet = await self.vision.detect(class_name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._raise_detection_failure(exc, "Detecção do alvo falhou")
        if not isinstance(packet, dict) or packet.get("status") != "ready":
            message = packet.get("error") if isinstance(packet, dict) else None
            self._raise_detection_failure(
                ValueError(message or "Alvo não detectado em uma observação válida."),
                "Detecção do alvo falhou",
            )
        detections = packet.get("detections", [])
        if not isinstance(detections, list):
            raise RecoverablePickFailure("Observação do alvo inválida ou instável.")
        choices = [d for d in detections if isinstance(d, dict)
                   and d.get("class_name") == class_name
                   and isinstance(d.get("grasp_candidate"), dict)
                   and d["grasp_candidate"].get("valid")]
        if len(choices) != 1:
            raise RecoverablePickFailure("Alvo não encontrado com segurança ou seleção instável.")
        try:
            detection = self.vision.resolve(packet["scene_revision"], packet["frame_id"],
                                            choices[0]["detection_id"])
            point = detection["grasp_candidate"]["position_world_m"]
            if point is None or len(point) != 3:
                raise ValueError("posição do alvo indisponível")
        except ValueError as exc:
            self._raise_detection_failure(exc, "Alvo não pôde ser resolvido")
        except (KeyError, TypeError) as exc:
            raise RecoverablePickFailure("Alvo não pôde ser resolvido a partir da observação.") from exc
        return packet, detection, point

    async def _run_attempt(self, class_name):
        packet, detection, point = await self._detect_candidate(class_name)
        revision = packet["scene_revision"]
        self.revision = revision
        self.status.update(frame_id=packet["frame_id"], scene_revision=revision,
                           target_world_m=point, plan=self._plan(point))
        self._stage("planning")
        self.motion_started = True
        self.controller.execute_payload({"command": "open", "speed": 0.5, "force": 50})
        await self._wait(lambda s: s["aperture"] > 0.055, 3)
        # Aproximação lateral preserva a linha de visão da câmera superior.
        await self._move([point[0], point[1] + .12, 0.20], "approaching")
        # Reobservar antes do contato: nenhuma posição real de objeto é usada.
        self._stage("revalidating")
        try:
            packet = await self.vision.detect(class_name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._raise_detection_failure(exc, "Alvo não pôde ser revalidado")
        if not isinstance(packet, dict) or packet.get("status") != "ready":
            message = packet.get("error") if isinstance(packet, dict) else None
            self._raise_detection_failure(
                ValueError(message or "Alvo não pôde ser revalidado em uma observação válida."),
                "Alvo não pôde ser revalidado",
            )
        if packet.get("scene_revision") != revision:
            raise ValueError("Cena alterada durante a revalidação do alvo.")
        detections = packet.get("detections", [])
        if not isinstance(detections, list):
            raise RecoverablePickFailure("Alvo não pôde ser revalidado em uma observação válida.")
        choices = [d for d in detections if isinstance(d, dict)
                   and d.get("class_name") == class_name
                   and isinstance(d.get("grasp_candidate"), dict)
                   and d["grasp_candidate"].get("valid")
                   and isinstance(d["grasp_candidate"].get("position_world_m"), (list, tuple))
                   and len(d["grasp_candidate"]["position_world_m"]) == 3
                   and math.dist(d["grasp_candidate"]["position_world_m"], point) < 0.008]
        if len(choices) != 1:
            raise RecoverablePickFailure("Alvo não pôde ser revalidado antes da descida.")
        point = choices[0]["grasp_candidate"]["position_world_m"]
        self.status.update(revalidated_target_world_m=point,
                           remaining_plan=self._plan(point)[3:])
        # Durante alinhamento/descida a garra oclui o objeto. Travar o alvo
        # recém-validado por no máximo 6 s, em vez de perseguir caixas antigas.
        async def align_and_descend():
            await self._move([point[0], point[1], .20], "aligning")
            await self._move(point, "descending")
        await asyncio.wait_for(align_and_descend(), timeout=6)
        self._stage("closing")
        self.controller.execute_payload({"command": "close", "speed": 0.5, "force": 50})
        try:
            await self._wait(lambda s: s["is_grasping"] and s["contact_force"] > 0
                             and s.get("grasped_object") == detection["class_name"], 3)
        except ValueError as exc:
            if "Timeout" not in str(exc):
                raise
            raise RecoverablePickFailure("Contato ou preensão não confirmados.") from exc
        await self._move([point[0], point[1], 0.22], "lifting")
        if not self.env.get_state_snapshot()["is_grasping"]:
            raise RecoverablePickFailure("Contato perdido após elevação.")
        await self._move([-0.30, 0.26, 0.22], "delivering")
        await self._move([-0.30, 0.26, 0.042], "lowering")
        self.controller.execute_payload({"command": "open", "speed": 0.5, "force": 40})
        await self._wait(lambda s: not s["is_grasping"] and s["aperture"] > 0.055, 3)
        await self._move([-0.30, 0.26, 0.22], "retreating")

    async def _recover_between_attempts(self):
        """Abre a garra e retorna à pose segura antes de uma nova observação."""
        if self._cancel_requested:
            raise asyncio.CancelledError
        self._stage("recovering")
        self.controller.execute_payload({"command": "open", "speed": 0.5, "force": 50})
        await self._wait(lambda s: not s["is_grasping"] and s["aperture"] > 0.055, 3)
        await self._move([-0.30, 0.26, 0.22], "recovering")

    async def _run(self, class_name):
        max_attempts = self.vision.config.pick_max_attempts
        last_failure = None
        recovered = False
        for attempt in range(1, max_attempts + 1):
            if self._cancel_requested:
                raise asyncio.CancelledError
            self.status.update(attempt=attempt, max_attempts=max_attempts)
            self._stage("detecting")
            try:
                await self._run_attempt(class_name)
            except asyncio.CancelledError:
                # cancel() mantém a posição antes de qualquer comando de reset.
                raise
            except RecoverablePickFailure as exc:
                last_failure = str(exc)
                self.status.update(last_failure=last_failure)
                if attempt >= max_attempts:
                    if self.motion_started:
                        self.env.hold_position()
                    self.status.update(stage="failed", error=last_failure, recovered=False)
                    self.logger.warning("Captura visual falhou: %s", self.status)
                    return
                try:
                    if self.motion_started:
                        await self._recover_between_attempts()
                    recovered = True
                except asyncio.CancelledError:
                    raise
                except Exception as recovery_error:
                    if self.motion_started:
                        self.env.hold_position()
                    self.status.update(
                        stage="failed",
                        error=f"Falha durante a recuperação: {recovery_error}",
                        last_failure=last_failure,
                        recovered=False,
                    )
                    self.logger.warning("Recuperação da captura visual falhou: %s", self.status)
                    return
            except Exception as exc:
                if self.motion_started:
                    self.env.hold_position()
                self.status.update(stage="failed", error=str(exc), last_failure=str(exc),
                                   recovered=False)
                self.logger.warning("Captura visual falhou: %s", self.status)
                return
            else:
                self.status.update(stage="completed", error=None,
                                   last_failure=last_failure, recovered=recovered)
                return
