"""Contratos e falhas de visão sem dependência de pesos/GPU."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from backend.config import VisionConfig
from backend.vision_geometry import VisionGeometry, pixel_to_world, world_to_pixel
from backend.vision_service import VisionService
from backend.vision_pick import VisionPick
from backend.vision import YoloDetector

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "models/vision/calibration.json"


@pytest.fixture
def camera():
    return json.loads(CALIBRATION.read_text(encoding="utf-8"))["camera"]


def test_project_round_trip_and_axis(camera):
    for point in ([.25,-.22,.01], [.15,-.34,.004], [.39,-.11,.01]):
        pixel = world_to_pixel(point, camera)
        assert np.allclose(pixel_to_world(*pixel, camera, point[2]), point)
    center = pixel_to_world(319.5,239.5,camera,0)
    assert np.allclose(center,[.28,-.22,0])
    assert pixel_to_world(319.5,100,camera,0)[1] > -.22


def test_invalid_camera_rays(camera):
    with pytest.raises(ValueError):
        pixel_to_world(100,100,camera,1)
    with pytest.raises(ValueError):
        world_to_pixel(np.array(camera["position"]) + np.array(camera["rotation"])[:,2], camera)


def test_calibration_requires_validation_and_matches_pose(camera):
    geometry = VisionGeometry(CALIBRATION)
    geometry.calibration["validated_for_pick"] = False
    detection = {"class_name":"dental_tool", "bbox_xyxy_px":[295,180,310,310]}
    candidate = geometry.candidate(detection,camera)
    assert not candidate["valid"] and candidate["position_world_m"] is not None
    geometry.calibration["validated_for_pick"] = True
    assert geometry.candidate(detection,camera)["valid"]
    camera["position"][0] += .01
    assert geometry.candidate(detection,camera)["position_world_m"] is None


@pytest.mark.parametrize("box", [[-1,20,30,40],[20,20,640,480],[10,10,10,30]])
def test_cropped_boxes_rejected(camera,box):
    g = VisionGeometry(CALIBRATION)
    assert not g.candidate({"class_name":"dental_tool","bbox_xyxy_px":box},camera)["valid"]


@pytest.mark.parametrize("kwargs", [{"target_fps":float("nan")},{"max_age_ms":-1},
    {"confidence_threshold":1.1},{"stable_frames":0},{"width":4000},
    {"pick_max_attempts":0}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        VisionConfig(**kwargs)


def test_pick_attempts_cli_flag():
    from backend.main import build_arg_parser
    args = build_arg_parser().parse_args(["--vision-pick-max-attempts", "3"])
    assert args.vision_pick_max_attempts == 3


def service_with_candidate():
    env = SimpleNamespace(scene_revision=0)
    service = VisionService(env,VisionConfig(enabled=True,enable_pick=True))
    candidate = {"detection_id":"1:0","class_name":"dental_tool",
                 "grasp_candidate":{"valid":True,"position_world_m":[.25,-.22,.024]}}
    packet = {"status":"ready","pick_enabled":True,"frame_id":1,"scene_revision":0,
              "detections":[candidate]}
    service._packet = packet
    service._captured = time.perf_counter()
    service._frames[1] = (0,b"jpeg",deepcopy(packet),service._captured)
    return env, service


def test_result_expiry_reset_and_frame_identity():
    env,s = service_with_candidate()
    assert s.resolve(0,1,"1:0")["class_name"] == "dental_tool"
    with pytest.raises(ValueError): s.resolve(0,1,"1:9")
    with pytest.raises(ValueError): s.resolve(0,2,"1:0")
    env.scene_revision = 1
    assert s.frame(1) is None and s.snapshot()["detections"] == []
    with pytest.raises(ValueError): s.resolve(0,1,"1:0")
    env.scene_revision = 0
    old = time.perf_counter()-1
    s._captured = old
    s._frames[1] = (0,b"jpeg",s._packet,old)
    assert s.snapshot()["status"] == "stale"
    assert "limite de 300 ms" in s.snapshot()["error"]
    with pytest.raises(ValueError): s.resolve(0,1,"1:0")


def test_cpu_thread_limit_survives_predictor_setup(tmp_path, monkeypatch):
    events = []
    threads = [10]
    def set_threads(count):
        threads[0] = count
        events.append(("threads", count))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        get_num_threads=lambda: threads[0], set_num_threads=set_threads))
    class Model:
        names = {0: "dental_tool", 1: "floss_holder"}
        def __init__(self, *args, **kwargs):
            events.append(("load", threads[0]))
        def add_callback(self, name, callback):
            assert name == "on_predict_start"
            self.callback = callback
        def predict(self, *args, **kwargs):
            set_threads(8)  # Ultralytics selecionando CPU/criando predictor.
            self.callback(self)
            events.append(("infer", threads[0]))
            return [SimpleNamespace(boxes=None)]
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    path = tmp_path / "model.pt"
    path.write_bytes(b"test")
    path.with_suffix(".json").write_text(json.dumps({
        "sha256": hashlib.sha256(b"test").hexdigest(),
        "classes": list(Model.names.values())}))
    detector = YoloDetector(VisionConfig(model_path=path, cpu_threads=2))
    detector.detect(np.zeros((480, 640, 3), np.uint8))
    assert ("load", 2) in events
    assert [event for event in events if event[0] == "infer"] == [("infer", 2), ("infer", 2)]


def test_async_detection_reports_expiry_details():
    async def check():
        _, service = service_with_candidate()
        def request(class_name):
            service._packet["request_id"] = "test-expired"
            service._captured = time.perf_counter() - 1
            return "test-expired"
        service.request_detection = request
        with pytest.raises(ValueError, match="Observação expirada: idade .*limite de 300 ms"):
            await service.detect("dental_tool")
    asyncio.run(check())


def test_overloaded_physics_loop_yields_to_detection():
    from backend.config import BackendConfig
    from backend.websocket_server import GripperWebSocketServer
    async def check():
        ticks = []
        server = object.__new__(GripperWebSocketServer)
        def step():
            time.sleep(.02)  # Mais lento que o orçamento de 16,7 ms.
            ticks.append(1)
            if len(ticks) == 5:
                server._running = False  # Evita travar o teste se houver regressão.
        server.env = SimpleNamespace(step_physics=step)
        server.config = BackendConfig()
        server.clients = set()
        server._running = True
        task = asyncio.create_task(server._physics_broadcast_loop())
        await asyncio.sleep(0)
        server._running = False
        await task
        assert len(ticks) == 1
    asyncio.run(check())


@pytest.mark.parametrize("found,slow_third", [(True,False),(True,True),(False,False)])
def test_worker_bounded_cache_stability_and_cleanup(camera,tmp_path,found,slow_third):
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    calibration["validated_for_pick"] = True
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    owner = []
    captures = []
    class Camera:
        def __init__(self,env,cfg): self.env=env; owner.append(threading.get_ident())
        def capture(self):
            assert threading.get_ident() == owner[0]
            captures.append(1)
            captured = time.perf_counter() - (1 if slow_third and len(captures)==3 else 0)
            return np.zeros((480,640,3),np.uint8), {**camera,"scene_revision":self.env.scene_revision,"sim_time_s":0}, captured
        def close(self): owner.append(threading.get_ident())
    class Detector:
        manifest = {"validated_for_pick":True}
        identifier = "test"
        def __init__(self,cfg): pass
        def detect(self,rgb):
            if not found: return []
            return [{"class_name":"dental_tool","class_id":0,"confidence":.9,"bbox_xyxy_px":[295,180,310,310]}]
    env = SimpleNamespace(scene_revision=0)
    s = VisionService(env,VisionConfig(enabled=True,target_fps=60,calibration_path=path,enable_pick=True),Detector,Camera)
    s.start()
    deadline = time.perf_counter()+5
    try:
        time.sleep(.05)
        assert owner == [] and s.snapshot()["frames_captured"] == 0
        s.request_detection("dental_tool")
        while time.perf_counter()<deadline:
            if s.snapshot().get("frame_id") and not s.snapshot()["request_active"]: break
            time.sleep(.02)
        packet = s.snapshot()
        assert packet["status"] == "ready"
        if found:
            assert packet["detections"][0]["grasp_candidate"]["valid"]
        else:
            assert packet["detections"] == []
        expected = (4 if slow_third else 3) if found else 6
        assert s.snapshot()["frames_captured"] == expected
        time.sleep(.15)
        assert s.snapshot()["frames_captured"] == expected  # Nenhuma captura em repouso.
        s.request_detection("dental_tool")
        deadline = time.perf_counter()+5
        while s.snapshot()["request_active"] and time.perf_counter()<deadline:
            time.sleep(.02)
        assert s.snapshot()["frames_captured"] == expected + (3 if found else 6)
        assert len(s._frames) <= 4
        json.dumps(packet,allow_nan=False)
        env.scene_revision += 1
        reset = s.snapshot()
        assert reset["scene_revision"] == env.scene_revision
        assert reset["status"] == "idle" and reset["detections"] == [] and reset["frame_id"] is None
        time.sleep(.05)
        assert s.snapshot()["frames_captured"] == expected + (3 if found else 6)
    finally: s.stop()
    assert owner == [owner[0],owner[0]]


def test_disabled_vision_has_no_worker():
    s = VisionService(SimpleNamespace(scene_revision=0),VisionConfig())
    s.start()
    assert s.snapshot()["status"] == "disabled" and s._thread is None


def test_cancelled_capture_cannot_publish_over_a_new_request(camera):
    entered, release = threading.Event(), threading.Event()
    captures = []
    class Camera:
        def __init__(self,env,cfg): pass
        def capture(self):
            captures.append(1)
            if len(captures) == 1:
                entered.set()
                assert release.wait(3)
            return np.zeros((480,640,3),np.uint8), {**camera,"scene_revision":0,"sim_time_s":0}, time.perf_counter()
        def close(self): pass
    class Detector:
        manifest = {"validated_for_pick":True}
        identifier = "test"
        def __init__(self,cfg): pass
        def detect(self,rgb):
            return [{"class_name":"floss_holder","class_id":1,"confidence":.9,
                     "bbox_xyxy_px":[353,213,373,262]}]
    service = VisionService(SimpleNamespace(scene_revision=0),
                            VisionConfig(enabled=True,target_fps=60,enable_pick=True),Detector,Camera)
    service.start()
    try:
        old = service.request_detection("dental_tool")
        assert entered.wait(3)
        service.cancel_detection(old)
        new = service.request_detection("floss_holder")
        release.set()
        deadline = time.perf_counter()+5
        while service.snapshot()["request_active"] and time.perf_counter()<deadline:
            time.sleep(.01)
        result = service.snapshot()
        assert result["request_id"] == new and result["status"] == "ready"
        assert result["requested_class"] == "floss_holder"
        assert service.frame(1) is None  # Resultado da requisição cancelada foi descartado.
        count = len(captures)
        time.sleep(.1)
        assert len(captures) == count
    finally:
        release.set()
        service.stop()


def test_restart_does_not_reuse_cached_observations():
    _, service = service_with_candidate()
    def broken_camera(env,config):
        raise RuntimeError("Camera unavailable")
    service.camera_factory = broken_camera
    service.start()
    service.request_detection("dental_tool")
    service._thread.join(timeout=.1)
    deadline = time.perf_counter()+2
    while service.snapshot()["request_active"] and time.perf_counter()<deadline:
        time.sleep(.01)
    assert service.snapshot()["status"] == "error"
    assert service.snapshot()["frame_id"] is None
    assert service.frame(1) is None
    service.stop()


def test_missing_weights_and_manifest_hash_are_rejected(tmp_path):
    path = tmp_path / "model.pt"
    config = replace(VisionConfig(),model_path=path)
    with pytest.raises(FileNotFoundError): YoloDetector(config)
    path.write_bytes(b"untrusted or incomplete weights")
    with pytest.raises(ValueError,match="Manifesto ausente"): YoloDetector(config)
    path.with_suffix(".json").write_text(json.dumps({"sha256":"wrong"}))
    with pytest.raises(ValueError,match="Hash"): YoloDetector(config)


def test_cancel_holds_without_opening_jaws():
    async def check():
        events = []
        env = SimpleNamespace(hold_position=lambda:events.append("hold"))
        pick = VisionPick(env,None,None)
        pick.motion_started = True
        pick.task = asyncio.create_task(asyncio.sleep(5))
        pick.cancel()
        await asyncio.gather(pick.task,return_exceptions=True)
        assert events == ["hold"] and pick.status["stage"] == "cancelled"
    asyncio.run(check())


@pytest.mark.parametrize("cancel", [False, True])
def test_no_motion_until_detection_is_returned_and_validated(cancel):
    async def check():
        _, service = service_with_candidate()
        events = []
        waiting = asyncio.Event()
        async def detect(class_name):
            waiting.set()
            if cancel:
                await asyncio.Event().wait()
            return {"detections":[]}
        service.detect = detect
        env = SimpleNamespace(hold_position=lambda:events.append("hold"))
        controller = SimpleNamespace(execute_payload=lambda payload:events.append(payload))
        pick = VisionPick(env,controller,service)
        pick.start({"class_name":"dental_tool"})
        await waiting.wait()
        if cancel:
            pick.cancel()
        await asyncio.gather(pick.task,return_exceptions=True)
        assert events == []
        assert pick.status["stage"] == ("cancelled" if cancel else "failed")
        if not cancel:
            assert pick.status["attempt"] == 2 and pick.status["max_attempts"] == 2
            assert pick.status["last_failure"]
    asyncio.run(check())


@pytest.mark.parametrize("revalidate", [True, False])
def test_pick_state_machine_and_revalidation(revalidate):
    async def check():
        _,service = service_with_candidate()
        service.config = replace(service.config,pick_max_attempts=1)
        target = [.25,-.22,.024]
        detection = service.resolve(0,1,"1:0")
        state = {"pos":[0,.4,.3],"aperture":.08,"is_grasping":False,"contact_force":0}
        commands = []
        class Env:
            scene_revision = 0
            from backend.config import BackendConfig
            config = BackendConfig()
            def get_state_snapshot(self): return deepcopy(state)
            def hold_position(self): commands.append("hold")
            def _clamp_cartesian_xyz(self,*point): return list(point)
        class Controller:
            def execute_payload(self,payload):
                command = payload["command"]
                commands.append(command)
                if command == "move_to": state["pos"] = list(payload["target"].values())
                if command == "open": state.update(aperture=.08,is_grasping=False,contact_force=0)
                if command == "close": state.update(aperture=.01,is_grasping=True,contact_force=10,grasped_object="dental_tool")
        # Snapshot atualizado representa uma observação estável, nunca truth da cena.
        packet = deepcopy(service._packet)
        calls = []
        async def detect(class_name):
            calls.append(class_name)
            result = deepcopy(packet)
            if len(calls)>1 and not revalidate:
                result["detections"] = []
            return result
        service.detect = detect
        pick = VisionPick(Env(),Controller(),service)
        pick.start({"class_name":"dental_tool"})
        with pytest.raises(ValueError):
            pick.start({"class_name":"dental_tool"})
        await pick.task
        if revalidate:
            assert calls == ["dental_tool","dental_tool"]
            assert pick.status["plan"]
            assert pick.status["stage"] == "completed"
            assert commands.count("close") == 1 and commands.count("open") == 2
        else:
            assert pick.status["stage"] == "failed" and "close" not in commands and commands[-1] == "hold"
            assert pick.status["max_attempts"] == 1 and pick.status["attempt"] == 1
    asyncio.run(check())


def recovery_pick_fixture(*, failed_closes=1, block_recovery_open=False, move_target_on_retry=False):
    env_stub, service = service_with_candidate()
    state = {"pos":[0,.4,.3],"aperture":.08,"is_grasping":False,"contact_force":0}
    commands, detection_calls, frame_positions = [], [], {}
    recovery_waiting = asyncio.Event()

    class Env:
        scene_revision = 0
        from backend.config import BackendConfig
        config = BackendConfig()
        def get_state_snapshot(self): return deepcopy(state)
        def hold_position(self): commands.append("hold")
        def _clamp_cartesian_xyz(self,*point): return list(point)

    class Controller:
        close_count = 0
        open_count = 0
        def execute_payload(self,payload):
            command = payload["command"]
            commands.append(command)
            if command == "move_to": state["pos"] = list(payload["target"].values())
            if command == "open":
                self.open_count += 1
                aperture = .01 if block_recovery_open and self.open_count == 2 else .08
                state.update(aperture=aperture,is_grasping=False,contact_force=0)
            if command == "close":
                self.close_count += 1
                if self.close_count > failed_closes:
                    state.update(aperture=.01,is_grasping=True,contact_force=10,
                                 grasped_object="dental_tool")

    async def detect(class_name):
        detection_calls.append(class_name)
        frame_id = len(detection_calls)
        point = [.25,-.22,.024]
        if move_target_on_retry and frame_id >= 3:
            point = [.27,-.21,.024]
        frame_positions[frame_id] = point
        candidate = {"detection_id":f"{frame_id}:0","class_name":class_name,
                     "grasp_candidate":{"valid":True,"position_world_m":point}}
        packet = {"status":"ready","pick_enabled":True,"frame_id":frame_id,
                  "scene_revision":0,"detections":[candidate]}
        service._packet = deepcopy(packet)
        service._captured = time.perf_counter()
        service._frames[frame_id] = (0,b"jpeg",deepcopy(packet),service._captured)
        return packet

    service.detect = detect
    pick = VisionPick(Env(),Controller(),service)
    original_wait = pick._wait

    async def fast_wait(predicate, timeout=8):
        if pick.status.get("stage") == "closing" and pick.controller.close_count <= failed_closes:
            raise ValueError("Timeout aguardando conclusão física da etapa.")
        if block_recovery_open and pick.status.get("stage") == "recovering" and state["aperture"] <= .055:
            recovery_waiting.set()
            await asyncio.Event().wait()
        return await original_wait(predicate,timeout)

    pick._wait = fast_wait
    pick.recovery_waiting = recovery_waiting
    return pick, detection_calls, frame_positions


def test_pick_recupera_falha_de_preensao_na_segunda_tentativa():
    async def check():
        pick, calls, frame_positions = recovery_pick_fixture(move_target_on_retry=True)
        pick.start({"class_name":"dental_tool"})
        await pick.task
        assert pick.status["stage"] == "completed"
        assert pick.status["attempt"] == 2 and pick.status["max_attempts"] == 2
        assert pick.status["recovered"] is True
        assert pick.status["last_failure"] == "Contato ou preensão não confirmados."
        assert calls == ["dental_tool"] * 4
        assert pick.status["target_world_m"] == frame_positions[3]
    asyncio.run(check())


def test_pick_esgota_tentativas_recuperaveis():
    async def check():
        pick, calls, _ = recovery_pick_fixture(failed_closes=2)
        pick.start({"class_name":"dental_tool"})
        await pick.task
        assert pick.status["stage"] == "failed"
        assert pick.status["attempt"] == 2 and pick.status["max_attempts"] == 2
        assert pick.status["recovered"] is False
        assert pick.status["last_failure"] == "Contato ou preensão não confirmados."
        assert calls == ["dental_tool"] * 4
    asyncio.run(check())


def test_cancelamento_interrompe_espera_de_recuperacao():
    async def check():
        pick, calls, _ = recovery_pick_fixture(block_recovery_open=True)
        pick.start({"class_name":"dental_tool"})
        await asyncio.wait_for(pick.recovery_waiting.wait(),timeout=2)
        pick.cancel()
        await asyncio.gather(pick.task,return_exceptions=True)
        assert pick.status["stage"] == "cancelled"
        assert calls == ["dental_tool"] * 2
    asyncio.run(check())


def test_erro_de_simulacao_nao_gera_nova_tentativa():
    async def check():
        pick, calls, _ = recovery_pick_fixture()
        execute = pick.controller.execute_payload
        def fail_movement(payload):
            if payload["command"] == "move_to":
                raise RuntimeError("erro de simulação")
            execute(payload)
        pick.controller.execute_payload = fail_movement
        pick.start({"class_name":"dental_tool"})
        await pick.task
        assert pick.status["stage"] == "failed"
        assert pick.status["attempt"] == 1
        assert pick.status["recovered"] is False
        assert "erro de simulação" in pick.status["last_failure"]
        assert calls == ["dental_tool"]
    asyncio.run(check())
