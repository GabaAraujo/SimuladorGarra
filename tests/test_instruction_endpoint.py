"""Testes HTTP e WebSocket para instruções em linguagem natural."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from backend import websocket_server
from backend.config import load_backend_config
from backend.main import build_arg_parser
from backend.websocket_server import GripperWebSocketServer


class CaptureWriter:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, data):
        self.data.extend(data)

    async def drain(self):
        return None

    def close(self):
        self.closed = True


def test_padrao_de_rede_usa_loopback_e_mesma_origem(monkeypatch):
    monkeypatch.delenv("SIM_HOST", raising=False)
    monkeypatch.delenv("SIM_CORS_ORIGIN", raising=False)
    args = build_arg_parser().parse_args([])
    config = load_backend_config(host=args.host)
    assert config.server.host == "127.0.0.1"
    assert config.server.cors_origins == "same-origin"


def make_server(tmp_path: Path, *, enabled=True, enable_pick=True, busy=False):
    server = GripperWebSocketServer.__new__(GripperWebSocketServer)
    server.config = SimpleNamespace(server=SimpleNamespace(cors_origins="*"))
    server.vision = SimpleNamespace(config=SimpleNamespace(
        allowed_classes=("dental_tool", "floss_holder"),
        enabled=enabled,
        enable_pick=enable_pick,
    ))
    server.vision_pick = SimpleNamespace(
        active=busy,
        status={"type": "vision_pick_status", "stage": "detecting" if busy else "idle",
                "operation_id": "already-running" if busy else None},
        task=None,
    )
    server.nl_log_path = tmp_path / "nl_log.jsonl"
    server.logger = SimpleNamespace(warning=lambda *_args: None)
    server._nl_inflight = False
    server._nl_last_request_at = 0.0
    server.env = SimpleNamespace(get_state_snapshot=lambda: {"pos": [0, 0, 0]})
    return server


def parsed_response(writer: CaptureWriter):
    head, body = bytes(writer.data).split(b"\r\n\r\n", 1)
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(body.decode("utf-8"))


async def post_instruction(server, payload, extra_headers=None):
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    reader = asyncio.StreamReader()
    reader.feed_data(raw)
    reader.feed_eof()
    writer = CaptureWriter()
    headers = {
        "content-length": str(len(raw)),
    }
    if extra_headers:
        headers.update(extra_headers)
    await server._handle_http(reader, writer, "POST", "/api/instruction", headers)
    return parsed_response(writer)


def test_http_aceita_tarefa_e_registra_resultado_final(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path)
        calls = []

        def finish():
            async def operation():
                server.vision_pick.status.update(stage="completed", attempt=1, max_attempts=2,
                                                 recovered=False)
            return asyncio.create_task(operation())

        server.execute_command = lambda payload: (
            calls.append(payload),
            server.vision_pick.status.update(stage="detecting", operation_id="new-operation"),
            setattr(server.vision_pick, "task", finish()),
        )
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "fetch", "class_name": "dental_tool", "confidence": 0.9,
            "needs_clarification": False, "clarification": None, "source": "keyword",
        })

        status, body = await post_instruction(server, {"text": "pegue o instrumento"})
        await asyncio.sleep(0)
        await server.vision_pick.task
        for _ in range(20):
            if server.nl_log_path.exists():
                break
            await asyncio.sleep(0.01)

        assert status == 200
        assert body["accepted"] is True
        assert body["task"]["class_name"] == "dental_tool"
        assert body["status"]["stage"] == "detecting"
        assert calls == [{"command": "vision_pick", "class_name": "dental_tool"}]
        record = json.loads(server.nl_log_path.read_text(encoding="utf-8"))
        assert record["result"]["stage"] == "completed"
        assert record["duration_ms"] >= 0

    asyncio.run(check())


@pytest.mark.parametrize("task", [
    {"intent": "unknown", "class_name": None, "confidence": 0.0,
     "needs_clarification": True, "clarification": "Qual ferramenta?", "source": "keyword"},
])
def test_http_pede_esclarecimento_sem_disparar_captura(tmp_path, monkeypatch, task):
    async def check():
        server = make_server(tmp_path)
        calls = []
        server.execute_command = lambda payload: calls.append(payload)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: task)

        status, body = await post_instruction(server, {"text": "pega aquilo"})

        assert status == 200 and body["accepted"] is False
        assert body["task"]["needs_clarification"] is True
        assert body["status"]["stage"] == "idle"
        assert calls == []
        assert json.loads(server.nl_log_path.read_text(encoding="utf-8"))["result"]["stage"] == "clarification"

    asyncio.run(check())


def test_http_recusa_quando_ocupado(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path, busy=True)
        calls = []
        server.execute_command = lambda payload: calls.append(payload)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "fetch", "class_name": "dental_tool", "confidence": 0.9,
            "needs_clarification": False, "clarification": None, "source": "gemini",
        })

        status, body = await post_instruction(server, {"text": "instrumento dental"})

        assert status == 409 and body["accepted"] is False
        assert "em andamento" in body["error"]
        assert body["status"]["stage"] == "detecting"
        assert calls == []

    asyncio.run(check())


def test_http_recusa_com_visao_desabilitada(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path, enabled=False)
        calls = []
        server.execute_command = lambda payload: calls.append(payload)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "fetch", "class_name": "dental_tool", "confidence": 0.9,
            "needs_clarification": False, "clarification": None, "source": "gemini",
        })

        status, body = await post_instruction(server, {"text": "instrumento dental"})

        assert status == 503 and body["accepted"] is False
        assert "desabilitada" in body["error"]
        assert calls == []

    asyncio.run(check())


@pytest.mark.parametrize("text", ["", "x" * 301])
def test_http_rejeita_texto_vazio_ou_maior_que_limite(tmp_path, monkeypatch, text):
    async def check():
        server = make_server(tmp_path)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "unknown", "class_name": None, "confidence": 0.0,
            "needs_clarification": True, "clarification": "Esclareça.", "source": "keyword",
        })
        status, body = await post_instruction(server, {"text": text})
        assert status == 400 and body["accepted"] is False
        assert body["error"]

    asyncio.run(check())


def test_http_limita_tamanho_do_corpo_e_registra_recusa(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "unknown", "class_name": None, "confidence": 0.0,
            "needs_clarification": True, "clarification": "Envie uma instrução menor.", "source": "keyword",
        })
        reader = asyncio.StreamReader()
        reader.feed_eof()
        writer = CaptureWriter()
        await server._handle_http(reader, writer, "POST", "/api/instruction", {
            "content-length": "8193",
        })
        status, body = parsed_response(writer)
        assert status == 413 and body["accepted"] is False
        record = json.loads(server.nl_log_path.read_text(encoding="utf-8"))
        assert record["result"]["stage"] == "rejected"

    asyncio.run(check())


def test_http_aplica_limite_de_taxa_sem_consultar_interpretador(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path)
        server._nl_last_request_at = asyncio.get_running_loop().time()
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: pytest.fail("não deve interpretar"))
        status, body = await post_instruction(server, {"text": "Pegue a pinça"})
        assert status == 429
        assert body["accepted"] is False
        assert body["task"]["source"] == "keyword"

    asyncio.run(check())


def test_http_limita_instrucoes_concorrentes(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path)
        calls = []

        def slow_interpreter(*_args):
            import time
            calls.append("interpret")
            time.sleep(0.15)
            return {
                "intent": "unknown", "class_name": None, "confidence": 0.0,
                "needs_clarification": True, "clarification": "Qual ferramenta?", "source": "keyword",
            }

        monkeypatch.setattr(websocket_server, "interpret", slow_interpreter)
        first = asyncio.create_task(server._handle_instruction("Pegue aquilo"))
        await asyncio.sleep(0.02)
        second_status, second = await server._handle_instruction("Pegue outra coisa")
        first_status, _first = await first
        assert first_status == 200
        assert second_status == 429
        assert second["accepted"] is False
        assert calls == ["interpret"]

    asyncio.run(check())


def test_http_rejeita_corpo_truncado(tmp_path):
    async def check():
        server = make_server(tmp_path)
        reader = asyncio.StreamReader()
        reader.feed_data(b'{"text":')
        reader.feed_eof()
        writer = CaptureWriter()
        await server._handle_http(reader, writer, "POST", "/api/instruction", {"content-length": "30"})
        status, body = parsed_response(writer)
        assert status == 400
        assert "incompleto" in body["error"]

    asyncio.run(check())


def test_http_bloqueia_origem_cruzada_por_padrao(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path)
        server.config.server.cors_origins = "same-origin"
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: pytest.fail("pedido deve ser bloqueado"))
        status, body = await post_instruction(server, {"text": "Pegue a pinça"}, {
            "host": "127.0.0.1:8000",
            "origin": "https://site-malicioso.example",
        })
        assert status == 403
        assert "origem" in body["error"].casefold()

    asyncio.run(check())


def test_frontend_recusa_escape_por_prefixo_de_diretorio(tmp_path):
    async def check():
        frontend = tmp_path / "frontend"
        sibling = tmp_path / "frontend_private"
        frontend.mkdir()
        sibling.mkdir()
        (sibling / "secret.txt").write_text("SHOULD_NOT_SERVE_SENTINEL", encoding="utf-8")
        server = GripperWebSocketServer.__new__(GripperWebSocketServer)
        server.config = SimpleNamespace(
            frontend_dir=frontend,
            assets_dir=tmp_path / "assets",
            server=SimpleNamespace(cors_origins="same-origin"),
        )
        writer = CaptureWriter()
        await server._serve_static_frontend(writer, "/../frontend_private/secret.txt")
        status, body = parsed_response(writer)
        assert status == 404
        assert "SHOULD_NOT_SERVE_SENTINEL" not in json.dumps(body)

    asyncio.run(check())


def test_log_de_instrucao_remove_segredos(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path)
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-value")
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "unknown", "class_name": None, "confidence": 0.0,
            "needs_clarification": True, "clarification": f"token:abc123 {secret}", "source": "keyword",
        })
        secret = "test-gemini-key-value"
        await post_instruction(server, {"text": f"token:abc123 {secret}"})
        raw_log = server.nl_log_path.read_text(encoding="utf-8")
        assert secret not in raw_log
        assert "abc123" not in raw_log
        assert "[REDACTED]" in raw_log

    asyncio.run(check())


def test_redator_mascara_chave_google_sem_expo_la():
    key = "AI" + "za" + "1" * 24
    assert websocket_server._redact_instruction(key) == "[REDACTED]"


def test_websocket_recusa_frame_maior_que_limite():
    async def check():
        reader = asyncio.StreamReader()
        reader.feed_data(b"\x81\x7f" + struct.pack("!Q", websocket_server.MAX_WS_MESSAGE_SIZE + 1))
        with pytest.raises(websocket_server.WebSocketProtocolError) as exc:
            await websocket_server.read_ws_frame(reader)
        assert exc.value.close_code == 1009

    asyncio.run(check())


def test_websocket_retorna_erro_de_entrada_para_utf8_invalido():
    async def check():
        server = GripperWebSocketServer.__new__(GripperWebSocketServer)
        server.logger = SimpleNamespace(warning=lambda *_args: None)
        server.env = SimpleNamespace(get_state_snapshot=lambda: {"pos": [0, 0, 0]})
        sent = []

        async def send_json(data):
            sent.append(data)

        session = SimpleNamespace(peer="test", send_json=send_json)
        await server._process_ws_message(session, b"\xff")
        assert sent[0]["type"] == "error"
        assert "Internal server error" not in sent[0]["error"]

    asyncio.run(check())


def test_websocket_aceita_mensagem_instruction(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path)
        calls = []
        server.execute_command = lambda payload: calls.append(payload)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "unknown", "class_name": None, "confidence": 0.0,
            "needs_clarification": True, "clarification": "Qual ferramenta?", "source": "keyword",
        })
        sent = []
        session = SimpleNamespace(peer="test", send_json=lambda data: _append(sent, data))

        await server._process_ws_message(session, b'{"type":"instruction","text":"pega aquilo"}')

        assert sent[0]["type"] == "instruction_result"
        assert sent[0]["accepted"] is False
        assert sent[0]["task"]["needs_clarification"] is True
        assert calls == []

    asyncio.run(check())


async def _append(items, value):
    items.append(value)
