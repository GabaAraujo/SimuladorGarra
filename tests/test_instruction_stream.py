"""Mede a continuidade do stream enquanto a interpretação e o log bloqueiam workers."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time

from backend.config import VisionConfig, load_backend_config
from backend.websocket_server import AsyncWSClient, GripperWebSocketServer


def _fps(timestamps: list[float], duration: float) -> float:
    return round(len(timestamps) / duration, 1) if duration > 0 else 0.0


def _p95_gap_ms(timestamps: list[float]) -> float:
    gaps = sorted((right - left) * 1000 for left, right in zip(timestamps, timestamps[1:]))
    if not gaps:
        return 0.0
    return round(gaps[min(len(gaps) - 1, math.ceil(0.95 * (len(gaps) - 1)))], 1)


def test_stream_continua_durante_interpretacao_e_log_lentos(tmp_path, monkeypatch):
    async def check():
        config = load_backend_config(
            host="127.0.0.1", port=0, fps=60, log_level="ERROR",
            vision=VisionConfig(enabled=True, enable_pick=True),
        )
        server = GripperWebSocketServer(config, logger=logging.getLogger("stream-audit-test"))
        server.nl_log_path = tmp_path / "nl_log.jsonl"
        # Não carregar pesos ausentes; o stream e o caminho HTTP continuam reais.
        server.vision.start = lambda: None
        port = await server.start()

        def slow_interpreter(*_args):
            time.sleep(0.6)
            return {
                "intent": "unknown", "class_name": None, "confidence": 0.0,
                "needs_clarification": True, "clarification": "Qual ferramenta?",
                "source": "gemini",
            }

        monkeypatch.setattr("backend.websocket_server.interpret", slow_interpreter)
        server._write_nl_log = lambda _record: time.sleep(0.15)

        timestamps: dict[str, list[float]] = {"baseline": [], "during": []}
        phase = ["baseline"]
        client = AsyncWSClient(f"ws://127.0.0.1:{port}/ws")
        reader_task = None
        try:
            await client.connect()

            async def collect_stream():
                while True:
                    message = await client.recv_json(timeout=2)
                    if message.get("type") == "stream":
                        timestamps[phase[0]].append(time.perf_counter())

            reader_task = asyncio.create_task(collect_stream())
            baseline_start = time.perf_counter()
            await asyncio.sleep(0.8)
            baseline_duration = time.perf_counter() - baseline_start

            phase[0] = "during"
            started = time.perf_counter()
            body = json.dumps({"text": "Pegue aquilo"}).encode("utf-8")
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"POST /api/instruction HTTP/1.1\r\n"
                + f"Host: 127.0.0.1:{port}\r\n".encode("ascii")
                + b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
                + body
            )
            await writer.drain()
            raw_response = await asyncio.wait_for(reader.read(), timeout=4)
            latency_ms = (time.perf_counter() - started) * 1000
            during_duration = latency_ms / 1000
            phase[0] = "after"
            status = int(raw_response.split(b" ", 2)[1])
            assert status == 200

            baseline_fps = _fps(timestamps["baseline"], baseline_duration)
            during_fps = _fps(timestamps["during"], during_duration)
            measurements = {
                "baseline_fps": baseline_fps,
                "during_fps": during_fps,
                "baseline_p95_gap_ms": _p95_gap_ms(timestamps["baseline"]),
                "during_p95_gap_ms": _p95_gap_ms(timestamps["during"]),
                "instruction_latency_ms": round(latency_ms, 1),
                "frames_baseline": len(timestamps["baseline"]),
                "frames_during": len(timestamps["during"]),
            }
            assert latency_ms >= 700
            assert baseline_fps > 30 and during_fps > 30
            assert measurements["during_p95_gap_ms"] < 100
            assert len(timestamps["during"]) >= 20
        finally:
            phase[0] = "after"
            if reader_task is not None:
                reader_task.cancel()
                await asyncio.gather(reader_task, return_exceptions=True)
            await client.close()
            await server.stop()
        return measurements

    measurements = asyncio.run(check())
    print("STREAM_AUDIT " + json.dumps(measurements, sort_keys=True))
