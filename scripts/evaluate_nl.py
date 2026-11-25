#!/usr/bin/env python3
"""Avalia interpretação natural de instruções e, opcionalmente, a captura MuJoCo."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import VisionConfig  # noqa: E402

DEFAULT_VISION_CONFIG = VisionConfig()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return round(ordered[index], 3)


def _interpret(text: str, allowed_classes: tuple[str, ...], mode: str) -> dict[str, Any]:
    from backend.nl_interpreter import interpret, interpret_keywords

    if mode == "keyword":
        return interpret_keywords(text, allowed_classes)
    return interpret(text, allowed_classes)


def evaluate_dataset(samples: list[dict[str, Any]], mode: str,
                     allowed_classes: tuple[str, ...]) -> dict[str, Any]:
    rows = []

    for sample in samples:
        started = time.perf_counter()
        prediction = _interpret(sample["text"], allowed_classes, mode)
        latency_ms = (time.perf_counter() - started) * 1000.0
        predicted_class = prediction["class_name"]
        predicted_clarification = prediction["needs_clarification"]
        expected_class = sample["expected_class"]
        expected_clarification = sample["expected_clarification"]
        correct = (predicted_class == expected_class
                   and predicted_clarification == expected_clarification)
        rows.append({
            "text": sample["text"],
            "expected_class": expected_class,
            "predicted_class": predicted_class,
            "expected_clarification": expected_clarification,
            "predicted_clarification": predicted_clarification,
            "correct": correct,
            "confidence": prediction["confidence"],
            "source": prediction["source"],
            "category": sample["category"],
            "split": sample["split"],
            "revisar": bool(sample.get("revisar", False)),
            "latency_ms": round(latency_ms, 3),
        })

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        category_totals: dict[str, dict[str, int]] = {}
        tp = fp = fn = correct_count = 0
        for row in group:
            correct_count += int(row["correct"])
            if row["expected_clarification"] and row["predicted_clarification"]:
                tp += 1
            elif not row["expected_clarification"] and row["predicted_clarification"]:
                fp += 1
            elif row["expected_clarification"] and not row["predicted_clarification"]:
                fn += 1
            totals = category_totals.setdefault(row["category"], {"total": 0, "correct": 0})
            totals["total"] += 1
            totals["correct"] += int(row["correct"])
        latencies = [row["latency_ms"] for row in group]
        count = len(group)
        return {
            "sample_count": count,
            "accuracy": round(correct_count / max(1, count), 4),
            "correct_count": correct_count,
            "review_required_count": sum(bool(row["revisar"]) for row in group),
            "per_category": {
                name: {
                    "total": values["total"],
                    "correct": values["correct"],
                    "accuracy": round(values["correct"] / max(1, values["total"]), 4),
                }
                for name, values in sorted(category_totals.items())
            },
            "clarification_precision": round(tp / max(1, tp + fp), 4),
            "clarification_recall": round(tp / max(1, tp + fn), 4),
            "latency_ms_p50": _percentile(latencies, 0.50),
            "latency_ms_p95": _percentile(latencies, 0.95),
        }

    metrics = summarize(rows)
    metrics["per_split"] = {
        split: summarize([row for row in rows if row["split"] == split])
        for split in ("dev", "test")
    }
    metrics["predictions"] = rows
    return metrics


async def evaluate_e2e(samples: list[dict[str, Any]], mode: str,
                       allowed_classes: tuple[str, ...], args: argparse.Namespace) -> dict[str, Any]:
    from backend.config import VisionConfig, load_backend_config
    from backend.websocket_server import GripperWebSocketServer

    config = load_backend_config(
        host="127.0.0.1",
        port=0,
        log_level="WARNING",
        vision=VisionConfig(
            enabled=True,
            enable_pick=True,
            model_path=args.vision_model,
            calibration_path=args.calibration,
            device=args.device,
            max_age_ms=args.max_age_ms,
            pick_max_attempts=args.pick_max_attempts,
        ),
    )
    server = GripperWebSocketServer(config)
    trials = []
    eligible = [item for item in samples
                if item["expected_class"] in allowed_classes
                and not item["expected_clarification"]]
    if args.e2e > len(eligible):
        raise ValueError(f"--e2e {args.e2e} excede as {len(eligible)} frases válidas do dataset.")

    await server.start()
    try:
        for sample in eligible[:args.e2e]:
            server.execute_command({"command": "reset"})
            started = time.perf_counter()
            task = await asyncio.to_thread(_interpret, sample["text"], allowed_classes, mode)
            accepted = task["intent"] == "fetch" and not task["needs_clarification"]
            status = dict(server.vision_pick.status)
            if accepted:
                server.execute_command({"command": "vision_pick", "class_name": task["class_name"]})
                operation = server.vision_pick.task
                try:
                    await asyncio.wait_for(asyncio.shield(operation), timeout=180)
                except asyncio.TimeoutError:
                    server.vision_pick.cancel()
                    await asyncio.gather(operation, return_exceptions=True)
                status = dict(server.vision_pick.status)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            trials.append({
                "text": sample["text"],
                "expected_class": sample["expected_class"],
                "task": task,
                "accepted": accepted,
                "stage": status.get("stage"),
                "attempts": status.get("attempt", 0),
                "recovered": status.get("recovered", False),
                "duration_ms": round(elapsed_ms, 2),
            })
    finally:
        await server.stop()

    durations = [trial["duration_ms"] for trial in trials]
    attempts = [trial["attempts"] for trial in trials]
    successes = sum(trial["stage"] == "completed" for trial in trials)
    return {
        "request_count": len(trials),
        "success_count": successes,
        "success_rate": round(successes / max(1, len(trials)), 4),
        "attempts_average": round(statistics.mean(attempts), 3) if attempts else 0,
        "duration_ms_p50": _percentile(durations, 0.50),
        "duration_ms_p95": _percentile(durations, 0.95),
        "trials": trials,
    }


def _validate_samples(samples: Any) -> list[dict[str, Any]]:
    if not isinstance(samples, list):
        raise ValueError("O dataset deve ser uma lista JSON.")
    required = {"text", "expected_class", "expected_clarification", "category", "split"}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not required.issubset(sample):
            raise ValueError(f"A amostra {index + 1} não contém os campos obrigatórios.")
        if not isinstance(sample["text"], str) or not isinstance(sample["expected_clarification"], bool):
            raise ValueError(f"Tipos inválidos na amostra {index + 1}.")
        expected_class = sample["expected_class"]
        if expected_class is not None and not isinstance(expected_class, str):
            raise ValueError(f"Classe esperada inválida na amostra {index + 1}.")
        if sample["expected_clarification"] != (expected_class is None):
            raise ValueError(f"Classe e esclarecimento inconsistentes na amostra {index + 1}.")
        if not isinstance(sample["category"], str):
            raise ValueError(f"Categoria inválida na amostra {index + 1}.")
        if "revisar" in sample and not isinstance(sample["revisar"], bool):
            raise ValueError(f"Marcador revisar inválido na amostra {index + 1}.")
        if sample["split"] not in ("dev", "test"):
            raise ValueError(f"Split inválido na amostra {index + 1}; use dev ou test.")
    return samples


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "gemini" and not os.getenv("GEMINI_API_KEY", "").strip():
        raise ValueError("Defina GEMINI_API_KEY para avaliar no modo Gemini.")
    if args.e2e < 0:
        raise ValueError("--e2e deve ser zero ou positivo.")

    samples = _validate_samples(json.loads(args.dataset.read_text(encoding="utf-8")))
    allowed_classes = tuple(DEFAULT_VISION_CONFIG.allowed_classes)
    metrics = evaluate_dataset(samples, args.mode, allowed_classes)
    e2e_metrics = (asyncio.run(evaluate_e2e(samples, args.mode, allowed_classes, args))
                   if args.e2e else None)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"nl_eval_{timestamp}.json"
    try:
        dataset_name = args.dataset.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        dataset_name = args.dataset.name
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "dataset": dataset_name,
        "metrics": metrics,
        "e2e": e2e_metrics,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["output"] = str(output)
    return report


def print_report(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    print(f"Avaliação NL ({report['mode']}) — {metrics['sample_count']} frases")
    print(f"Acurácia geral: {metrics['accuracy']:.1%} ({metrics['correct_count']}/{metrics['sample_count']})")
    print(f"Amostras marcadas para revisão manual: {metrics['review_required_count']}")
    print(f"Precisão de esclarecimento: {metrics['clarification_precision']:.1%}")
    print(f"Recall de esclarecimento: {metrics['clarification_recall']:.1%}")
    print(f"Latência p50/p95: {metrics['latency_ms_p50']} / {metrics['latency_ms_p95']} ms")
    print("\nSplit  Acertos  Total  Acurácia  Recall esclarecer")
    for split, values in metrics["per_split"].items():
        print(f"{split:<6} {values['correct_count']:>7}  {values['sample_count']:>5}  "
              f"{values['accuracy']:>8.1%}  {values['clarification_recall']:>17.1%}")
    print("\nCategoria          Acertos  Total  Acurácia")
    print("-----------------  -------  -----  --------")
    for category, values in metrics["per_category"].items():
        print(f"{category:<17}  {values['correct']:>7}  {values['total']:>5}  {values['accuracy']:>7.1%}")
    if report["e2e"] is not None:
        e2e = report["e2e"]
        print(f"\nPonta a ponta: {e2e['success_count']}/{e2e['request_count']} concluídas; "
              f"média de {e2e['attempts_average']} tentativas; "
              f"duração p50/p95 {e2e['duration_ms_p50']} / {e2e['duration_ms_p95']} ms")
    print(f"Relatório: {report['output']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("keyword", "gemini"), default="keyword")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/nl_commands.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--e2e", type=int, default=0, metavar="N",
                        help="Executa N capturas no simulador com os exemplos rotulados.")
    parser.add_argument("--vision-model", type=Path, default=DEFAULT_VISION_CONFIG.model_path)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_VISION_CONFIG.calibration_path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-age-ms", type=float, default=300)
    parser.add_argument("--pick-max-attempts", type=int, default=2)
    return parser

if __name__ == "__main__":
    try:
        parsed = run(build_parser().parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print_report(parsed)
