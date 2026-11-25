"""Validação offline do dataset e das métricas de linguagem natural."""
from __future__ import annotations

import json
from pathlib import Path

from backend import nl_interpreter
from scripts.evaluate_nl import build_parser, evaluate_dataset, run


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_tem_splits_de_desenvolvimento_e_teste():
    samples = json.loads((ROOT / "datasets/nl_commands.json").read_text(encoding="utf-8"))
    assert len(samples) == 56
    assert {sample["split"] for sample in samples} == {"dev", "test"}
    assert sum(sample["split"] == "dev" for sample in samples) == 30
    assert sum(sample["split"] == "test" for sample in samples) == 26
    assert all(sample["revisar"] is True for sample in samples)
    split_membership = {}
    for sample in samples:
        split_membership.setdefault(sample["text"].strip().casefold(), set()).add(sample["split"])
    assert all(len(splits) == 1 for splits in split_membership.values())
    assert {"formal", "informal", "sinonimo", "ambigua", "invalida", "erro_digitacao"}.issubset(
        {sample["category"] for sample in samples}
    )
    assert all(isinstance(sample["expected_clarification"], bool) for sample in samples)


def test_avaliacao_keyword_usa_fonte_local():
    samples = json.loads((ROOT / "datasets/nl_commands.json").read_text(encoding="utf-8"))
    metrics = evaluate_dataset(samples, "keyword", ("dental_tool", "floss_holder"))
    assert metrics["sample_count"] == 56
    assert metrics["per_split"]["dev"]["sample_count"] == 30
    assert metrics["per_split"]["test"]["sample_count"] == 26
    assert {row["split"] for row in metrics["predictions"]} == {"dev", "test"}
    assert set(metrics["per_category"]) == {
        "formal", "informal", "sinonimo", "ambigua", "invalida", "erro_digitacao",
        "negacao", "injecao", "comando_fisico", "frase_ruidosa",
    }
    assert all(row["source"] == "keyword" for row in metrics["predictions"])
    assert metrics["latency_ms_p50"] is not None and metrics["latency_ms_p95"] is not None


def test_comando_keyword_gera_json_sem_rede(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("GEMINI_API_KEY", "chave-que-nao-deve-ser-usada")
    monkeypatch.setattr(nl_interpreter.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: pytest.fail("modo keyword tentou acessar a rede"))
    args = build_parser().parse_args(["--mode", "keyword", "--output-dir", str(tmp_path)])

    report = run(args)

    assert report["metrics"]["sample_count"] == 56
    assert set(report["metrics"]["per_split"]) == {"dev", "test"}
    assert Path(report["output"]).is_file()
