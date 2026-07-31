"""
Semantic cache threshold calibration.

A false semantic hit returns the wrong answer to a user, so the threshold cannot
be guessed. This script measures two similarity distributions over the workload:

  positives - paraphrases of the same base prompt (should hit)
  negatives - different base prompts (must never hit)

It then reports precision/recall across candidate thresholds so the configured
value is an evidence-based choice rather than a guess.

Run:  python -m bench.calibrate
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from gateway.embeddings import cosine, embed

from .workload import BASE_PROMPTS, _PARAPHRASE_TEMPLATES

RESULTS = Path(__file__).parent / "results"


def build_pairs() -> tuple[list[float], list[float]]:
    positives: list[float] = []
    negatives: list[float] = []

    vectors = {p: embed(p) for p in BASE_PROMPTS}

    # positives: each base prompt vs each of its paraphrases
    for base in BASE_PROMPTS:
        lowered = base[0].lower() + base[1:]
        for tpl in _PARAPHRASE_TEMPLATES:
            variant = tpl.format(p=lowered)
            positives.append(cosine(vectors[base], embed(variant)))

    # negatives: every distinct pair of different base prompts
    for a, b in itertools.combinations(BASE_PROMPTS, 2):
        negatives.append(cosine(vectors[a], vectors[b]))

    return positives, negatives


def evaluate(positives: list[float], negatives: list[float]) -> dict:
    rows = []
    for t in [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.88, 0.92, 0.96]:
        tp = sum(1 for s in positives if s >= t)
        fn = len(positives) - tp
        fp = sum(1 for s in negatives if s >= t)
        tn = len(negatives) - fp
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({
            "threshold": t,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })
    return {
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "positive_min": round(min(positives), 4),
        "positive_mean": round(sum(positives) / len(positives), 4),
        "negative_max": round(max(negatives), 4),
        "negative_mean": round(sum(negatives) / len(negatives), 4),
        "sweep": rows,
    }


def main() -> None:
    positives, negatives = build_pairs()
    report = evaluate(positives, negatives)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"positive pairs: {report['positive_pairs']}  "
          f"min={report['positive_min']} mean={report['positive_mean']}")
    print(f"negative pairs: {report['negative_pairs']}  "
          f"max={report['negative_max']} mean={report['negative_mean']}")
    print()
    print(f"{'thr':>5} {'prec':>7} {'recall':>7} {'FP':>4} {'FN':>4}")
    for r in report["sweep"]:
        print(f"{r['threshold']:>5} {r['precision']:>7} {r['recall']:>7} "
              f"{r['false_positives']:>4} {r['false_negatives']:>4}")
    print("\nwrote results/calibration.json")


if __name__ == "__main__":
    main()
