"""Compare full explanations and early relation decoding without changing model inputs.

Run validation first, freeze the implementation, then run the held-out test split.
Both modes use the same loaded Adapter, prompt, greedy decoder and batch order.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from scripts.compare_model_versions import adapter_metadata, release_gpu_cache, sha256
from scripts.compare_training_predictions import LABELS, score
from src.inference import (RELATION_MAX_NEW_TOKENS, InferenceConfig, RequirementClassifier,
                           save_predictions, validate_pair)
from src.train import read_gold


def cuda_synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_reset_peak():
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_gib():
    import torch
    return round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None


def checked_predictions(pairs, predictions, mode):
    """Preserve missing/error cases; empty explanations need an explicit fast contract."""
    if mode not in {"full", "fast"} or not isinstance(predictions, list):
        raise ValueError("Invalid mode or prediction container")
    expected = {pair["case_id"]: pair for pair in pairs}
    indexed = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError("Prediction must contain a case_id")
        case_id = row["case_id"]
        if case_id in indexed or case_id not in expected:
            raise ValueError("Duplicate or unknown prediction case_id")
        for field in ("req_a_id", "req_a_text", "req_b_id", "req_b_text"):
            if row.get(field) != expected[case_id][field]:
                raise ValueError(f"Prediction source mismatch: {case_id} {field}")
        indexed[case_id] = row
    result = []
    for pair in pairs:
        row = indexed.get(pair["case_id"])
        if row is None:
            result.append({**pair, "pred_label": "", "pred_reason": "", "pred_diff_conditions": "",
                           "status": "error", "error": "Missing prediction"})
            continue
        row = {**row, **pair}
        label_valid = isinstance(row.get("pred_label"), str) and row["pred_label"] in LABELS
        explanation_valid = all(isinstance(row.get(field), str) and bool(row[field].strip())
                                for field in ("pred_reason", "pred_diff_conditions"))
        prefix_valid = (row.get("response_format") == "label_prefix"
                        and row.get("explanation_status") == "not_requested"
                        and row.get("pred_reason") == "" and row.get("pred_diff_conditions") == "")
        fallback_valid = (row.get("relation_fallback") is True
                          and row.get("response_format") == "full_json"
                          and row.get("explanation_status") == "generated" and explanation_valid)
        contract_valid = explanation_valid if mode == "full" else prefix_valid or fallback_valid
        if row.get("status") != "ok" or not label_valid or not contract_valid:
            row.update(status="error", pred_label="",
                       error=row.get("error") or "Invalid label or explanation contract")
        result.append(row)
    return result


def _signature(rows):
    return [(row["case_id"], row["status"], row["pred_label"]) for row in rows]


def run_benchmark(pairs_path, adapter, output, *, split="val", batch_size=4,
                  max_input_tokens=4096, max_new_tokens=768, full_repeats=1, fast_repeats=3,
                  classifier_factory=None):
    pairs_path, adapter, output = map(Path, (pairs_path, adapter, output))
    if split not in {"val", "test"}:
        raise ValueError("Use validation first, then test; split must be val or test")
    if any(type(value) is not int or value < 1 for value in (
            batch_size, max_input_tokens, max_new_tokens, full_repeats, fast_repeats)):
        raise ValueError("Token limits, batch size and repeat counts must be positive integers")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {output}")
    dataset_hash = sha256(pairs_path)
    rows = read_gold(pairs_path)
    gold = [row for row in rows if row["split"] == split]
    pairs = [validate_pair(row) for row in gold]
    warmup_pair = validate_pair(next(row for row in rows if row["split"] == "train"))
    if sha256(pairs_path) != dataset_hash:
        raise ValueError("Dataset changed while reading")
    adapter_info = adapter_metadata(adapter)
    if adapter_info["training_dataset_sha256"] != dataset_hash:
        raise ValueError("Adapter training dataset does not match supplied data")
    implementation_paths = {
        "inference.py": Path(__file__).resolve().parents[1] / "src/inference.py",
        "benchmark_inference_modes.py": Path(__file__),
    }
    implementation = {name: sha256(path) for name, path in implementation_paths.items()}
    settings = {"do_sample": False, "batch_size": batch_size,
                "max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens,
                "relation_max_new_tokens": min(RELATION_MAX_NEW_TOKENS, max_new_tokens),
                "local_files_only": True, "full_repeats": full_repeats, "fast_repeats": fast_repeats,
                "warmup_case_id": warmup_pair["case_id"], "warmup_split": "train",
                "warmup_modes": ["full", "fast"], "phase_order": ["full", "fast"],
                "timing": "CUDA synchronized, excludes model loading, warmup and file serialization"}
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_snapshot.csv").write_bytes(pairs_path.read_bytes())
    if sha256(output / "source_snapshot.csv") != dataset_hash:
        raise ValueError("Dataset changed before snapshot")
    factory = classifier_factory or RequirementClassifier
    classifier = None
    all_predictions, results = {}, {}
    try:
        classifier = factory(InferenceConfig(adapter=adapter, local_files_only=True,
                                              max_input_tokens=max_input_tokens,
                                              max_new_tokens=max_new_tokens, seed=42))
        methods = {"full": classifier.predict_pairs, "fast": classifier.predict_relations}
        print("Warmup: one train pair in both modes (excluded from metrics/timing)", flush=True)
        for mode, method in methods.items():
            method([warmup_pair], batch_size=batch_size)
        cuda_synchronize()
        for mode, repeats in (("full", full_repeats), ("fast", fast_repeats)):
            runs, predictions = [], []
            for repeat in range(1, repeats + 1):
                cuda_synchronize()
                cuda_reset_peak()
                started = time.perf_counter()
                records = []
                for start in range(0, len(pairs), batch_size):
                    batch = pairs[start:start + batch_size]
                    raw = methods[mode](batch, batch_size=batch_size)
                    records.extend(checked_predictions(batch, raw, mode))
                cuda_synchronize()
                elapsed = time.perf_counter() - started
                metrics = score(gold, records)
                token_counts = [row.get("output_tokens") for row in records]
                tokens_known = all(type(count) is int and count >= 0 for count in token_counts)
                run = {
                    "repeat": repeat, "seconds": round(elapsed, 6), "metrics": metrics,
                    "output_tokens_total": sum(token_counts) if tokens_known else None,
                    "output_tokens_mean": statistics.mean(token_counts) if tokens_known else None,
                    "generated_explanations": sum(row["status"] == "ok" and bool(row.get("pred_reason"))
                                                  for row in records),
                    "omitted_explanations": sum(row["status"] == "ok"
                                                and row.get("explanation_status") == "not_requested"
                                                for row in records),
                    "fallback_cases": [row["case_id"] for row in records if row.get("relation_fallback")],
                    "peak_cuda_allocated_gib": cuda_peak_gib(),
                }
                metadata = {**classifier.metadata, "generation": {**settings, "mode": mode},
                            "split": split, "dataset_sha256": dataset_hash,
                            "adapter_weights_sha256": adapter_info["adapter_weights_sha256"],
                            "benchmark": run}
                save_predictions(records, metadata, output / f"{mode}_{repeat:02d}_pred.csv")
                runs.append(run)
                predictions.append(records)
                print(f"{mode} {repeat}/{repeats}: {elapsed:.3f}s; accuracy={metrics['accuracy']:.4f}; "
                      f"errors={len(metrics['errors'])}", flush=True)
                if sha256(pairs_path) != dataset_hash or adapter_metadata(adapter) != adapter_info:
                    raise ValueError("Dataset or Adapter changed during benchmark")
            results[mode] = {"metrics": runs[0]["metrics"],
                             "median_seconds": statistics.median(run["seconds"] for run in runs),
                             "runs": runs,
                             "predictions_identical_across_runs": all(
                                 _signature(rows) == _signature(predictions[0]) for rows in predictions)}
            all_predictions[mode] = predictions
    finally:
        # Bound methods also hold the classifier, so release them before clearing the GPU cache.
        methods = None
        method = None
        classifier = None
        release_gpu_cache()
    if {name: sha256(path) for name, path in implementation_paths.items()} != implementation:
        raise ValueError("Inference or benchmark implementation changed during benchmark")
    reference = _signature(all_predictions["full"][0])
    identical = all(_signature(predictions) == reference
                    for mode in all_predictions.values() for predictions in mode)
    complete = all(run["metrics"]["complete"] for mode in results.values() for run in mode["runs"])
    report = {
        "schema_version": "rfp-inference-speed-v1", "split": split,
        "dataset_sha256": dataset_hash, "dataset_path": str(pairs_path.resolve()),
        **adapter_info, "test_count": len(gold), "case_ids": [row["case_id"] for row in gold],
        "support": {label: sum(row["label"] == label for row in gold) for label in LABELS},
        "implementation_sha256": implementation, "settings": settings,
        **results, "labels_identical": identical, "all_predictions_complete": complete,
        "accuracy_preserved": identical and complete,
        "speedup": results["full"]["median_seconds"] / results["fast"]["median_seconds"],
        "case_comparison": [{"case_id": gold_row["case_id"], "expected": gold_row["label"],
                             "full_label": full["pred_label"], "full_status": full["status"],
                             "fast_label": fast["pred_label"], "fast_status": fast["status"]}
                            for gold_row, full, fast in zip(gold, all_predictions["full"][0],
                                                           all_predictions["fast"][0], strict=True)],
        "limitations": [
            "Fast mode validates the relation prefix; it does not generate or validate the full explanation JSON.",
            "Detailed pair explanations remain available in full mode; group summaries still use source requirements.",
            f"Full mode has {full_repeats} timing run(s); fast mode has {fast_repeats}. Fixed full-then-fast order may affect timing.",
            "Small non-document-held-out split with draft labels; agreement here does not prove universal accuracy preservation.",
            "No weights or prompts are tuned in this benchmark. Use validation before the final test run.",
        ],
    }
    (output / "performance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("split", "labels_identical", "accuracy_preserved", "speedup")},
                     ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--full-repeats", type=int, default=1)
    parser.add_argument("--fast-repeats", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(args.pairs, args.adapter, args.out, split=args.split, batch_size=args.batch_size,
                               max_new_tokens=args.max_new_tokens, max_input_tokens=args.max_input_tokens,
                               full_repeats=args.full_repeats, fast_repeats=args.fast_repeats)
        return 0 if result["accuracy_preserved"] else 1
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Inference benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
