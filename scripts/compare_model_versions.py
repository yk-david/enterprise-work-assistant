"""Evaluate previous/current adapters on the same latest test rows and greedy settings.

Run as python -m scripts.compare_model_versions. This never trains or selects a checkpoint.
"""

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

from scripts.compare_training_predictions import LABELS, score
from src.inference import InferenceConfig, RequirementClassifier, save_predictions, validate_pair
from src.train import read_gold


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_metadata(path):
    if path is None:
        return {"adapter_path": None, "training_dataset_sha256": None, "adapter_weights_sha256": None}
    path = path.resolve()
    info = json.loads((path / "training_info.json").read_text(encoding="utf-8"))
    digest = info.get("dataset_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Adapter has no valid training dataset SHA256: {path}")
    return {"adapter_path": str(path), "training_dataset_sha256": digest,
            "adapter_weights_sha256": sha256(path / "adapter_model.safetensors")}


def normalize_predictions(pairs, predictions):
    """Keep missing/invalid rows in the denominator; reject mismatched source evidence."""
    if not isinstance(predictions, list):
        raise ValueError("Classifier predictions must be a list")
    by_id = {}
    for prediction in predictions:
        if not isinstance(prediction, dict) or not isinstance(prediction.get("case_id"), str):
            raise ValueError("Prediction has no case ID")
        if prediction["case_id"] in by_id:
            raise ValueError("Duplicate prediction case ID")
        by_id[prediction["case_id"]] = prediction
    if set(by_id) - {pair["case_id"] for pair in pairs}:
        raise ValueError("Unexpected prediction case ID")
    rows = []
    for pair in pairs:
        prediction = by_id.get(pair["case_id"])
        if prediction is None:
            rows.append({**pair, "pred_label": "", "pred_reason": "", "pred_diff_conditions": "",
                         "status": "error", "error": "Missing model prediction"})
            continue
        for field in ("req_a_id", "req_b_id", "req_a_text", "req_b_text"):
            if prediction.get(field) != pair[field]:
                raise ValueError(f"Prediction source mismatch: {pair['case_id']} {field}")
        prediction = {**prediction, **pair}
        label = prediction.get("pred_label")
        if (prediction.get("status") != "ok" or not isinstance(label, str) or label not in LABELS
                or any(not isinstance(prediction.get(field), str) or not prediction[field].strip()
                       for field in ("pred_reason", "pred_diff_conditions"))):
            prediction.update(status="error", pred_label="",
                              error=prediction.get("error") or "Invalid prediction label/reason/diff_conditions")
        rows.append(prediction)
    return rows


def release_gpu_cache():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_comparison(pairs_path, previous, current, output, *, include_base=False, batch_size=4,
                   max_input_tokens=4096, max_new_tokens=384, classifier_factory=None):
    paths = [Path(path) for path in (pairs_path, previous, current, output)]
    pairs_path, previous, current, output = paths
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output}")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    dataset_hash = sha256(pairs_path)
    gold = [row for row in read_gold(pairs_path) if row["split"] == "test"]
    if not gold:
        raise ValueError("No test data")
    if sha256(pairs_path) != dataset_hash:
        raise ValueError("Dataset changed while reading")
    variants = {**({"base": None} if include_base else {}), "previous": previous, "current": current}
    adapter_infos = {name: adapter_metadata(path) for name, path in variants.items()}
    if adapter_infos["current"]["training_dataset_sha256"] != dataset_hash:
        raise ValueError("Current adapter was not trained on the supplied latest dataset")
    pairs = [validate_pair(row) for row in gold]  # strips label/reason/diff_conditions before model input
    generation = {"do_sample": False, "batch_size": batch_size,
                  "max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens,
                  "seed": 42, "local_files_only": True}
    factory = classifier_factory or RequirementClassifier
    output.mkdir(parents=True, exist_ok=False)
    (output / "gold_snapshot.csv").write_bytes(pairs_path.read_bytes())
    if sha256(output / "gold_snapshot.csv") != dataset_hash:
        raise ValueError("Dataset changed before snapshot")
    model_results, prediction_sets = {}, {}
    for name, adapter in variants.items():
        classifier = None
        started = time.perf_counter()
        print(f"Loading {name}: {adapter or 'base model'}", flush=True)
        try:
            classifier = factory(InferenceConfig(
                adapter=adapter, local_files_only=True, max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens, seed=42))
            inference_started = time.perf_counter()
            records = []
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                raw = classifier.predict_pairs(batch, batch_size=batch_size)
                records.extend(normalize_predictions(batch, raw))
                print(f"{name}: {len(records)}/{len(pairs)}", flush=True)
            inference_seconds = round(time.perf_counter() - inference_started, 3)
            metadata = {**classifier.metadata, "generation": generation,
                        "evaluation_variant": name, "input_path": str(pairs_path.resolve()),
                        "dataset_sha256": dataset_hash, "split": "test"}
            save_predictions(records, metadata, output / f"{name}_test_pred.csv")
            metrics = score(gold, records)
            model_results[name] = {**adapter_infos[name], "metrics": metrics,
                                   "inference_seconds": inference_seconds,
                                   "total_seconds": round(time.perf_counter() - started, 3),
                                   "predictions_csv": str((output / f"{name}_test_pred.csv").resolve())}
            prediction_sets[name] = {row["case_id"]: row for row in records}
        finally:
            classifier = None
            release_gpu_cache()
        if sha256(pairs_path) != dataset_hash:
            raise ValueError("Dataset changed during evaluation; results cannot be compared")
        if adapter_metadata(adapter) != adapter_infos[name]:
            raise ValueError(f"Adapter changed during evaluation: {name}")
    cases = []
    for row in gold:
        case = {"case_id": row["case_id"], "expected": row["label"]}
        for name, predictions in prediction_sets.items():
            pred = predictions[row["case_id"]]
            case[name] = {"label": pred["pred_label"], "status": pred["status"],
                          "correct": pred["status"] == "ok" and pred["pred_label"] == row["label"]}
        case["prediction_changed"] = case["previous"]["label"] != case["current"]["label"]
        cases.append(case)
    support = {label: sum(row["label"] == label for row in gold) for label in LABELS}
    result = {
        "schema_version": "rfp-model-comparison-v1",
        "dataset": {"sha256": dataset_hash, "path": str(pairs_path.resolve()),
                    "test_case_ids": [row["case_id"] for row in gold],
                    "test_support": support, "test_count": len(gold)},
        "generation": generation, "models": model_results,
        "delta_current_minus_previous": {
            metric: model_results["current"]["metrics"][metric] - model_results["previous"]["metrics"][metric]
            for metric in ("accuracy", "macro_f1")},
        "case_comparison": cases,
        "annotation_status": "assistant_reviewed_pending_team_adjudication",
        "limitations": [
            f"Small test set: {len(gold)} pairs; class support: {support}. Scores are descriptive pilot results.",
            "Labels await team adjudication. Reasons and condition faithfulness are not scored by classification metrics.",
            "The split is not document-held-out; it does not establish generalization to unseen RFPs.",
            "Previous sampled-decoding reports are not directly comparable to this greedy-decoding evaluation.",
            "Test data was used only for final comparison, not model training or checkpoint selection.",
        ],
        "all_predictions_complete": all(model["metrics"]["complete"] for model in model_results.values()),
    }
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": {name: model["metrics"] for name, model in model_results.items()},
                      "delta": result["delta_current_minus_previous"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args(argv)
    try:
        result = run_comparison(args.pairs, args.previous, args.current, args.out,
                                include_base=args.include_base, batch_size=args.batch_size,
                                max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens)
        return 0 if result["all_predictions_complete"] else 1
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Model comparison failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
