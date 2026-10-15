"""Compare a verified duration-only continuation without loading any pixels."""

import hashlib
import json
import math
from pathlib import Path
from statistics import mean


METRICS = ("mae", "pearson", "ssim")
CONTINUATION_KEYS = {"steps", "resume_from_step", "resume_checkpoint_sha256"}


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field_map(records):
    result = {}
    for row in records:
        source = row["source_id"]
        if source in result:
            raise ValueError("duplicate validation source")
        for metric in METRICS:
            value = row[metric]
            if value is None and metric == "pearson":
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("non-finite or missing validation metric")
        if row["mae"] < 0:
            raise ValueError("negative validation MAE")
        result[source] = row
    if not result:
        raise ValueError("validation fields are required")
    return result


def _verify_summary(summary, records):
    groups = {r["group_id"] for r in records}
    if (summary["field_count"] != len(records) or summary["group_count"] != len(groups) or
            summary["undefined_pearson_fields"] != sum(r["pearson"] is None for r in records)):
        raise ValueError("validation summary counts disagree with fields")
    for metric in METRICS:
        per_group = []
        for group in groups:
            values = [r[metric] for r in records if r["group_id"] == group and r[metric] is not None]
            if values:
                per_group.append(mean(values))
        expected = mean(per_group) if per_group else None
        actual = summary[metric]
        if expected is None:
            if actual is not None:
                raise ValueError("undefined validation summary was replaced")
        elif (not isinstance(actual, (int, float)) or not math.isfinite(actual) or
              not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9)):
            raise ValueError("validation summary disagrees with group-balanced fields")


def _read_run(directory):
    run_path = directory / "training.json"
    run = json.loads(run_path.read_text())
    if run["test_pixels_loaded"] is not False:
        raise ValueError("duration comparison requires unopened test pixels")
    for name in ("best", "last"):
        if _digest(directory / f"{name}-model.pt") != run[f"{name}_checkpoint_sha256"]:
            raise ValueError(f"{name} checkpoint checksum mismatch")
    best = _field_map(run["best_fields"])
    constant = _field_map(run["constant_baseline_fields"])
    if best.keys() != constant.keys():
        raise ValueError("model and constant validation sources disagree")
    for source in best:
        if any(best[source][key] != constant[source][key] for key in ("group_id", "z")):
            raise ValueError("model and constant validation identities disagree")
    _verify_summary(run["best_model"], list(best.values()))
    _verify_summary(run["constant_baseline"], list(constant.values()))
    if (run["validation_fields"] != len(best) or not isinstance(run["best_step"], int) or
            not 0 <= run["best_step"] <= run["config"]["steps"]):
        raise ValueError("invalid validation count or best step")
    history = json.loads((directory / "history.json").read_text())
    if not history or history[-1]["step"] != run["config"]["steps"]:
        raise ValueError("run history does not reach the reported final step")
    return run, best, constant, history, _digest(run_path)


def _delta(before, after):
    return {metric: after[metric] - before[metric]
            if before[metric] is not None and after[metric] is not None else None
            for metric in METRICS}


def compare_duration(baseline_dir: Path, candidate_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise ValueError("comparison output already exists")
    before, before_fields, before_constant, old_history, before_sha = _read_run(baseline_dir)
    after, after_fields, after_constant, new_history, after_sha = _read_run(candidate_dir)
    a, b = before["config"], after["config"]
    # Fail closed on any extra config change, including environment, device,
    # validation cadence, source revision, depth policy and data fingerprints.
    for key in sorted((a.keys() | b.keys()) - CONTINUATION_KEYS):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"not a duration-only comparison: {key} differs")
    if (b["steps"] <= a["steps"] or b.get("resume_from_step") != a["steps"] or
            b.get("resume_checkpoint_sha256") != before["last_checkpoint_sha256"] or
            after["steps_this_run"] != b["steps"] - a["steps"]):
        raise ValueError("candidate must directly continue the baseline last checkpoint")
    if new_history[:len(old_history)] != old_history:
        raise ValueError("candidate does not preserve the baseline history")
    if before_fields.keys() != after_fields.keys():
        raise ValueError("validation source identities changed")
    for source in before_fields:
        if any(before_fields[source][key] != after_fields[source][key] for key in ("group_id", "z")):
            raise ValueError("validation group or depth changed")
    for key in ("train_fields", "validation_fields", "constant_value", "constant_definition"):
        if before[key] != after[key]:
            raise ValueError(f"baseline contract changed: {key}")
    if before_constant != after_constant:
        raise ValueError("constant baseline validation results changed")
    difference = _delta(before["best_model"], after["best_model"])
    initial_mae = before["best_model"]["mae"]
    result = {
        "purpose": "verified duration-only development comparison; not test evaluation",
        "test_pixels_loaded": False,
        "baseline_report_sha256": before_sha, "candidate_report_sha256": after_sha,
        "prepared_metadata_sha256": a["prepared_metadata_sha256"],
        "pilot_plan_sha256": a["pilot_plan_sha256"],
        "baseline_total_steps": a["steps"], "candidate_total_steps": b["steps"],
        "baseline_best_step": before["best_step"], "candidate_best_step": after["best_step"],
        "baseline_best": before["best_model"], "candidate_best": after["best_model"],
        "constant_baseline": before["constant_baseline"],
        "delta_candidate_minus_baseline": difference,
        "relative_mae_reduction": -difference["mae"] / initial_mae if initial_mae else None,
        "field_deltas": [
            {"source_id": source, "group_id": before_fields[source]["group_id"],
             "z": before_fields[source]["z"],
             "delta_candidate_minus_baseline": _delta(before_fields[source], after_fields[source])}
            for source in sorted(before_fields)],
        "limitations": [
            "Best checkpoints are selected on validation MAE over different training budgets.",
            "A lower MAE alone does not establish spatial or useful staining prediction.",
            "This report verifies recorded artifacts; it does not recompute image predictions.",
            "Source revision and recorded package equality do not prove hardware-level determinism."]}
    output_dir.mkdir(parents=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def compare_objectives(l1_dir: Path, mse_dir: Path, output_dir: Path):
    """Fail closed unless two fresh runs differ only in their loss objective."""
    if output_dir.exists():
        raise ValueError("comparison output already exists")
    before, left, constant, old_history, left_sha = _read_run(l1_dir)
    after, right, other_constant, new_history, right_sha = _read_run(mse_dir)
    a, b = before["config"], after["config"]
    if a.get("objective") != "l1" or b.get("objective") != "mse":
        raise ValueError("explicit L1 baseline and MSE candidate objectives are required")
    for config, run in ((a, before), (b, after)):
        if (any(key in config for key in ("resume_from_step", "resume_checkpoint_sha256")) or
                run["steps_this_run"] != config["steps"]):
            raise ValueError("objective comparison requires fresh runs, not warm starts")
        digest = config.get("initial_model_state_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("initial model fingerprint is required")
    for key in sorted((a.keys() | b.keys()) - {"objective"}):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"not an objective-only comparison: {key} differs")
    for key in ("sampled_batches_sha256", "sampling_state_sha256"):
        digest = before.get(key, "")
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or
                digest != after.get(key)):
            raise ValueError(f"training sampling differs or is unverified: {key}")
    if any(run.get("sampled_batches_scope") != "updates in this run only" for run in (before, after)):
        raise ValueError("sampling fingerprint scope is unverified")
    if (old_history[0]["step"] != 0 or new_history[0]["step"] != 0 or
            old_history[0] != new_history[0] or before["untrained_model"] != after["untrained_model"]):
        raise ValueError("untrained model results differ")
    if left.keys() != right.keys():
        raise ValueError("validation source identities changed")
    for source in left:
        if any(left[source][key] != right[source][key] for key in ("group_id", "z")):
            raise ValueError("validation group or depth changed")
    for key in ("train_fields", "validation_fields", "constant_value", "constant_definition"):
        if before[key] != after[key]:
            raise ValueError(f"baseline contract changed: {key}")
    if constant != other_constant:
        raise ValueError("constant baseline validation results changed")
    result = {
        "purpose": "matched fresh L1 versus MSE development comparison; not test evaluation",
        "test_pixels_loaded": False,
        "l1_report_sha256": left_sha, "mse_report_sha256": right_sha,
        "prepared_metadata_sha256": a["prepared_metadata_sha256"],
        "pilot_plan_sha256": a["pilot_plan_sha256"],
        "initial_model_state_sha256": a["initial_model_state_sha256"],
        "sampled_batches_sha256": before["sampled_batches_sha256"],
        "total_steps_each": a["steps"],
        "l1_best_step": before["best_step"], "mse_best_step": after["best_step"],
        "l1_best": before["best_model"], "mse_best": after["best_model"],
        "constant_baseline": before["constant_baseline"],
        "delta_mse_minus_l1": _delta(before["best_model"], after["best_model"]),
        "field_deltas": [
            {"source_id": source, "group_id": left[source]["group_id"], "z": left[source]["z"],
             "delta_mse_minus_l1": _delta(left[source], right[source])} for source in sorted(left)],
        "limitations": [
            "Checkpoints are selected on validation MAE, not their training objectives.",
            "Training L1 and MSE loss values are not directly comparable.",
            "Lower MAE alone does not establish useful spatial staining.",
            "This verifies recorded artifacts; predictions are not recomputed here."]}
    output_dir.mkdir(parents=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
