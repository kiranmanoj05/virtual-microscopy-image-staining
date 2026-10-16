"""Verify a fresh, input-normalisation-only development comparison."""

import json
from pathlib import Path

from .run_comparison import _read_run, _delta


def _fingerprint(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def compare_normalisation(global_dir: Path, plane_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise ValueError("comparison output already exists")
    before, left, constant, _, left_sha = _read_run(global_dir)
    after, right, other_constant, _, right_sha = _read_run(plane_dir)
    a, b = before["config"], after["config"]
    if a.get("input_normalisation") != "global" or b.get("input_normalisation") != "per-plane":
        raise ValueError("explicit global and per-plane input policies are required")
    for run in (before, after):
        config = run["config"]
        if (any(k in config for k in ("resume_from_step", "resume_checkpoint_sha256")) or
                run["steps_this_run"] != config["steps"]):
            raise ValueError("normalisation comparison requires fresh runs")
        if not all(_fingerprint(config.get(k)) for k in
                   ("initial_model_state_sha256", "preprocessing_sha256")):
            raise ValueError("initial-model and preprocessing fingerprints are required")
    for key in sorted((a.keys() | b.keys()) - {"input_normalisation", "preprocessing_sha256"}):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"not a normalisation-only comparison: {key} differs")
    if a["preprocessing_sha256"] == b["preprocessing_sha256"]:
        raise ValueError("distinct preprocessing fingerprints are required")
    for key in ("sampler_trace_sha256", "target_batches_sha256", "sampling_state_sha256"):
        if not _fingerprint(before.get(key)) or before[key] != after.get(key):
            raise ValueError(f"sampling or targets differ: {key}")
    if any(r.get("sampled_batches_scope") != "updates in this run only" for r in (before, after)):
        raise ValueError("sampling scope is unverified")
    if left.keys() != right.keys():
        raise ValueError("validation source identities changed")
    for source in left:
        if any(left[source][k] != right[source][k] for k in ("group_id", "z")):
            raise ValueError("validation group or depth changed")
    for key in ("train_fields", "validation_fields", "constant_value", "constant_definition"):
        if before[key] != after[key]:
            raise ValueError(f"baseline contract changed: {key}")
    if constant != other_constant:
        raise ValueError("constant baseline validation results changed")
    result = {
        "purpose": "matched input-normalisation-only development comparison; not test evaluation",
        "test_pixels_loaded": False, "global_report_sha256": left_sha, "per_plane_report_sha256": right_sha,
        "prepared_metadata_sha256": a["prepared_metadata_sha256"],
        "pilot_plan_sha256": a["pilot_plan_sha256"], "objective": a["objective"],
        "initial_model_state_sha256": a["initial_model_state_sha256"],
        "sampler_trace_sha256": before["sampler_trace_sha256"],
        "target_batches_sha256": before["target_batches_sha256"], "total_steps_each": a["steps"],
        "global_best_step": before["best_step"], "per_plane_best_step": after["best_step"],
        "global_best": before["best_model"], "per_plane_best": after["best_model"],
        "constant_baseline": before["constant_baseline"],
        "delta_per_plane_minus_global": _delta(before["best_model"], after["best_model"]),
        "field_deltas": [{"source_id": s, "group_id": left[s]["group_id"], "z": left[s]["z"],
                          "delta_per_plane_minus_global": _delta(left[s], right[s])} for s in sorted(left)],
        "limitations": ["Untrained scores may differ because input values intentionally differ.",
                        "This verifies recorded artifacts; it does not recompute predictions.",
                        "Lower MAE alone does not establish useful staining or generalisation."]}
    output_dir.mkdir(parents=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
