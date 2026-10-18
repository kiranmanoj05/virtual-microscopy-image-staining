"""Verify a fresh matched three-input context-only development comparison."""

import json
from pathlib import Path

import torch

from .model import UNet
from .trainer import model_state_digest
from .preprocessing_comparison import _fingerprint
from .run_comparison import _read_run, _delta


def compare_context(centre_dir: Path, neighbour_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise ValueError("comparison output already exists")
    before, left, constant, _, left_sha = _read_run(centre_dir)
    after, right, other_constant, _, right_sha = _read_run(neighbour_dir)
    a, b = before["config"], after["config"]
    if a.get("input_context") != "repeated-centre" or b.get("input_context") != "neighbouring":
        raise ValueError("matched repeated-centre and neighbouring policies are required")
    for key in sorted((a.keys() | b.keys()) - {"input_context"}):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"not a context-only comparison: {key} differs")
    for directory, run in ((centre_dir, before), (neighbour_dir, after)):
        config = run["config"]
        if (any(k in config for k in ("resume_from_step", "resume_checkpoint_sha256")) or
                run["steps_this_run"] != config["steps"]):
            raise ValueError("context comparison requires fresh runs")
        if config.get("input_channels") != 3 or config.get("input_normalisation") != "global":
            raise ValueError("matched three-input models with frozen global scaling are required")
        if not all(_fingerprint(config.get(k)) for k in
                   ("initial_model_state_sha256", "preprocessing_sha256", "prepared_metadata_sha256")):
            raise ValueError("model, preprocessing and data fingerprints are required")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            model = UNet(config["width"], input_channels=3)
        if (model_state_digest(model) != config["initial_model_state_sha256"] or
                sum(p.numel() for p in model.parameters()) != config["parameter_count"]):
            raise ValueError("initial model or parameter count does not match recorded architecture")
        for name, step in (("best", run["best_step"]), ("last", config["steps"])):
            checkpoint = torch.load(directory / f"{name}-model.pt", map_location="cpu", weights_only=True)
            if checkpoint["config"] != config or checkpoint["step"] != step:
                raise ValueError("checkpoint configuration or step differs from report")
            model.load_state_dict(checkpoint["model_state"])
    for key in ("sampler_trace_sha256", "target_batches_sha256", "sampling_state_sha256"):
        if not _fingerprint(before.get(key)) or before[key] != after.get(key):
            raise ValueError(f"sampling or targets differ: {key}")
    if any(r.get("sampled_batches_scope") != "updates in this run only" for r in (before, after)):
        raise ValueError("sampling scope is unverified")
    if (not all(_fingerprint(r.get("sampled_batches_sha256")) for r in (before, after)) or
            before["sampled_batches_sha256"] == after["sampled_batches_sha256"]):
        raise ValueError("context input batches must differ")
    if left.keys() != right.keys() or any(
            left[s][k] != right[s][k] for s in left for k in ("group_id", "z")):
        raise ValueError("validation source/group/depth identities changed")
    for key in ("train_fields", "validation_fields", "constant_value", "constant_definition"):
        if before[key] != after[key]:
            raise ValueError(f"baseline contract changed: {key}")
    if constant != other_constant:
        raise ValueError("constant baseline validation results changed")
    result = {
        "purpose": "fresh matched three-input context-only comparison; not test evaluation",
        "test_pixels_loaded": False, "centre_report_sha256": left_sha, "neighbour_report_sha256": right_sha,
        "prepared_metadata_sha256": a["prepared_metadata_sha256"], "pilot_plan_sha256": a["pilot_plan_sha256"],
        "initial_model_state_sha256": a["initial_model_state_sha256"], "parameter_count_each": a["parameter_count"],
        "sampler_trace_sha256": before["sampler_trace_sha256"], "target_batches_sha256": before["target_batches_sha256"],
        "total_steps_each": a["steps"], "centre_best_step": before["best_step"], "neighbour_best_step": after["best_step"],
        "centre_best": before["best_model"], "neighbour_best": after["best_model"],
        "constant_baseline": before["constant_baseline"],
        "delta_neighbour_minus_centre": _delta(before["best_model"], after["best_model"]),
        "field_deltas": [{"source_id": s, "group_id": left[s]["group_id"], "z": left[s]["z"],
                          "delta_neighbour_minus_centre": _delta(left[s], right[s])} for s in sorted(left)],
        "limitations": ["Repeated-centre control is a new three-input model, not the historical one-input run.",
                        "Untrained predictions can differ because the input information differs.",
                        "Recorded-artifact verification is separate from diagnostic prediction recomputation.",
                        "One validation plate does not establish generalisation; low MAE alone is not useful staining."]}
    output_dir.mkdir(parents=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
