import argparse
import csv
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from .manifest import audit_manifest


def environment_report() -> dict:
    report = {"python": platform.python_version(), "platform": platform.system(),
              "machine": platform.machine(), "packages": {}}
    for name in ("torch", "numpy", "tifffile", "imagecodecs", "scikit-image", "matplotlib"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "not installed"
    if report["packages"]["torch"] == "not installed":
        report["device"] = "not checked (install the ml extra first)"
    else:
        try:
            import torch
            report["mps_available"] = torch.backends.mps.is_available()
            report["cuda_available"] = torch.cuda.is_available()
            report["device"] = ("cuda" if report["cuda_available"] else
                                "mps" if report["mps_available"] else "cpu")
        except (ImportError, OSError, RuntimeError) as error:
            report["device"] = "torch import failed"
            report["torch_error"] = str(error)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Virtual staining project tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="show installed versions and device availability")
    inspect = commands.add_parser("inspect-tiff", help="read TIFF axes and OME channel metadata")
    inspect.add_argument("path", type=Path)
    preview = commands.add_parser("preview-pair", help="display measured input/target planes and overlays")
    preview.add_argument("path", type=Path)
    preview.add_argument("--z", type=int, nargs="+", required=True)
    preview.add_argument("--input-channel", type=int, required=True)
    preview.add_argument("--target-channel", type=int, required=True)
    preview.add_argument("--out-dir", type=Path, required=True)
    split = commands.add_parser("split", help="freeze source assignments by acquisition group")
    split.add_argument("inventory", type=Path)
    split.add_argument("--validation-groups", type=int, required=True)
    split.add_argument("--test-groups", type=int, required=True)
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--out-dir", type=Path, required=True)
    inventory = commands.add_parser("allen-inventory", help="review fields and plate groups in Allen metadata")
    inventory.add_argument("metadata", type=Path)
    inventory.add_argument("--cell-line", required=True)
    inventory.add_argument("--out-dir", type=Path, required=True)
    pilot = commands.add_parser("allen-pilot", help="freeze a small plate-held-out pilot and resolve download versions")
    pilot.add_argument("metadata", type=Path)
    pilot.add_argument("--cell-line", required=True)
    pilot.add_argument("--groups", type=int, required=True)
    pilot.add_argument("--fields-per-group", type=int, required=True)
    pilot.add_argument("--seed", type=int, required=True)
    pilot.add_argument("--out-dir", type=Path, required=True)
    download = commands.add_parser("download-pilot", help="download version-pinned pilot images and record checksums")
    download.add_argument("plan", type=Path)
    download.add_argument("--out-dir", type=Path, required=True)
    download.add_argument("--development-only", action="store_true", help="skip test downloads without changing the frozen plan")
    pilot_audit = commands.add_parser("audit-pilot", help="verify pilot TIFFs and preview training planes")
    pilot_audit.add_argument("plan", type=Path)
    pilot_audit.add_argument("--data-dir", type=Path, required=True)
    pilot_audit.add_argument("--out-dir", type=Path, required=True)
    prepare = commands.add_parser("prepare-training", help="prepare fixed middle planes and fit training-only scaling")
    prepare.add_argument("audit", type=Path)
    prepare.add_argument("--out-dir", type=Path, required=True)
    context = commands.add_parser("prepare-context", help="verify neighbouring inputs against a frozen middle-plane baseline")
    context.add_argument("plan", type=Path)
    context.add_argument("baseline", type=Path)
    context.add_argument("receipts", type=Path)
    context.add_argument("--data-dir", type=Path, required=True)
    context.add_argument("--out-dir", type=Path, required=True)
    overfit = commands.add_parser("overfit", help="debug learning on one fixed training patch")
    overfit.add_argument("dataset", type=Path)
    overfit.add_argument("--steps", type=int, default=100)
    overfit.add_argument("--patch-size", type=int, default=128)
    overfit.add_argument("--width", type=int, default=16)
    overfit.add_argument("--seed", type=int, default=17)
    overfit.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    overfit.add_argument("--out-dir", type=Path, required=True)
    debug_preview = commands.add_parser("preview-overfit", help="inspect debug predictions against a constant baseline")
    debug_preview.add_argument("dataset", type=Path)
    debug_preview.add_argument("checkpoint", type=Path)
    debug_preview.add_argument("--out-dir", type=Path, required=True)
    train = commands.add_parser("train", help="train across development fields with full-plane validation")
    train.add_argument("dataset", type=Path)
    train.add_argument("--steps", type=int, default=500)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--patch-size", type=int, default=128)
    train.add_argument("--width", type=int, default=16)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--objective", choices=("l1", "mse"), default="l1")
    train.add_argument("--input-normalisation", choices=("global", "per-plane"), default="global")
    train.add_argument("--input-context", choices=("single", "repeated-centre", "neighbouring"), default="single")
    train.add_argument("--validate-every", type=int, default=100)
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    train.add_argument("--resume", type=Path)
    train.add_argument("--out-dir", type=Path, required=True)
    diagnosis = commands.add_parser("diagnose-training", help="compare training/validation intensities and prediction context")
    diagnosis.add_argument("dataset", type=Path)
    diagnosis.add_argument("run", type=Path)
    diagnosis.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    diagnosis.add_argument("--out-dir", type=Path, required=True)
    data_diagnosis = commands.add_parser("diagnose-data", help="inspect raw development intensities without a checkpoint")
    data_diagnosis.add_argument("dataset", type=Path)
    data_diagnosis.add_argument("--patch-size", type=int, default=128)
    data_diagnosis.add_argument("--out-dir", type=Path, required=True)
    comparison = commands.add_parser("compare-duration", help="verify and compare a duration-only continuation")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--out-dir", type=Path, required=True)
    objectives = commands.add_parser("compare-objectives", help="verify a matched fresh L1/MSE comparison")
    objectives.add_argument("l1_run", type=Path)
    objectives.add_argument("mse_run", type=Path)
    objectives.add_argument("--out-dir", type=Path, required=True)
    normalisation = commands.add_parser("compare-normalisation", help="verify matched global/per-plane input scaling runs")
    normalisation.add_argument("global_run", type=Path)
    normalisation.add_argument("per_plane_run", type=Path)
    normalisation.add_argument("--out-dir", type=Path, required=True)
    context_comparison = commands.add_parser("compare-context", help="verify matched fresh three-input context runs")
    context_comparison.add_argument("centre_run", type=Path)
    context_comparison.add_argument("neighbour_run", type=Path)
    context_comparison.add_argument("--out-dir", type=Path, required=True)
    evaluation = commands.add_parser("evaluate-heldout", help="evaluate both precommitted models on locked test fields (CPU)")
    evaluation.add_argument("protocol", type=Path)
    evaluation.add_argument("plan", type=Path)
    evaluation.add_argument("receipts", type=Path)
    evaluation.add_argument("--checkpoint-root", type=Path, required=True)
    evaluation.add_argument("--data-dir", type=Path, required=True)
    evaluation.add_argument("--protocol-sha256", required=True, help="fingerprint retained before any test access")
    evaluation.add_argument("--out-dir", type=Path, required=True)
    audit = commands.add_parser("audit", help="check a paired-image CSV manifest")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        result = environment_report()
        status = 1 if "torch_error" in result else 0
    elif args.command == "inspect-tiff":
        from .image_info import read_tiff_info
        try:
            result = read_tiff_info(args.path)
        except (OSError, ValueError, ImportError) as error:
            print(f"Cannot inspect TIFF (install the ml extra if needed): {error}", file=sys.stderr)
            return 2
        status = 0
    elif args.command == "preview-pair":
        try:
            from .preview import preview_pair
            result = preview_pair(args.path, z_indices=args.z, input_channel=args.input_channel,
                                  target_channel=args.target_channel, out_dir=args.out_dir)
        except (OSError, ValueError, ImportError) as error:
            print(f"Cannot preview pair (install the ml extra if needed): {error}", file=sys.stderr)
            return 2
        status = 0
    elif args.command in ("prepare-training", "prepare-context", "overfit", "preview-overfit", "train", "diagnose-training", "diagnose-data", "compare-duration", "compare-objectives", "compare-normalisation", "compare-context"):
        try:
            if args.command == "compare-context":
                from .context_comparison import compare_context
                result = compare_context(args.centre_run, args.neighbour_run, args.out_dir)
            elif args.command == "prepare-context":
                from .context_data import prepare_context_data
                result = prepare_context_data(args.plan, args.baseline, args.receipts, args.data_dir, args.out_dir)
            elif args.command == "prepare-training":
                from .training_data import prepare_training_data
                result = prepare_training_data(args.audit, args.out_dir)
            elif args.command == "compare-normalisation":
                from .preprocessing_comparison import compare_normalisation
                result = compare_normalisation(args.global_run, args.per_plane_run, args.out_dir)
            elif args.command == "compare-objectives":
                from .run_comparison import compare_objectives
                result = compare_objectives(args.l1_run, args.mse_run, args.out_dir)
            elif args.command == "compare-duration":
                from .run_comparison import compare_duration
                result = compare_duration(args.baseline, args.candidate, args.out_dir)
            elif args.command == "diagnose-data":
                from .data_diagnostics import diagnose_data
                result = diagnose_data(args.dataset, args.out_dir, patch_size=args.patch_size)
            elif args.command == "preview-overfit":
                from .debug_preview import preview_overfit
                result = preview_overfit(args.dataset, args.checkpoint, args.out_dir)
            elif args.command == "train":
                from .trainer import train_pilot
                result = train_pilot(args.dataset, args.out_dir, steps=args.steps,
                                     batch_size=args.batch_size, patch_size=args.patch_size,
                                     width=args.width, seed=args.seed, validate_every=args.validate_every,
                                     device_name=args.device, resume_path=args.resume, objective=args.objective,
                                     input_normalisation=args.input_normalisation, input_context=args.input_context)
            elif args.command == "diagnose-training":
                from .diagnostics import diagnose_training
                result = diagnose_training(args.dataset, args.run, args.out_dir, device_name=args.device)
            else:
                from .overfit import overfit_patch
                result = overfit_patch(args.dataset, args.out_dir, steps=args.steps,
                                      patch_size=args.patch_size, width=args.width, seed=args.seed,
                                      device_name=args.device)
        except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
            print(f"Cannot prepare/check training: {error}", file=sys.stderr)
            return 2
        status = 0
    elif args.command == "evaluate-heldout":
        try:
            from .evaluation import evaluate_heldout
            result = evaluate_heldout(args.protocol, args.plan, args.receipts, args.checkpoint_root,
                                     args.data_dir, args.out_dir, expected_protocol_sha256=args.protocol_sha256)
        except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
            print(f"Cannot evaluate held-out data: {error}", file=sys.stderr)
            return 2
        status = 0
    elif args.command == "audit-pilot":
        try:
            from .pilot_audit import audit_pilot
            result = audit_pilot(args.plan, args.data_dir, args.out_dir)
        except (OSError, ValueError, KeyError, ImportError) as error:
            print(f"Cannot audit pilot: {error}", file=sys.stderr)
            return 2
        status = 0 if result["valid"] else 1
    elif args.command in ("allen-pilot", "download-pilot"):
        try:
            from .pilot import select_pilot, resolve_downloads, write_pilot, download_pilot
            if args.command == "download-pilot":
                result = download_pilot(args.plan, args.out_dir, development_only=args.development_only)
            else:
                from .allen_inventory import read_allen_inventory
                if args.out_dir.exists():
                    raise ValueError("pilot output already exists")
                report = read_allen_inventory(args.metadata, cell_line=args.cell_line)
                plan = select_pilot(report, groups=args.groups,
                                    fields_per_group=args.fields_per_group, seed=args.seed)
                result = write_pilot(resolve_downloads(plan), args.out_dir)
        except (OSError, ValueError, KeyError, csv.Error, UnicodeError) as error:
            print(f"Cannot prepare/download pilot: {error}", file=sys.stderr)
            return 2
        status = 0
    elif args.command == "allen-inventory":
        try:
            from .allen_inventory import read_allen_inventory, write_allen_inventory
            report = read_allen_inventory(args.metadata, cell_line=args.cell_line)
            result = write_allen_inventory(report, args.out_dir)
        except (OSError, ValueError, csv.Error, UnicodeError) as error:
            print(f"Cannot inventory Allen metadata: {error}", file=sys.stderr)
            return 2
        status = 0
    elif args.command == "split":
        try:
            from .splits import create_split_plan, read_inventory, write_split_plan
            inventory = read_inventory(args.inventory)
            plan = create_split_plan(inventory, validation_groups=args.validation_groups,
                                     test_groups=args.test_groups, seed=args.seed)
            result = write_split_plan(plan, args.out_dir)
        except (OSError, ValueError, csv.Error, UnicodeError) as error:
            print(f"Cannot create split: {error}", file=sys.stderr)
            return 2
        status = 0
    else:
        try:
            result = audit_manifest(args.manifest, args.data_root)
        except (OSError, csv.Error, UnicodeError) as error:
            print(f"Cannot read manifest or data: {error}", file=sys.stderr)
            return 2
        status = 0 if result["valid"] else 1
    print(json.dumps(result, indent=2))
    return status
