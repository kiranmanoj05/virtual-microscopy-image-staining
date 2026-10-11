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
    pilot_audit = commands.add_parser("audit-pilot", help="verify pilot TIFFs and preview training planes")
    pilot_audit.add_argument("plan", type=Path)
    pilot_audit.add_argument("--data-dir", type=Path, required=True)
    pilot_audit.add_argument("--out-dir", type=Path, required=True)
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
                result = download_pilot(args.plan, args.out_dir)
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
