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
    for name in ("torch", "numpy", "tifffile", "scikit-image", "matplotlib"):
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
    audit = commands.add_parser("audit", help="check a paired-image CSV manifest")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        result = environment_report()
        status = 1 if "torch_error" in result else 0
    else:
        try:
            result = audit_manifest(args.manifest, args.data_root)
        except (OSError, csv.Error, UnicodeError) as error:
            print(f"Cannot read manifest or data: {error}", file=sys.stderr)
            return 2
        status = 0 if result["valid"] else 1
    print(json.dumps(result, indent=2))
    return status
