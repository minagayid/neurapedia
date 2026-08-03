"""Neurapedia command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .analysis import AnomalyDetector, BrainSegmenter
from .core import generate_synthetic_mri
from .io import NeuroimagingDataLoader
from .reporting import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurapedia", description="Research-only local neuroimaging inspection.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run a deterministic synthetic MRI demonstration")
    demo.add_argument("--shape", nargs=3, type=int, default=(48, 48, 24), metavar=("X", "Y", "Z"))
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--patient-id", default="synthetic")
    demo.add_argument("--output", type=Path, default=Path("neurapedia-report.html"))
    demo.add_argument("--threshold", type=float, default=4.0)
    analyze = commands.add_parser("analyze", help="analyze a NPY, NIfTI, or DICOM study")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--format", choices=["auto", "npy", "nifti", "dicom"], default="auto")
    analyze.add_argument("--patient-id")
    analyze.add_argument("--output", type=Path, default=Path("neurapedia-report.html"))
    analyze.add_argument("--threshold", type=float, default=4.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            image = generate_synthetic_mri(
                shape=tuple(args.shape), seed=args.seed, patient_id=args.patient_id, include_observation=True
            )
        else:
            image = NeuroimagingDataLoader().load(args.input, patient_id=args.patient_id, format=args.format)
        segments = BrainSegmenter().segment(image)
        anomalies = AnomalyDetector(threshold=args.threshold).detect(image, regions=segments)
        output = write_report(args.output, image, anomalies=anomalies, segments=segments)
        print(f"Research report written to {output}")
        print(f"JSON summary written to {output.with_suffix('.json')}")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
        return 2

