"""Neurapedia command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .analysis import AnomalyDetector, BrainSegmenter
from .core import generate_synthetic_mri
from .correction import CorrectionPlanner
from .io import NeuroimagingDataLoader
from .references import DEFAULT_MANIFEST, ReferenceRegistry
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
    demo.add_argument("--reference-purpose", choices=["normal_reference", "anomaly_detection", "artifact_correction", "benchmarking"], default="anomaly_detection")
    demo.add_argument("--reference-modality", default="MRI")
    analyze = commands.add_parser("analyze", help="analyze a NPY, NIfTI, or DICOM study")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--format", choices=["auto", "npy", "nifti", "dicom"], default="auto")
    analyze.add_argument(
        "--frame",
        type=int,
        help="zero-based temporal frame to analyze when the input is 4-D",
    )
    analyze.add_argument("--patient-id")
    analyze.add_argument("--output", type=Path, default=Path("neurapedia-report.html"))
    analyze.add_argument("--threshold", type=float, default=4.0)
    analyze.add_argument("--reference-purpose", choices=["normal_reference", "anomaly_detection", "artifact_correction", "benchmarking"], default="anomaly_detection")
    analyze.add_argument("--reference-modality")

    reference = commands.add_parser("reference", help="inspect the versioned metadata-only reference registry")
    reference_commands = reference.add_subparsers(dest="reference_command", required=True)
    for command_name, help_text in (("list", "list registry sources"), ("search", "search registry sources")):
        reference_list = reference_commands.add_parser(command_name, help=help_text)
        reference_list.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        reference_list.add_argument("--query", default="")
        reference_list.add_argument("--tier")
        reference_list.add_argument("--evidence-class")
        reference_list.add_argument("--role")
        reference_list.add_argument("--modality")
        reference_list.add_argument("--access-mode")
        reference_list.add_argument("--json", action="store_true", dest="as_json")
    validate = reference_commands.add_parser("validate", help="validate a local manifest without downloading data")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    validate.add_argument("--json", action="store_true", dest="as_json")
    plan = reference_commands.add_parser("plan", help="build a deterministic source-selection plan")
    plan.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    plan.add_argument("--purpose", choices=["normal_reference", "anomaly_detection", "artifact_correction", "benchmarking"], default="normal_reference")
    plan.add_argument("--modality", default="MRI")
    plan.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _reference_rows(registry: ReferenceRegistry, args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {
            "id": source.id,
            "name": source.name,
            "tier": source.tier,
            "evidence_class": source.evidence_class,
            "roles": list(source.roles),
            "modalities": list(source.modalities),
            "provider_url": source.provider_url,
            "access_mode": source.access_mode,
            "redistribution": source.redistribution,
            "acquisition": source.acquisition,
            "data_products": list(source.data_products),
        }
        for source in registry.search(
            args.query,
            tier=args.tier,
            evidence_class=args.evidence_class,
            role=args.role,
            modality=args.modality,
            access_mode=args.access_mode,
        )
    ]


def _run_reference_command(args: argparse.Namespace) -> int:
    registry = ReferenceRegistry.from_path(args.manifest)
    if args.reference_command == "validate":
        payload = {
            "status": "valid",
            "manifest": str(args.manifest),
            "schema_version": registry.schema_version,
            "manifest_version": registry.manifest_version,
            "source_count": len(registry.sources),
        }
        print(json.dumps(payload, indent=2, sort_keys=True) if args.as_json else payload["status"])
        return 0
    if args.reference_command in {"list", "search"}:
        rows = _reference_rows(registry, args)
        if args.as_json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            for row in rows:
                print(f"{row['id']}\t{row['tier']}\t{row['evidence_class']}\t{row['name']}\t{row['access_mode']}")
        return 0
    payload = registry.plan(purpose=args.purpose, modality=args.modality).to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True) if args.as_json else "\n".join(payload["source_ids"]))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "reference":
            return _run_reference_command(args)
        if args.command == "demo":
            image = generate_synthetic_mri(
                shape=tuple(args.shape), seed=args.seed, patient_id=args.patient_id, include_observation=True
            )
        else:
            image = NeuroimagingDataLoader().load(args.input, patient_id=args.patient_id, format=args.format)
        frame_index = getattr(args, "frame", None)
        segments = BrainSegmenter().segment(image, frame_index=frame_index)
        anomalies = AnomalyDetector(threshold=args.threshold).detect(
            image, regions=segments, frame_index=frame_index
        )
        if frame_index is not None:
            image.metadata["analysis_frame"] = frame_index
        modality = args.reference_modality or image.scan_type
        reference_plan = ReferenceRegistry.from_path().plan(args.reference_purpose, modality)
        correction_suggestions = CorrectionPlanner().plan(
            image, anomalies=anomalies, reference_plan=reference_plan
        )
        output = write_report(
            args.output,
            image,
            anomalies=anomalies,
            segments=segments,
            reference_plan=reference_plan,
            correction_suggestions=correction_suggestions,
        )
        print(f"Research report written to {output}")
        print(f"JSON summary written to {output.with_suffix('.json')}")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
        return 2

