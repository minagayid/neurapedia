# Neurapedia reference engine plan

This repository now includes a metadata-only neuroimaging reference registry
in `neurapedia/reference-manifest.json`. It separates normal/reference,
clinical/pathology, artifact/QC, reconstruction, and ground-truth evidence.
The repository stores no downloaded scans and never bypasses controlled-access
or dataset-specific terms.

## Implemented baseline

- Registry loading and structural validation with the standard library.
- Search and filtering by query, tier, evidence class, role, modality, and access mode.
- Deterministic plans for normal reference, anomaly detection, artifact correction, and benchmarking.
- Report integration: selected reference IDs and access metadata are recorded in the JSON/HTML sidecar.
- Conservative technical correction suggestions for explicit artifact flags and deterministic anomaly observations.
- CLI entry points: `reference list`, `reference search`, `reference validate`, and `reference plan`.

Examples:

```text
python -m neurapedia reference validate
python -m neurapedia reference search --modality MRI --evidence-class lesion_benchmark --json
python -m neurapedia reference plan --purpose artifact_correction --modality MRI --json
```

## Upgrade path to a production-capable engine

1. Add provider adapters that harvest dataset/project metadata first and
   require an explicit accession, subject, or snapshot allowlist.
2. Add a lockfile with exact dataset versions/snapshots, checksums, license or
   DUA links, de-identification status, and retrieval timestamps.
3. Normalize selected data into BIDS/NIfTI while preserving original DICOM,
   affine, orientation, scanner, sequence, site, and acquisition metadata.
4. Validate BIDS, geometry, orientation, intensity ranges, missing slices,
   modality compatibility, and de-identification before indexing.
5. Build separate metadata, anatomy, lesion, artifact, and embedding indexes;
   keep raw scans and large indexes outside Git under `NEURAPEDIA_DATA_ROOT`.
6. Add benchmark fixtures with known segmentation, QC, reconstruction, and
   artifact truth. Measure sensitivity, specificity, Dice/Hausdorff where
   appropriate, calibration, abstention, and false-correction rates.
7. Introduce age-, sex-, scanner-, site-, sequence-, and voxel-resolution-
   conditioned normative models. Never compare a scan with an unmatched cohort
   and call the difference pathology.

## Acceptance gates

- No controlled or non-redistributable resource is downloaded automatically.
- Every selected dataset has a pinned snapshot/version and provenance record.
- Anomaly output distinguishes artifact, biological candidate, and unknown;
  similarity is evidence, not diagnosis.
- Correction output is review-only technical QC guidance and never mutates
  voxels or recommends treatment.
- Registry, plan, correction, and report behavior is covered by tests.
