# Neurapedia

Neurapedia is a local, research-only neuroimaging inspection toolkit. It accepts small NumPy volumes in the core installation and provides optional NIfTI/DICOM adapters for real imaging files.

It is not a diagnostic system and does not provide treatment recommendations. Every experimental observation requires qualified human review.

## Quick start

Requirements: 64-bit CPython 3.10–3.12 and NumPy. The core workflow runs locally without a server, database, cloud account, or model download.

```powershell
python -m pip install -r requirements.txt
python -m neurapedia demo --shape 48 48 24 --output neurapedia-report.html
```

Open the generated HTML file locally. A JSON sidecar is created beside it.

Analyze a NumPy volume:

```powershell
python -m neurapedia analyze scan.npy --output scan-report.html
```

Install real imaging readers only when needed:

```powershell
python -m pip install -e .[imaging]
python -m neurapedia analyze scan.nii.gz --output scan-report.html
python -m neurapedia analyze dicom-study/ --format dicom --output dicom-report.html
```

For a 4-D NIfTI or NumPy time series, select one zero-based frame explicitly:

```powershell
python -m neurapedia analyze bold.nii.gz --frame 12 --output frame-12-report.html
```

Analysis never averages a time series implicitly. Intake is bounded to 512 MiB per file, 16 million spatial voxels, 32 million voxels total, and 512 temporal frames. DICOM folders are limited to 512 instances and 512 MiB total; multi-frame DICOM objects are rejected until their per-frame geometry can be represented safely. DICOM inputs must describe one consistently oriented and regularly spaced grayscale series. Physical NIfTI and DICOM affines use RAS+ millimeters; DICOM LPS coordinates are converted to RAS+. NumPy volumes use voxel-index coordinates because they carry no physical orientation.

The direct launcher also works from a fresh checkout:

```powershell
python run_neurapedia.py demo
```

## Tests

The shipped tests use the standard library test runner:

```powershell
python -m unittest discover -s tests -v
```

Optional development tools are available with `python -m pip install -e .[dev]`.

## What the pipeline does

- Validates numeric 3-D and 4-D volumes and rejects non-finite data.
- Loads `.npy` volumes with the core installation.
- Loads NIfTI and DICOM only through explicit optional adapters; missing or invalid files fail clearly and never become fabricated synthetic scans.
- Generates deterministic synthetic MRI fixtures for demonstrations.
- Produces a disjoint, coarse research atlas with ten named region labels for visualization and testing. It is not a clinical anatomical segmentation.
- Detects robust intensity observations and connected components. It does not assign diagnoses or calibrated clinical probabilities.
- Writes self-contained HTML and JSON research summaries with source metadata and safety warnings.
- Includes a versioned, metadata-only reference registry for normal anatomy,
  pathology/lesion, artifact/QC, reconstruction, and ground-truth datasets.
- Produces deterministic reference-selection plans and review-only technical
  correction suggestions; it never edits voxels or recommends treatment.

## Optional open-source backends

The deterministic report backend works offline by default. `OpenAICompatibleBackend` can summarize already-computed structured measurements through a local OpenAI-compatible server such as Ollama or vLLM. Configure the endpoint and model yourself; Neurapedia never downloads a model or sends imaging data automatically.

Open-source imaging model integrations can be added through optional MONAI or nnU-Net adapters, but they require compatible checkpoints, hardware, validation data, and a modality-specific workflow. They are not silently used as a substitute for the deterministic core.

## Privacy and safety

Patient labels are kept local. DICOM loading defaults to an `anonymous` label and does not print DICOM identifiers. The project does not transmit scans. Before using real data, follow your institution’s de-identification, storage, access-control, and review requirements.

## Layout

```text
neurapedia/
├── neurapedia/       # package, CLI, loaders, analysis, reports, backends
├── tests/            # standard-library regression tests
├── examples/         # runnable demo
├── pyproject.toml    # package metadata and optional extras
└── run_neurapedia.py # fresh-checkout launcher
```

## License

MIT. See [LICENSE](LICENSE).

## Reference registry

Inspect the reference registry without downloading data:

```powershell
python -m neurapedia reference validate
python -m neurapedia reference search --modality MRI --evidence-class lesion_benchmark --json
python -m neurapedia reference plan --purpose artifact_correction --modality MRI --json
```

See [REFERENCE_ENGINE_PLAN.md](REFERENCE_ENGINE_PLAN.md) for the dataset
catalog, access/licensing boundary, implemented baseline, and production
upgrade path. Large, controlled, or non-redistributable datasets must be
acquired under their provider terms.
