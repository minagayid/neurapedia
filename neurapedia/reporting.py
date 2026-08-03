"""Self-contained research summaries for local review."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

from .core import Anomaly, BRAIN_REGIONS, NeuroImage


def render_html_report(
    image: NeuroImage,
    anomalies: Iterable[Anomaly] = (),
    segments: dict[str, object] | None = None,
    title: str = "Experimental Research Summary",
) -> str:
    image.validate()
    anomalies = list(anomalies)
    legend = "".join(
        f'<li><span class="swatch" style="background:{details["color"]}"></span>{html.escape(details["name"])}</li>'
        for details in BRAIN_REGIONS.values()
    )
    rows = []
    for anomaly in anomalies:
        rows.append(
            "<tr>"
            f"<td>{html.escape(anomaly.region)}</td>"
            f"<td>{html.escape(anomaly.anomaly_type)}</td>"
            f"<td>{anomaly.score:.3f}</td>"
            f"<td>{anomaly.voxel_count}</td>"
            f"<td>{html.escape(anomaly.interpretation)}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="5">No experimental observations exceeded the configured threshold.</td></tr>')
    region_counts = {}
    if segments:
        region_counts = {name: int(mask.sum()) for name, mask in segments.items()}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ font-family: system-ui,sans-serif; color:#17202a; background:#f5f7fa; }} body {{ max-width:1100px; margin:0 auto; padding:2rem; }}
section,header {{ background:#fff; border:1px solid #d8e0e8; border-radius:.75rem; padding:1rem; margin:1rem 0; }}
.warning {{ background:#fff4d6; border-left:4px solid #f4a261; padding:.8rem; }} .legend {{ display:flex; flex-wrap:wrap; gap:.8rem; padding:0; list-style:none; }}
.legend li {{ display:flex; align-items:center; gap:.3rem; }} .swatch {{ width:.85rem; height:.85rem; border-radius:50%; display:inline-block; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ border-bottom:1px solid #e5e7eb; text-align:left; padding:.55rem; }} th {{ background:#eef3f8; }} code {{ font-family:monospace; }}
</style></head><body>
<header><h1>{html.escape(title)}</h1><p class="warning"><strong>Research use only:</strong> this output is not for diagnosis, prognosis, treatment recommendation, or clinical decision-making. Every observation requires qualified human review.</p></header>
<section><h2>Image</h2><dl><dt>Patient label</dt><dd>{html.escape(image.patient_id)}</dd><dt>Scan type</dt><dd>{html.escape(image.scan_type)}</dd><dt>Shape</dt><dd>{html.escape(str(list(image.data.shape)))}</dd><dt>Source format</dt><dd>{html.escape(str(image.metadata.get("source_format", "unknown")))}</dd></dl></section>
<section><h2>Region legend</h2><ul class="legend">{legend}</ul></section>
<section><h2>Experimental observations</h2><table><thead><tr><th>Region</th><th>Observation</th><th>Robust score</th><th>Voxels</th><th>Interpretation</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>Region voxel counts</h2><pre>{html.escape(json.dumps(region_counts, indent=2, sort_keys=True))}</pre></section>
</body></html>"""


def write_report(
    output: str | Path,
    image: NeuroImage,
    anomalies: Iterable[Anomaly] = (),
    segments: dict[str, object] | None = None,
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    anomalies = list(anomalies)
    destination.write_text(render_html_report(image, anomalies, segments), encoding="utf-8")
    payload = {
        "image": image.summary(),
        "anomalies": [anomaly.to_dict() for anomaly in anomalies],
        "regions": {name: int(mask.sum()) for name, mask in (segments or {}).items()},
        "safety": "Research use only; not for diagnosis or clinical decision-making.",
    }
    destination.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination
