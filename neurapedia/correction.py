"""Evidence-gated imaging correction and QC planning.

The planner suggests reproducible technical review steps. It does not alter
voxels, infer a disease, or recommend patient care.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .core import Anomaly, NeuroImage
from .references import ReferencePlan


@dataclass(frozen=True)
class CorrectionSuggestion:
    category: str
    action: str
    evidence: tuple[str, ...]
    status: str = "requires_human_review"
    reference_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "action": self.action,
            "evidence": list(self.evidence),
            "status": self.status,
            "reference_ids": list(self.reference_ids),
            "safety": "Algorithmic QC guidance only; no voxel edit, diagnosis, or treatment recommendation.",
        }


class CorrectionPlanner:
    """Map observable QC signals to conservative, reproducible next steps."""

    _ACTIONS = {
        "motion": "review motion correction and registration parameters, then compare against a matched QC cohort",
        "ghosting": "inspect phase-encoding direction and ghosting metrics before accepting downstream analysis",
        "susceptibility": "review distortion correction and field-map compatibility for the affected sequence",
        "distortion": "verify susceptibility/gradient correction and retain the uncorrected source for comparison",
        "bias_field": "run a validated bias-field comparison and report intensity changes separately from anatomy",
        "noise": "review denoising assumptions and compare signal-to-noise metrics before reconstruction",
        "aliasing": "inspect sampling and reconstruction settings against an appropriate k-space benchmark",
        "undersampling": "compare reconstruction choices against fully sampled reference data where permitted",
        "wrong_orientation": "verify orientation metadata and re-run geometry validation before resampling",
    }

    def plan(
        self,
        image: NeuroImage,
        anomalies: Iterable[Anomaly] = (),
        *,
        reference_plan: ReferencePlan | None = None,
    ) -> list[CorrectionSuggestion]:
        image.validate()
        reference_ids = reference_plan.source_ids if reference_plan else ()
        flags = image.metadata.get("artifact_flags", ())
        if isinstance(flags, str):
            flags = (flags,)
        suggestions: list[CorrectionSuggestion] = []
        for raw_flag in flags:
            category = str(raw_flag).lower().replace(" ", "_")
            action = self._ACTIONS.get(
                category,
                "inspect the acquisition and preprocessing metadata, then compare against a matched artifact cohort",
            )
            suggestions.append(
                CorrectionSuggestion(
                    category=category,
                    action=action,
                    evidence=("artifact flag supplied in image metadata",),
                    reference_ids=reference_ids,
                )
            )
        anomaly_list = list(anomalies)
        if anomaly_list:
            categories = sorted({anomaly.anomaly_type for anomaly in anomaly_list})
            suggestions.append(
                CorrectionSuggestion(
                    category="intensity_or_difference_observation",
                    action="inspect registration, intensity normalization, and acquisition compatibility before any correction",
                    evidence=(
                        f"{len(anomaly_list)} deterministic anomaly observation(s): {', '.join(categories)}",
                        "reference similarity and anomaly score are not calibrated clinical probabilities",
                    ),
                    reference_ids=reference_ids,
                )
            )
        if not suggestions:
            suggestions.append(
                CorrectionSuggestion(
                    category="no_actionable_qc_signal",
                    action="retain the source image and record the acquisition metadata used for future comparison",
                    evidence=("no configured artifact flag or anomaly observation",),
                    reference_ids=reference_ids,
                )
            )
        return suggestions
