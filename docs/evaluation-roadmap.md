# Research evaluation contract

`neurapedia.evaluation` accepts independent binary masks for Dice, IoU, false-positive/negative voxel counts; empty paired masks return Dice/IoU 1 and flag the empty reference. Finite foreground probabilities produce Brier score, fixed-bin ECE and mean binary entropy. Entropy is a descriptive quantity, not proof of calibrated epistemic uncertainty.

Reproduce: `python -m pytest tests/test_evaluation.py -q`. Current fixtures verify calculations only. Real segmentation performance, patient-disjoint splits, licensed masks, scanner/motion/noise shifts, held-out OOD analysis and uncertainty validation remain pending. No new model was trained and no diagnostic accuracy is claimed.
