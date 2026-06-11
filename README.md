# Neurapedia - Neuroscience Imaging Pipeline

**Neurapedia** is a neuroscience-focused machine learning pipeline for brain imaging analysis, segmentation, and anomaly detection.

## ✅ Core Features

### Neuroscience Pipeline
- **Brain region segmentation**: 10 color-coded regions
- **Neuroimaging formats**: NIfTI, DICOM (placeholder support)
- **Anomaly detection**: Statistical/ML-based detection
- **Clinical reporting**: Professional-grade reports
- **Visualization support**: Region mapping for 3D viewers

### ML Models
- Brain tumor detection (CNN architecture planned)
- Anomaly classification
- fMRI activity prediction
- Segmentation models (nnU-Net integration)

## 📦 Project Structure
```
neurapedia/
├── src/
│   ├── neuroscience/      # Core neuroscience pipeline
│   ├── models/            # ML models
│   ├── api/               # FastAPI backend
│   └── utils/             # Utilities
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── examples/              # Demo scripts
├── docs/                  # Documentation
└── requirements.txt
```

## 🔬 Brain Regions (10 Color-coded)
| Region | Color | Function |
|--------|-------|----------|
| Frontal Lobe | 🔴 Red (`#FF0000`) | Decision making, Planning, Personality, Motor control |
| Parietal Lobe | 🔵 Blue (`#0000FF`) | Touch, Sensory integration, Spatial awareness |
| Temporal Lobe | 🟢 Green (`#00FF00`) | Hearing, Language, Memory |
| Occipital Lobe | 🟡 Yellow (`#FFFF00`) | Vision, Visual processing |
| Cerebellum | 🟣 Purple (`#800080`) | Coordination, Balance, Fine motor control |
| Brainstem | 🟠 Orange (`#FFA500`) | Vital functions, Breathing, Heart rate |
| Hippocampus | 🩷 Pink (`#FF69B4`) | Memory formation, Spatial navigation |
| Amygdala | 🔴 Red-Orange (`#FF4500`) | Emotion, Fear response |
| Thalamus | 💧 Teal (`#00CED1`) | Sensory relay, Consciousness |
| Basal Ganglia | 🟢 Lime (`#32CD32`) | Motor control, Habit formation, Reward |

## 🚀 Quick Start
```bash
cd ~/Desktop/neurapedia
pip install -r requirements.txt

# Run a demo
python examples/demo_neuroscience.py

# Run tests
python -m pytest tests/
```

## 🧠 Testing
Tests cover:
- Synthetic MRI generation
- Brain segmentation
- Anomaly detection
- Clinical report generation

```bash
python -m pytest tests/test_neuroscience.py -v
```

## 🧩 Segmentation Example
```python
from src.neuroscience import BrainSegmenter
segmenter = BrainSegmenter()
mri_image = loader.load_nifti("synthetic")
segments = segmenter.segment_lobes(mri_image)
# Returns dict mapping region names to boolean masks
```

## 🩺 Clinical Reporting
Generates professional-grade reports for physician review:

```python
from src.neuroscience import AnomalyDetector
detector = AnomalyDetector()
anomalies = detector.detect(mri_image)
report = detector.generate_clinical_report(anomalies)
# Includes: location, type, confidence, description, suggested actions
```

## 🌐 API (Planned)
FastAPI backend for:
- Neuroimaging upload endpoints
- Segmentation services
- Anomaly detection API
- Real-time 3D visualization support

## 🔗 Cloud Dataset Access
No large files stored locally - access datasets via:
- **Human Connectome Project**: https://www.humanconnectome.org/
- **Allen Brain Atlas**: https://portal.brain-map.org/
- **Open Connectome Project**: https://openconnecto.me/
- **BrainGraph Database**: https://braingraph.org/
- **Allen Brain Cell Atlas**: https://portal.brain-map.org/atlases-and-data/bkp/abcatlas

## 🧠 Neuroscience Context
Based on:
- [ChatGPT conversation](https://chatgpt.com/share/6a2aca3e-7334-83ea-9f30-dbdccc4c03a2)

## 🔮 Future Enhancements
1. Deep learning-based tumor detection
2. fMRI connectivity analysis
3. Multi-modal fusion (MRI + fMRI + EEG)
4. Real-time neuroimaging dashboard
5. Integration with clinical imaging devices

## 📜 License
MIT License - see [LICENSE](LICENSE) ```

---
© 2026 Neurapedia Project | [GitHub](#)