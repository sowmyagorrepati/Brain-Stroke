---
title: Brain Stroke Detection
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Brain Stroke Detection with Grad-CAM

This Space provides a **FastAPI-based backend** for detecting brain stroke from CT scan images.

### Features
- Classifies CT scans into:
  - Normal
  - Ischemia
  - Hemorrhage
- Generates **Grad-CAM heatmaps** for explainability
- Designed for clinical decision support (research use)

### API
- `POST /predict`
  - Input: CT scan image
  - Output:
    - Prediction
    - Confidence
    - Grad-CAM visualization
