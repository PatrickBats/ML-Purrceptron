# From-Scratch CNN Training

This folder contains everything for training a 5-layer CNN from scratch on cat breed classification.

## Architecture

- 5 convolutional layers (64 → 128 → 256 → 512 → 512)
- Batch normalization after each conv layer
- Strong dropout (0.5, 0.3) for regularization
- ~13M parameters

## Training

```bash
cd from_scratch
python train.py
```

**Expected results:**
- Validation accuracy: ~60-80%
- Training time: ~30-60 minutes on RTX 4070
- Checkpoints saved to `experiments/from_scratch_5layer/checkpoints/`

## Visualization

### GradCAM (Standard)
```bash
python visualize_gradcam.py
```

Creates heatmaps showing what the model focuses on. Outputs to `gradcam_visualizations/`.

### SplineCAM (Smoother)
```bash
python visualize_splinecam.py
```

Creates smoother, more detailed heatmaps using spline interpolation. Outputs to `splinecam_visualizations/`.

## Loading Trained Model

```python
from load_model import load_model, predict_image

# Load model
model, breed_names, config = load_model()

# Make prediction
breed, confidence, probs = predict_image('path/to/cat.jpg', model, breed_names)
print(f"Predicted: {breed} ({confidence:.2%})")
```

## Files

- `train.py` - Main training script
- `load_model.py` - Load trained model for inference
- `visualize_gradcam.py` - GradCAM visualization
- `visualize_splinecam.py` - SplineCAM visualization
- `experiments/` - Training outputs (checkpoints, metrics)
