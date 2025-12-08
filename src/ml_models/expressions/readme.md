# Preprocessing Alignment Tool User Guide

## Step 1: Preprocess the Dataset

### Dataset Preprocessing
```bash
# Perform face alignment preprocessing on the original dataset
python preprocess_align.py --input /path/to/original_dataset --output /path/to/aligned_dataset

# Example
python preprocess_align.py --input ./data_original --output ./data_aligned
```

### Processing Workflow
- The script automatically detects all category folders
- Performs face alignment on every image
- Saves the aligned images into the new dataset folder
- Displays processing progress and success/failure statistics

### Input Dataset Structure
```
data_original/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    ├── happy/
    └── ...
```

### Output Dataset Structure
```
data_aligned/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    ├── happy/
    └── ...
```

## Step 2: Train the Model

### Train Using the Preprocessed Dataset
```bash
# Train using the aligned dataset
python train.py --dataset ./data_aligned --epochs 30 --batch_size 32 --lr 0.001
```

### Advantages
- **Greatly improved training speed** — no real-time alignment required
- **Consistent alignment quality** — data is already aligned
- **Lower resource usage** — MediaPipe not needed during training

## Command Examples

### Full Workflow Example
```bash
# 1. Install dependencies
pip install timm mediapipe tqdm

# 2. Preprocess the dataset
python preprocess_align.py --input ./data_original --output ./data_aligned

# 3. Train the model
python train.py --dataset ./data_aligned --epochs 30 --batch_size 32 --lr 0.001

# 4. Test the model
python test_detection.py
```

## Notes

1. **Preprocessing time** — First preprocessing may take some time but only needs to be done once
2. **Storage space** — Extra space is required for the aligned dataset
3. **MediaPipe dependency** — Required for preprocessing, not for training
4. **Inference alignment** — `inference.py` still performs alignment for real-time inference

## FAQ

### Q: What happens to images that fail during preprocessing?
A: They are automatically copied from the original dataset to the output directory to ensure dataset completeness.

### Q: Can I preprocess only part of the dataset?
A: Yes. The script processes all category folders under the input directory.
