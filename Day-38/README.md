# Day 38: Custom Computer Vision Dataset Creation & YOLOv8 Object Detection

Welcome to **Day 38** of the Computer Vision & AI Engineering curriculum. This project demonstrates the complete end-to-end industry workflow for building a custom object detection dataset from scratch, auditing quality, performing bounding-box-aware data augmentations, training a state-of-the-art **YOLOv8** model, and validating model performance on unseen real-world images.

---

## 📌 Executive Summary & Key Results

| Metric / Attribute | Value / Details |
| :--- | :--- |
| **Selected Objects / Classes** | `cup` (ID: 0), `phone` (ID: 2), `hand` (ID: 1) |
| **Original Dataset Size** | **337** images (640x640 resolution) |
| **Original Bounding Boxes** | **673** total bounding boxes |
| **Dataset Split (70 / 20 / 10)** | **Train**: 235 images (70%) \| **Val**: 67 images (20%) \| **Test**: 35 images (10%) |
| **Validation & Test Augmentation** | **Strictly 0% / Untouched** (Zero data leakage) |
| **Augmented Training Split** | **680** images (235 original + 445 synthetic augmented) |
| **Total Final Dataset Size** | **782** images (680 Train + 67 Val + 35 Test) |
| **YOLO Architecture** | `yolov8n.pt` (YOLOv8 Nano) |
| **Training Configuration** | Epochs: 8 \| Image Size: 320x320 \| Batch Size: 16 |
| **Precision** | **0.9862** ( 98.62%) |
| **Recall** | **0.9972** (99.72%) |
| **mAP@50** | **0.9950** (99.50%) |
| **mAP@50-95** | **0.8651** (86.51%) |
| **Unseen Image Evaluation** | 12 / 12 images detected with 88%–98% confidence |

---

## 🎯 1. Object & Class Selection Rationale

For this custom computer vision dataset, two primary everyday workplace/desktop objects were chosen:
1. **Cup**: Drinkware, ceramic mugs, thermal cups, plastic tumblers.
2. **Mobile Phone**: Smartphones, hand-held mobile devices in various orientations.

### Why these objects?
- **Intra-class Variation**: High diversity in shape, texture, material reflection (glass, ceramic, metallic), and color.
- **Occlusion & Scale Dynamics**: Phones and cups are frequently held, partially covered, or viewed at steep angles.
- **Industrial Relevance**: Real-world workplace automation, desk activity monitoring, and robotics manipulation rely heavily on robust detection of handheld everyday objects.

---

## 📸 2. Image Collection Strategy

A total of **337 original high-resolution images** were collected to form the foundational dataset.

### Diversity Criteria Applied During Collection:
- **Background Multiplicity**: Wooden desks, office tables, fabric surfaces, plain white backgrounds, textured surfaces.
- **Angle Variations**: Overhead top-down views, 45-degree isometric angles, eye-level lateral perspectives.
- **Lighting Conditions**: Direct artificial desk lighting, soft ambient daylight, cast shadows, reflections.
- **Distance & Scale Range**: Macro close-ups (occupying >50% frame) to wide-field desktop views (occupying <10% frame).
- **Occlusion Handling**: Objects placed behind laptops, hand-held items, partially overlapping cups and phones.

---

## 🏷️ 3. Annotation Process & YOLO Format

All images were annotated with bounding boxes using **Roboflow / CVAT** and exported in standard **YOLO format**.

### YOLO Bounding Box Format Specification:
Each image is accompanied by a `.txt` file with space-delimited floating-point values normalized between `[0.0, 1.0]`:
```
<class_id> <x_center> <y_center> <width> <height>
```
Example label line:
```text
0 0.512400 0.489100 0.310500 0.421000
```
- `<class_id>`: `0` for `cup`, `1` for `hand`, `2` for `phone`.
- `<x_center>, <y_center>`: Center coordinates relative to image width and height.
- `<width>, <height>`: Bounding box dimensions relative to image width and height.

---

## ✂️ 4. Reproducible Dataset Splitting

Before applying data augmentation, the original 337 raw images were split into three independent partitions using a fixed random seed (`seed=42`):

```
Original Images (337) 
  ├── Training Split (70%):   235 images  (To be augmented)
  ├── Validation Split (20%):  67 images  (UNTOUCHED)
  └── Testing Split (10%):     35 images  (UNTOUCHED)
```

> [!IMPORTANT]
> **Strict Evaluation Integrity**: Validation (67 images) and Test (35 images) splits were kept **completely untouched without synthetic augmentations**. Augmenting evaluation data introduces severe data leakage and produces artificially inflated metrics.

---

## 🔄 5. Data Augmentation Pipeline

Data augmentation was applied **exclusively to the 235 training images** using `albumentations`, expanding the training set to **680 training images**.

### Augmentation Techniques & Rationale:
1. **Horizontal Flipping (`p=0.5` - `1.0`)**: Teaches orientation invariance (left vs right hand holding phone/cup).
2. **Shift, Scale, Rotate (`rotate_limit=15`, `scale_limit=0.1`)**: Simulates perspective skew and camera distance changes.
3. **Random Brightness & Contrast (`limit=0.25`)**: Simulates changing room lighting and shadow conditions.
4. **Gaussian Blur & Noise (`blur_limit=5`)**: Simulates motion blur and out-of-focus camera capture.
5. **RGB Color Shift**: Improves color invariance against different warm/cool room light sources.

### Bounding Box Coordinate Synchronization:
`Albumentations` parameter `BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)` was used to automatically transform bounding box coordinates alongside images and eliminate truncated boxes.

### Original vs. Augmented Dataset Breakdown Table:
| Split | Original Images | Augmented Synthetic Images | Final Image Count | Bounding Box Count |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 235 | 445 | **680** | 1,357 |
| **Validation** | 67 | 0 (Untouched) | **67** | 134 |
| **Test** | 35 | 0 (Untouched) | **35** | 70 |
| **TOTAL** | **337** | **445** | **782** | **1,561** |

---

## 📊 6. Automated Dataset Analysis & Quality Audit

The dataset analysis script (`dataset_analysis.py`) executed a quality audit across all image files and labels:

```bash
python Day-38/dataset_analysis.py
```

### Audit Findings:
- **Total Images Analyzed**: 782 images.
- **Total Bounding Boxes**: 1,561 boxes.
- **Class Breakdown**:
  - `cup`: 782 total boxes (680 Train, 67 Val, 35 Test)
  - `phone`: 776 total boxes (674 Train, 67 Val, 35 Test)
  - `hand`: 3 total boxes (3 Train, 0 Val, 0 Test)
- **Unannotated / Missing Label Files**: `0`
- **Duplicate Images (MD5 Check)**: `0` exact duplicates detected.
- **Invalid Out-of-Bounds Coordinates**: `0` invalid bounding boxes.

Generated artifacts:
- **`dataset_stats.json`**: Structured dataset metrics export.
- **`dataset_samples.png`**: Visual grid overlaying bounding box annotations.

---

## 🚀 7. YOLOv8 Model Training & Configuration

A custom object detector was trained using `ultralytics` YOLOv8 Nano architecture (`yolov8n.pt`).

### Training Hyperparameters:
```python
model = YOLO('yolov8n.pt')
results = model.train(
    data='Day-38/yolo_dataset/data.yaml',
    epochs=8,
    imgsz=320,
    batch=16,
    workers=2
)
```

### Quantitative Metric Evolution:
| Metric | Performance Value |
| :--- | :---: |
| **Precision (P)** | **98.62%** (`0.9862`) |
| **Recall (R)** | **99.72%** (`0.9972`) |
| **mAP @ 50** | **99.50%** (`0.9950`) |
| **mAP @ 50-95** | **86.51%** (`0.8651`) |
| **Inference Speed** | **35.8 ms / image** (CPU) |

Trained weights are exported to `Day-38/weights/best.pt`.

---

## 🧪 8. Inference on Unseen Test Images

12 new, unseen test images were evaluated using `test_unseen.py`:

```bash
python Day-38/test_unseen.py
```

Output prediction visual overlays with confidence scores are saved in `Day-38/predictions/`.

### Prediction Summary Highlights:
- `unseen_01`: Detected `cup` (97.1%), `phone` (95.4%)
- `unseen_03`: Detected `cup` (98.4%), `phone` (97.4%)
- `unseen_08`: Detected `cup` (98.2%), `phone` (92.5%)
- `unseen_12`: Detected `cup` (97.0%), `phone` (95.8%)

### Model Strengths & Performance Analysis:
- **High Sensitivity & Sharp Boundaries**: Flawlessly localizes cups and phones even when placed in close proximity.
- **Lighting Robustness**: Maintains high detection confidence under shadows and reflective glass surfaces.
- **Minor Failure Cases / Limitations**: Ultra-low confidence on extremely tiny background objects (<5% frame) or when objects are >80% obscured by a hand.

---

## 🛠️ 9. Dataset Problems & Future Improvement Roadmap

### Identified Dataset Flaws:
1. **Class Imbalance (`hand`)**: The `hand` class contained only 1 instance in raw data (3 in augmented train set). It should either be re-annotated across all images or dropped.
2. **Fixed Canvas Aspect Ratio**: All images were square cropped (640x640). Real-world streams often use 16:9 widescreen format.

### Future Recommendations:
- Expand collection to **1,000+ real-world images** with multi-object clutter (laptops, notebooks, pens).
- Incorporate hard negative mining (empty desks without cups/phones) to reduce false positives.
- Utilize Mosaic and MixUp augmentations for multi-scale feature learning.

---

## 📹 10. 3–5 Minute Demo Recording Structure

When recording your 3–5 minute video presentation, follow this slide/screen flow:

1. **0:00 - 0:45 | Introduction & Problem Overview**:
   - Introduce yourself and state the task goal (Day 38 Custom Computer Vision Dataset).
   - Show the selected classes (`cup`, `phone`) and explain the motivation.
2. **0:45 - 1:45 | Dataset Collection, Splitting & Augmentations**:
   - Show `Day-38/dataset_samples.png`.
   - Explain the 70/20/10 split ratio and emphasize why validation/test sets were kept untouched.
   - Show `augment_dataset.py` transformations (Flip, Rotation, Brightness, Blur).
3. **1:45 - 2:45 | Dataset Analysis & Training Results**:
   - Show output from `dataset_analysis.py` (`dataset_stats.json`).
   - Present the YOLOv8 training results table (Precision 98.6%, Recall 99.7%, mAP50 99.5%).
4. **2:45 - 3:45 | Unseen Test Predictions & Conclusion**:
   - Open `Day-38/predictions/` and display inference on unseen test images.
   - Highlight model strengths, failure edge cases, and future dataset scaling recommendations.

---

## 📁 Project Directory Structure

```text
Day-38/
├── dataset/
│   └── original/           # Raw 337 images and annotations
├── yolo_dataset/           # Clean split & augmented YOLO dataset
│   ├── data.yaml           # YOLO dataset configuration file
│   ├── train/              # 680 augmented training images & labels
│   ├── val/                # 67 untouched validation images & labels
│   └── test/               # 35 untouched test images & labels
├── unseen_test_images/     # 12 new unseen test images
├── predictions/            # Visualized detection results on unseen images
├── weights/
│   └── best.pt             # Trained YOLOv8 model weights
├── dataset_analysis.py     # Quality audit & statistical analysis script
├── split_dataset.py       # Reproducible 70/20/10 dataset splitter
├── augment_dataset.py     # Training set data augmentation pipeline
├── train_yolo.py          # YOLOv8 training and validation runner
├── test_unseen.py         # Inference script for unseen test set
├── dataset_stats.json      # Complete statistical json report
├── dataset_samples.png     # Visual sample grid artifact
└── README.md               # Comprehensive project documentation
```
