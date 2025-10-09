# Egyptian ID Card Data Extraction System

A complete computer vision pipeline for detecting and extracting text from Egyptian ID cards using YOLOv8 object detection and OCR technologies.

## Project Overview

This system automatically processes Egyptian ID card images to:
1. Detect specific regions of interest (name, address, ID number, etc.)
2. Remove overlapping detections
3. Extract Arabic text from detected regions
4. Visualize results with bounding boxes

## Features

- **Object Detection**: YOLOv8-based detection of ID card fields
- **Overlap Removal**: NMS-style filtering to remove duplicate detections
- **Arabic OCR**: EasyOCR integration for Arabic text extraction
- **Visualization**: Annotated output images with bounding boxes and labels
- **Modular Design**: Clean, reusable functions for each processing step

## Dataset

The dataset was sourced from **Roboflow** and contains labeled Egyptian ID card images with annotations for:
- Code
- Image (photo)
- City
- Family name
- Name
- Neighborhood
- Number
- State

**Dataset Structure:**
```
egyptian-id-seg-1/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Requirements

### Python Packages
```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install easyocr
```

### System Requirements
- Python 3.8+
- 4GB+ RAM recommended
- GPU optional (for faster processing)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd CY_CV_ID_PROJECT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the trained model weights and place in `model_Parameters/`:
```
model_Parameters/
└── best.pt
```

## Project Structure

```
CY_CV_ID_PROJECT/
├── model_Parameters/
│   └── best.pt                 # Trained YOLOv8 model
├── Data_Extraction.py          # Main script for text extraction
├── element_Extraction.py       # Script for visualization only
├── TEST_ID.jpg                 # Sample input image
├── output_with_boxes.jpg       # Output with bounding boxes
└── README.md
```

## Usage

### 1. Extract Text from ID Card

```python
python Data_Extraction.py
```

This will:
- Load the YOLO model
- Detect regions in the ID card
- Remove overlapping detections
- Extract Arabic text using OCR
- Print extracted text with confidence scores

**Output:**
```
Processing Crop 0: shape (41, 173, 3)
  Found: 'محمد أحمد' (conf: 0.95)

FINAL EXTRACTED TEXT:
==================================================
1. Text: 'محمد أحمد' | Confidence: 0.95
2. Text: '29012345678901' | Confidence: 0.92
...
```

### 2. Visualize Detections Only

```python
python element_Extraction.py
```

This will:
- Detect and filter bounding boxes
- Draw colored boxes with labels
- Save annotated image to `output_with_boxes.jpg`

### 3. Custom Usage

```python
from Data_Extraction import B_Box_extracion, overlap_removal, CROP, text_Extraction
import cv2

# Load image
img = cv2.imread("path/to/id_card.jpg")

# Extract and filter bounding boxes
bbox_list = B_Box_extracion(img)
filtered_bbox_list = overlap_removal(bbox_list)

# Crop regions
cropped_images = CROP(img, filtered_bbox_list)

# Extract text
extracted_text = text_Extraction(cropped_images)

# Print results
for item in extracted_text:
    print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")
```

## Model Training

The YOLOv8 model was trained using transfer learning:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Freeze early layers
for param in list(model.model.parameters())[:-20]:
    param.requires_grad = False

# Train
results = model.train(
    data="egyptian-id-seg-1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.001,
    project="ID_TRAINING_SUMMARY",
    name="id-transfer-learning"
)
```

**Training Parameters:**
- Base model: YOLOv8n (nano)
- Epochs: 100
- Image size: 640x640
- Batch size: 16
- Learning rate: 0.001
- Augmentation: Conservative (suitable for ID cards)

## Core Functions

### `B_Box_extracion(img)`
Detects objects in the image using YOLO model.
- **Input**: Image array
- **Output**: List of `[confidence, class_id, ((x1, y1), (x2, y2))]`

### `overlap_removal(objects_list)`
Removes overlapping bounding boxes of the same class using IoU threshold.
- **Input**: List of detections
- **Output**: Filtered list without overlaps
- **Threshold**: IoU > 0.5

### `CROP(original_img, final_detection_list)`
Crops regions from image based on bounding boxes.
- **Input**: Original image, list of bounding boxes
- **Output**: List of cropped image arrays

### `text_Extraction(Filtered_list)`
Extracts Arabic text from cropped images using EasyOCR.
- **Input**: List of cropped images
- **Output**: List of `{"text": str, "confidence": float}`

### `draw_bounding_boxes(img, bbox_list, class_names)`
Draws colored bounding boxes with labels on image.
- **Input**: Image, bounding boxes, class names dictionary
- **Output**: Annotated image

## Class Mapping

```python
{
    0: 'Code',
    1: 'Image',
    2: 'city',
    3: 'family name',
    4: 'name',
    5: 'neighborhood',
    6: 'number',
    7: 'state'
}
```

## Troubleshooting

### Issue: EasyOCR fails to download models
**Solution**: Check internet connection or use retry logic in the code.

### Issue: Low OCR accuracy
**Solution**: 
- Ensure cropped images are clear and high resolution
- Try preprocessing (grayscale, thresholding, denoising)
- Consider fine-tuning OCR parameters

### Issue: Multiple overlapping detections
**Solution**: Adjust IoU threshold in `overlap_removal()` function (default: 0.5)

### Issue: OpenCV display error
**Solution**: The visualization script saves images directly without displaying them.

## Performance

- **Detection Speed**: ~100ms per image (CPU)
- **OCR Speed**: ~1-2s per crop (CPU)
- **Accuracy**: Depends on image quality and lighting

## Future Improvements

- [ ] Add preprocessing pipeline for image enhancement
- [ ] Implement post-processing for text correction
- [ ] Add support for batch processing
- [ ] Create web interface for easy deployment
- [ ] Fine-tune OCR specifically for Egyptian ID fonts
- [ ] Add validation logic for ID number format

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

This project is for educational and research purposes.

## Acknowledgments

- **Roboflow** for dataset hosting and annotation tools
- **Ultralytics** for YOLOv8 implementation
- **EasyOCR** for Arabic text recognition
- **OpenCV** for image processing utilities

 ## Output
 
<img width="605" height="447" alt="v5" src="https://github.com/user-attachments/assets/6172cce9-0cda-4817-9136-8c595dbc3e0e" />
<img width="788" height="472" alt="v2" src="https://github.com/user-attachments/assets/5650a215-27fe-4ba3-9893-bdb4c30f0aee" />

## Contact

For questions or issues, please open an issue on GitHub or contact the project maintainer.

---

**Note**: This system is designed for research and development purposes. Ensure compliance with data privacy regulations when processing real ID card images.
