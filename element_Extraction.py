import ultralytics
import cv2
from ultralytics import YOLO
import numpy as np


def draw_bounding_boxes(img, bbox_list, class_names=None):
    """
    Draw bounding boxes on the image
    
    Args:
        img: Original image
        bbox_list: List of [conf, cls, ((x1, y1), (x2, y2))]
        class_names: Dictionary mapping class IDs to names (optional)
    
    Returns:
        Image with bounding boxes drawn
    """
    # Create a copy of the image
    img_with_boxes = img.copy()
    
    # Define colors for different classes (BGR format)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    for bbox in bbox_list:
        conf = bbox[0]
        cls = bbox[1]
        x1, y1 = map(int, bbox[2][0])
        x2, y2 = map(int, bbox[2][1])
        
        # Select color based on class
        color = colors[cls % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if class_names and cls in class_names:
            label = f"{class_names[cls]}: {conf:.2f}"
        else:
            label = f"Class {cls}: {conf:.2f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img_with_boxes


def B_Box_extracion(img):
    output = model.predict(img)
    objects = []
    
    for results in output:
        boxes = results.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            bounding_box = [conf, cls, ((x1, y1), (x2, y2))]
            objects.append(bounding_box)
    
    return objects


def overlap_removal(objects_list):
    if len(objects_list) == 0:
        return []
    
    sorted_objects = sorted(objects_list, key=lambda x: x[0], reverse=True)
    filtered_objects = []
    
    for current_obj in sorted_objects:
        keep = True
        current_conf, current_cls, current_coords = current_obj
        x1_curr, y1_curr = current_coords[0]
        x2_curr, y2_curr = current_coords[1]
        
        for kept_obj in filtered_objects:
            kept_conf, kept_cls, kept_coords = kept_obj
            
            if current_cls != kept_cls:
                continue
            
            x1_kept, y1_kept = kept_coords[0]
            x2_kept, y2_kept = kept_coords[1]
            
            x1_inter = max(x1_curr, x1_kept)
            y1_inter = max(y1_curr, y1_kept)
            x2_inter = min(x2_curr, x2_kept)
            y2_inter = min(y2_curr, y2_kept)
            
            inter_width = max(0, x2_inter - x1_inter)
            inter_height = max(0, y2_inter - y1_inter)
            inter_area = inter_width * inter_height
            
            curr_area = (x2_curr - x1_curr) * (y2_curr - y1_curr)
            kept_area = (x2_kept - x1_kept) * (y2_kept - y1_kept)
            union_area = curr_area + kept_area - inter_area
            
            if union_area > 0:
                iou = inter_area / union_area
                
                if iou > 0.5:
                    keep = False
                    break
        
        if keep:
            filtered_objects.append(current_obj)
    
    return filtered_objects


if __name__ == "__main__":
    # Load YOLO model
    model = YOLO("model_Parameters/best.pt")
    
    # Load image
    img = cv2.imread("C:/CY_CV_ID_PROJECT/TEST_ID.jpg")
    
    # Get class names from model (if available)
    class_names = model.names if hasattr(model, 'names') else None
    print(f"Class names: {class_names}")
    
    # Extract bounding boxes
    bbox_list = B_Box_extracion(img)
    print(f"Detected {len(bbox_list)} objects")
    
    # Remove overlapping boxes
    filtered_bbox_list = overlap_removal(bbox_list)
    print(f"After overlap removal: {len(filtered_bbox_list)} objects")
    
    # Draw bounding boxes on image
    img_with_boxes = draw_bounding_boxes(img, filtered_bbox_list, class_names)
    
    # Save the image
    output_path = "C:/CY_CV_ID_PROJECT/output_with_boxes.jpg"
    cv2.imwrite(output_path, img_with_boxes)
    print(f"\nImage saved to: {output_path}")
    print("Open the file to view the detected bounding boxes!")