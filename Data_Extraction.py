import ultralytics
import cv2
from ultralytics import YOLO
import json
import numpy as np
from paddleocr import PaddleOCR


class object:
    def __init__(self,cls,x1,y1,x2,y2):
        self.cls=cls
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
        

def CROP(original_img, final_detection_list):
    cropped_B_BBox = []
    
    for object_element in final_detection_list:
        conf = object_element[0]
        cls = object_element[1]
        x1, y1 = object_element[2][0]
        x2, y2 = object_element[2][1]
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        h, w = original_img.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        cropped_img = original_img[y1:y2, x1:x2]
        
        if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
            cropped_B_BBox.append(cropped_img)
    
    return cropped_B_BBox
        

def B_Box_extracion(img):
    output = model.predict(img)
    print(output)
    objects = []
    print("**" * 10)
    for results in output:
        print("__ITRATION__")
        boxes = results.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            print(f"Class: {cls}, Confidence: {conf:.2f}, BBox: ({x1}, {y1}, {x2}, {y2})")
            bounding_box = [conf, cls, ((x1, y1), (x2, y2))] 
            objects.append(bounding_box)
    
    filtered_objects = overlap_removal(objects)
    print(f"After overlap removal: {len(filtered_objects)}")
    
    return filtered_objects  # ADD THIS LINE
            
            
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
    
    
def text_Extraction(Filtered_list):
    output = []
    for i, image_np in enumerate(Filtered_list):
        try:
            # Call the predict method directly (no extra parameters)
            results = ocr.predict(image_np)
            
            if results and results[0]:
                for line in results[0]:
                    text = line[1][0]       # extracted text
                    conf = line[1][1]       # confidence
                    output.append({"text": text, "confidence": conf})
                    print(f"Crop {i}: {text} (conf: {conf:.2f})")
            else:
                print(f"Crop {i}: No text detected")
        except Exception as e:
            print(f"Error processing crop {i}: {e}")
            continue

    return output
 

if __name__=="__main__":
    model = YOLO("model_Parameters/best.pt")
    ocr = PaddleOCR(lang='ar')
    img = cv2.imread("C:/CY_CV_ID_PROJECT/TEST_ID.jpg")
    filtered_bbox_list = B_Box_extracion(img)
    filtered_cropped = CROP(img, filtered_bbox_list)
    
    # Extract text from cropped images
    extracted_text = text_Extraction(filtered_cropped)
    
    # Print results
    print("\nExtracted Text:")
    for item in extracted_text:
        print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")