import os
import glob
import cv2
import json
from ultralytics import YOLO

def run_unseen_inference():
    base_dir = os.path.abspath('Day-38')
    weights_path = os.path.join(base_dir, 'weights', 'best.pt')
    unseen_dir = os.path.join(base_dir, 'unseen_test_images')
    pred_dir = os.path.join(base_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(unseen_dir, exist_ok=True)
    
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}. Please train model first.")
        return
        
    print("="*60)
    print("      DAY 38 INFERENCE ON NEW UNSEEN TEST IMAGES")
    print("="*60)
    print(f"Loading Weights: {weights_path}")
    print(f"Input Directory: {unseen_dir}")
    print(f"Output Directory: {pred_dir}")
    print("-"*60)
    
    model = YOLO(weights_path)
    
    unseen_images = glob.glob(os.path.join(unseen_dir, '*.*'))
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    unseen_images = [p for p in unseen_images if p.lower().endswith(valid_exts)]
    
    print(f"Found {len(unseen_images)} unseen images for testing.")
    
    prediction_results = []
    
    for img_path in unseen_images:
        filename = os.path.basename(img_path)
        
        # Run inference
        results = model.predict(
            source=img_path,
            conf=0.25,
            iou=0.45,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes
        
        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            detections.append({
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': round(conf, 4),
                'bbox_xyxy': [round(v, 2) for v in xyxy]
            })
            
        # Save annotated image
        annotated_bgr = result.plot()
        out_pred_path = os.path.join(pred_dir, f"pred_{filename}")
        cv2.imwrite(out_pred_path, annotated_bgr)
        
        pred_item = {
            'filename': filename,
            'detections_count': len(detections),
            'detections': detections
        }
        prediction_results.append(pred_item)
        
        print(f"Processed [{filename}]: Detected {len(detections)} objects")
        for d in detections:
            print(f"   -> {d['class_name']} ({d['confidence']*100:.1f}%)")
            
    summary_path = os.path.join(base_dir, 'prediction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(prediction_results, f, indent=4)
        
    print("\nInference complete! Results saved to Day-38/predictions/")
    print(f"Summary saved to {summary_path}\n")

if __name__ == '__main__':
    run_unseen_inference()
