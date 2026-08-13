import os
import time
import json
import shutil
from ultralytics import YOLO

def train_custom_yolo():
    base_dir = os.path.abspath('Day-38')
    data_yaml_path = os.path.join(base_dir, 'yolo_dataset', 'data.yaml')
    weights_dir = os.path.join(base_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    model_name = 'yolov8n.pt'
    epochs = 8
    imgsz = 320
    batch_size = 16
    
    print("="*60)
    print("      STARTING DAY 38 YOLO MODEL TRAINING")
    print("="*60)
    print(f"Model Architecture: {model_name}")
    print(f"Dataset Config: {data_yaml_path}")
    print(f"Epochs: {epochs} | Image Size: {imgsz} | Batch Size: {batch_size}")
    print("-"*60)
    
    start_time = time.time()
    
    # Initialize YOLOv8 Nano model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=2,
        project=os.path.join(base_dir, 'runs'),
        name='day38_cup_phone_model',
        exist_ok=True,
        verbose=True
    )
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{minutes}m {seconds}s"
    
    print("\nTraining completed in:", time_str)
    
    # Validate the model on the untouched test split
    print("\nRunning evaluation on untouched test set...")
    metrics = model.val(data=data_yaml_path, split='test')
    
    precision = float(metrics.results_dict.get('metrics/precision(B)', 0.0))
    recall = float(metrics.results_dict.get('metrics/recall(B)', 0.0))
    map50 = float(metrics.results_dict.get('metrics/mAP50(B)', 0.0))
    map50_95 = float(metrics.results_dict.get('metrics/mAP50-95(B)', 0.0))
    
    metrics_summary = {
        'model': model_name,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch_size': batch_size,
        'training_time_seconds': round(elapsed_time, 2),
        'training_time_formatted': time_str,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'mAP50': round(map50, 4),
        'mAP50_95': round(map50_95, 4)
    }
    
    # Copy best weights to Day-38/weights/best.pt
    best_weights_src = os.path.join(base_dir, 'runs', 'day38_cup_phone_model', 'weights', 'best.pt')
    best_weights_dst = os.path.join(weights_dir, 'best.pt')
    
    if os.path.exists(best_weights_src):
        shutil.copy2(best_weights_src, best_weights_dst)
        print(f"Saved best model weights to: {best_weights_dst}")
    else:
        print(f"Warning: {best_weights_src} not found.")
        
    metrics_json_path = os.path.join(base_dir, 'training_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("\n" + "="*50)
    print("       FINAL EVALUATION METRICS REPORT")
    print("="*50)
    print(f"Model:            {model_name}")
    print(f"Epochs:           {epochs}")
    print(f"Training Time:    {time_str}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"mAP@50:           {map50:.4f}")
    print(f"mAP@50-95:        {map50_95:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    train_custom_yolo()
