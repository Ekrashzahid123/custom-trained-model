import os
import glob
import json
import hashlib
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

CLASS_NAMES = {0: 'cup', 1: 'hand', 2: 'phone'}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 165, 0), 2: (255, 0, 0)} # BGR: cup=green, hand=orange, phone=blue

def compute_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def analyze_split(split_dir):
    img_dir = os.path.join(split_dir, 'images')
    lbl_dir = os.path.join(split_dir, 'labels')
    
    img_paths = glob.glob(os.path.join(img_dir, '*.*'))
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_paths = [p for p in img_paths if p.lower().endswith(valid_exts)]
    
    class_counts = Counter()
    unannotated_images = []
    empty_label_files = []
    invalid_bbox_count = 0
    total_bboxes = 0
    hashes = {}
    duplicates = []
    
    for img_path in img_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
        
        # Hash check for duplicates
        file_hash = compute_md5(img_path)
        if file_hash in hashes:
            duplicates.append((img_path, hashes[file_hash]))
        else:
            hashes[file_hash] = img_path
            
        if not os.path.exists(lbl_path):
            unannotated_images.append(img_path)
            continue
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        if len(lines) == 0:
            empty_label_files.append(lbl_path)
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                
                if x < 0 or x > 1 or y < 0 or y > 1 or w <= 0 or w > 1 or h <= 0 or h > 1:
                    invalid_bbox_count += 1
                    
                class_counts[cls_id] += 1
                total_bboxes += 1
                
    return {
        'total_images': len(img_paths),
        'total_bboxes': total_bboxes,
        'class_counts': {CLASS_NAMES.get(k, str(k)): v for k, v in class_counts.items()},
        'unannotated_images': len(unannotated_images),
        'empty_label_files': len(empty_label_files),
        'invalid_bboxes': invalid_bbox_count,
        'duplicate_images_count': len(duplicates),
        'duplicates_list': [d[0] for d in duplicates[:5]]
    }

def draw_bboxes_on_image(img_path, lbl_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w, _ = img.shape
    
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                
                color = CLASS_COLORS.get(cls_id, (0, 255, 255))
                label_text = CLASS_NAMES.get(cls_id, str(cls_id))
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label_text, (x1, max(15, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def generate_visual_sample_grid(dataset_dir, output_path, num_samples=6):
    train_img_dir = os.path.join(dataset_dir, 'train', 'images')
    train_lbl_dir = os.path.join(dataset_dir, 'train', 'labels')
    
    img_paths = sorted(glob.glob(os.path.join(train_img_dir, '*.*')))
    if not img_paths:
        return
        
    random.seed(42)
    sample_paths = random.sample(img_paths, min(num_samples, len(img_paths)))
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Day 38 Custom Dataset - Annotated Samples (Train Split)", fontsize=14, fontweight='bold')
    
    for idx, img_path in enumerate(sample_paths):
        ax = axes[idx // 3, idx % 3]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(train_lbl_dir, f"{base_name}.txt")
        
        annotated_img = draw_bboxes_on_image(img_path, lbl_path)
        if annotated_img is not None:
            ax.imshow(annotated_img)
            ax.set_title(base_name[:20] + "...", fontsize=9)
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample visual grid saved to {output_path}")

def run_dataset_analysis():
    base_dir = os.path.abspath('Day-38')
    dataset_dir = os.path.join(base_dir, 'yolo_dataset')
    
    stats = {}
    total_dataset_images = 0
    total_dataset_boxes = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.exists(split_dir):
            split_stats = analyze_split(split_dir)
            stats[split] = split_stats
            total_dataset_images += split_stats['total_images']
            total_dataset_boxes += split_stats['total_bboxes']
            
    stats['summary'] = {
        'total_dataset_images': total_dataset_images,
        'total_dataset_bboxes': total_dataset_boxes,
        'class_names': CLASS_NAMES,
        'split_ratio': {
            'train_pct': round((stats.get('train', {}).get('total_images', 0) / total_dataset_images) * 100, 2),
            'val_pct': round((stats.get('val', {}).get('total_images', 0) / total_dataset_images) * 100, 2),
            'test_pct': round((stats.get('test', {}).get('total_images', 0) / total_dataset_images) * 100, 2)
        }
    }
    
    # Save statistics JSON
    json_path = os.path.join(base_dir, 'dataset_stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Dataset stats saved to {json_path}")
    
    # Save sample image grid
    sample_grid_path = os.path.join(base_dir, 'dataset_samples.png')
    generate_visual_sample_grid(dataset_dir, sample_grid_path)
    
    # Print formatted console report
    print("\n" + "="*50)
    print("      DAY 38 DATASET ANALYSIS REPORT")
    print("="*50)
    print(f"Total Dataset Images: {total_dataset_images}")
    print(f"Total Bounding Boxes: {total_dataset_boxes}")
    print("-"*50)
    for split, s in stats.items():
        if split == 'summary': continue
        print(f"[{split.upper()} SPLIT]")
        print(f"  Images: {s['total_images']}")
        print(f"  Bounding Boxes: {s['total_bboxes']}")
        print(f"  Class Breakdown: {s['class_counts']}")
        print(f"  Unannotated Images: {s['unannotated_images']}")
        print(f"  Duplicates Detected: {s['duplicate_images_count']}")
        print(f"  Invalid Bounding Boxes: {s['invalid_bboxes']}")
    print("="*50 + "\n")

if __name__ == '__main__':
    run_dataset_analysis()
