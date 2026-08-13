import os
import glob
import cv2
import shutil
import numpy as np
import albumentations as A

def read_yolo_labels(label_path):
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Clip values to ensure strictly valid bounding box range [0, 1]
                x_center = min(max(x_center, 0.001), 0.999)
                y_center = min(max(y_center, 0.001), 0.999)
                width = min(max(width, 0.001), 0.999)
                height = min(max(height, 0.001), 0.999)
                
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(cls_id)
    return bboxes, class_labels

def write_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            x, y, w, h = bbox
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def get_augmentation_pipelines():
    pipeline_1 = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
    pipeline_2 = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.6),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
    pipeline_3 = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.HorizontalFlip(p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
    return [pipeline_1, pipeline_2, pipeline_3]

def augment_training_set(train_dir, target_count=680):
    img_dir = os.path.join(train_dir, 'images')
    lbl_dir = os.path.join(train_dir, 'labels')
    
    image_paths = sorted(glob.glob(os.path.join(img_dir, '*.*')))
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [p for p in image_paths if p.lower().endswith(valid_exts)]
    
    original_count = len(image_paths)
    print(f"Starting training dataset augmentation. Initial count: {original_count} images.")
    
    pipelines = get_augmentation_pipelines()
    num_augs_per_image = max(1, (target_count - original_count) // original_count)
    
    total_generated = 0
    aug_idx = 0
    
    while (original_count + total_generated) < target_count:
        for img_path in image_paths:
            if (original_count + total_generated) >= target_count:
                break
                
            base_name, ext = os.path.splitext(os.path.basename(img_path))
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bboxes, class_labels = read_yolo_labels(lbl_path)
            if not bboxes:
                continue
                
            pipeline = pipelines[aug_idx % len(pipelines)]
            aug_idx += 1
            
            try:
                transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_classes = transformed['class_labels']
                
                if len(aug_bboxes) > 0:
                    aug_base_name = f"{base_name}_aug_{total_generated+1}"
                    out_img_path = os.path.join(img_dir, f"{aug_base_name}{ext}")
                    out_lbl_path = os.path.join(lbl_dir, f"{aug_base_name}.txt")
                    
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(out_img_path, aug_image_bgr)
                    write_yolo_labels(out_lbl_path, aug_bboxes, aug_classes)
                    
                    total_generated += 1
            except Exception as e:
                # Fallback if bbox transformation hits edge case
                continue
                
    final_count = len(glob.glob(os.path.join(img_dir, '*.*')))
    print(f"Augmentation complete! Generated {total_generated} synthetic training samples.")
    print(f"Final training set size: {final_count} images.")

if __name__ == '__main__':
    train_split_dir = os.path.abspath('Day-38/yolo_dataset/train')
    augment_training_set(train_split_dir, target_count=680)
