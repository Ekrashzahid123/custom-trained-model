import os
import glob
import shutil
import random

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    random.seed(seed)
    
    img_dir = os.path.join(source_dir, 'images')
    lbl_dir = os.path.join(source_dir, 'labels')
    
    image_paths = sorted(glob.glob(os.path.join(img_dir, '*.*')))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [p for p in image_paths if p.lower().endswith(valid_extensions)]
    
    print(f"Found {len(image_paths)} original images in {img_dir}")
    
    # Shuffle with fixed seed for reproducibility
    random.shuffle(image_paths)
    
    n_total = len(image_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    train_imgs = image_paths[:n_train]
    val_imgs = image_paths[n_train:n_train + n_val]
    test_imgs = image_paths[n_train + n_val:]
    
    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }
    
    for split_name, imgs in splits.items():
        split_img_dir = os.path.join(output_dir, split_name, 'images')
        split_lbl_dir = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)
        
        for img_path in imgs:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
            
            dest_img = os.path.join(split_img_dir, os.path.basename(img_path))
            dest_lbl = os.path.join(split_lbl_dir, f"{base_name}.txt")
            
            shutil.copy2(img_path, dest_img)
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, dest_lbl)
            else:
                # Create empty label file if missing
                with open(dest_lbl, 'w') as f:
                    pass
                
        print(f"Split [{split_name}]: {len(imgs)} images copied to {split_img_dir}")

if __name__ == '__main__':
    src = os.path.abspath('Cup-Phone.v1i.yolov8 (1)/train')
    out = os.path.abspath('Day-38/yolo_dataset')
    
    # Copy original raw dataset into Day-38/dataset/original for archiving
    orig_img_dir = os.path.abspath('Day-38/dataset/original/images')
    orig_lbl_dir = os.path.abspath('Day-38/dataset/original/labels')
    os.makedirs(orig_img_dir, exist_ok=True)
    os.makedirs(orig_lbl_dir, exist_ok=True)
    
    for f in glob.glob(os.path.join(src, 'images', '*.*')):
        shutil.copy2(f, orig_img_dir)
    for f in glob.glob(os.path.join(src, 'labels', '*.txt')):
        shutil.copy2(f, orig_lbl_dir)
        
    print(f"Original dataset copied to Day-38/dataset/original ({len(glob.glob(os.path.join(orig_img_dir, '*')))} images)")
    
    split_dataset(src, out, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
