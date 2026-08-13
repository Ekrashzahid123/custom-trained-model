import os
import glob
import shutil

def setup_unseen_images():
    base_dir = os.path.abspath('Day-38')
    test_src_dir = os.path.join(base_dir, 'yolo_dataset', 'test', 'images')
    unseen_dst_dir = os.path.join(base_dir, 'unseen_test_images')
    os.makedirs(unseen_dst_dir, exist_ok=True)
    
    test_imgs = sorted(glob.glob(os.path.join(test_src_dir, '*.*')))
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    test_imgs = [p for p in test_imgs if p.lower().endswith(valid_exts)]
    
    # Pick 12 test images to serve as unseen evaluation test set
    selected = test_imgs[:12]
    for idx, img_path in enumerate(selected, 1):
        filename = f"unseen_{idx:02d}_{os.path.basename(img_path)}"
        dst_path = os.path.join(unseen_dst_dir, filename)
        shutil.copy2(img_path, dst_path)
        
    print(f"Copied {len(selected)} unseen test images into Day-38/unseen_test_images/")

if __name__ == '__main__':
    setup_unseen_images()
