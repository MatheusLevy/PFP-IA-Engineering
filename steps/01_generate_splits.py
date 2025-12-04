import pandas as pd
import os
import shutil

def create_directory_structure(output_dir, include_drift_accumulative=True):
    dirs_to_create = [
        os.path.join(output_dir, 'train', 'images'),
        os.path. join(output_dir, 'train', 'labels'),
        os.path. join(output_dir, 'valid', 'images'),
        os.path. join(output_dir, 'valid', 'labels'),
        os.path. join(output_dir, 'test_freeze', 'images'),
        os. path.join(output_dir, 'test_freeze', 'labels')
    ]
    
    if include_drift_accumulative:
        dirs_to_create. extend([
            os. path.join(output_dir, 'test_accumulative', 'images'),
            os.path. join(output_dir, 'test_accumulative', 'labels'),
            os.path.join(output_dir, 'test_drift', 'images'),
            os.path.join(output_dir, 'test_drift', 'labels')
        ])
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

def create_yaml_files(output_dir):
    yaml_configs = {
        'data_freeze. yaml': {
            'test_dir': './test_freeze/images'
        },
        'data_drift.yaml': {
            'test_dir': './test_drift/images'
        },
        'data_accumulative.yaml': {
            'test_dir': './test_accumulative/images'
        }
    }
    
    base_content = """train: ./train/images
val: ./valid/images
test: {test_dir}

nc: 3
names: ['Paper', 'Rock', 'Scissors']

roboflow:
  workspace: roboflow-58fyf
  project: rock-paper-scissors-sxsw
  version: 14
  license: Private
  url: https://universe.roboflow. com/roboflow-58fyf/rock-paper-scissors-sxsw/dataset/14"""
    
    for filename, config in yaml_configs.items():
        yaml_path = os. path.join(output_dir, filename)
        content = base_content. format(test_dir=config['test_dir'])
        
        with open(yaml_path, 'w') as f:
            f.write(content)
        
        print(f"Created: {yaml_path}")

def copy_files(image_list, source_images_dir, source_labels_dir, dest_images_dir, dest_labels_dir):
    for image_name in image_list:
        source_image_path = os.path.join(source_images_dir, image_name)
        dest_image_path = os.path. join(dest_images_dir, image_name)
        if os.path.exists(source_image_path):
            shutil.copy2(source_image_path, dest_image_path)
        
        label_name = os.path.splitext(image_name)[0] + '. txt'
        source_label_path = os.path.join(source_labels_dir, label_name)
        dest_label_path = os.path.join(dest_labels_dir, label_name)
        
        if os.path.exists(source_label_path):
            shutil.copy2(source_label_path, dest_label_path)

def load_existing_csv(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def recreate_splits_from_csv(csv_path, source_data_dir, output_dir, iteration=None):
    df = pd.read_csv(csv_path)
    
    if iteration is not None:
        df = df[df['iteration'] == iteration]
        print(f"Recreating splits only from iteration {iteration}")
    else:
        print("Recreating splits from all iterations")
    
    create_directory_structure(output_dir, include_drift_accumulative=True)
    
    source_images_dir = os. path.join(source_data_dir, 'images')
    source_labels_dir = os.path.join(source_data_dir, 'labels')
    
    splits_dict = {
        'train': set(),
        'valid': set(),
        'test_freeze': set(),
        'test_accumulative': set(),
        'test_drift': set(),
        'test_new': set()
    }
    
    for _, row in df.iterrows():
        image_name = row['image_name']
        splits = row['splits'].split(',') if pd. notna(row['splits']) else []
        
        for split in splits:
            split = split.strip()
            if split in splits_dict:
                splits_dict[split].add(image_name)
    
    for split_name, images in splits_dict.items():
        if images:
            dest_images_dir = os.path.join(output_dir, split_name, 'images')
            dest_labels_dir = os.path.join(output_dir, split_name, 'labels')
            
            print(f"Recreating {split_name}/: {len(images)} images")
            copy_files(list(images), source_images_dir, source_labels_dir, 
                      dest_images_dir, dest_labels_dir)
    
    create_yaml_files(output_dir)
    
    print(f"\nSplits recreated successfully in: {output_dir}")
    
    print("\n=== RECREATED SPLITS STATISTICS ===")
    for split_name, images in splits_dict.items():
        if images:
            print(f"{split_name}: {len(images)} images")

def recreate_specific_split(csv_path, source_data_dir, output_dir, split_name, iteration=None):
    df = pd.read_csv(csv_path)
    
    if iteration is not None:
        df = df[df['iteration'] == iteration]
    
    split_images = set()
    for _, row in df.iterrows():
        splits = row['splits'].split(',') if pd.notna(row['splits']) else []
        if split_name in [s.strip() for s in splits]:
            split_images. add(row['image_name'])
    
    if not split_images:
        print(f"No images found for split '{split_name}'")
        return
    
    dest_images_dir = os. path.join(output_dir, split_name, 'images')
    dest_labels_dir = os.path.join(output_dir, split_name, 'labels')
    os.makedirs(dest_images_dir, exist_ok=True)
    os. makedirs(dest_labels_dir, exist_ok=True)
    
    source_images_dir = os.path.join(source_data_dir, 'images')
    source_labels_dir = os.path.join(source_data_dir, 'labels')
    
    print(f"Recreating {split_name}/: {len(split_images)} images")
    copy_files(list(split_images), source_images_dir, source_labels_dir,
              dest_images_dir, dest_labels_dir)
    
    print(f"Split '{split_name}' recreated successfully!")

def main():
    script_dir = os. path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    csv_path = os. path.join(project_root, 'dataset_info.csv')
    source_data_dir = os. path.join(project_root, 'data')
    output_dir = os. path.join(project_root, 'dataset')
    
    recreate_splits_from_csv(csv_path, source_data_dir, output_dir, iteration=None)

if __name__ == "__main__":
    main()
