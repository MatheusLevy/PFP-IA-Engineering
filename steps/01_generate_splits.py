import pandas as pd
import os
import shutil
from datetime import datetime
from collections import defaultdict
import logging

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def create_directory_structure(output_dir, include_drift_accumulative=True):
    dirs_to_create = [
        os.path.join(output_dir, 'train', 'images'),
        os.path.join(output_dir, 'train', 'labels'),
        os.path.join(output_dir, 'valid', 'images'),
        os.path.join(output_dir, 'valid', 'labels'),
        os.path.join(output_dir, 'test_freeze', 'images'),
        os.path.join(output_dir, 'test_freeze', 'labels')
    ]
    
    if include_drift_accumulative:
        dirs_to_create.extend([
            os.path.join(output_dir, 'test_accumulative', 'images'),
            os.path.join(output_dir, 'test_accumulative', 'labels'),
            os.path.join(output_dir, 'test_drift', 'images'),
            os.path.join(output_dir, 'test_drift', 'labels')
        ])
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

def create_yaml_files(output_dir):
    yaml_configs = {
        'data_freeze.yaml': {
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
  url: https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw/dataset/14"""
    
    for filename, config in yaml_configs.items():
        yaml_path = os.path.join(output_dir, filename)
        content = base_content.format(test_dir=config['test_dir'])
        
        with open(yaml_path, 'w') as f:
            f.write(content)

def copy_files(image_list, source_images_dir, source_labels_dir, dest_images_dir, dest_labels_dir):
    copied_count = 0
    for image_name in image_list:
        source_image_path = os.path.join(source_images_dir, image_name)
        dest_image_path = os.path.join(dest_images_dir, image_name)
        
        if os.path.exists(source_image_path):
            shutil.copy2(source_image_path, dest_image_path)
            copied_count += 1
        
        label_name = os.path.splitext(image_name)[0] + '.txt'
        source_label_path = os.path.join(source_labels_dir, label_name)
        dest_label_path = os.path.join(dest_labels_dir, label_name)
        
        if os.path.exists(source_label_path):
            shutil.copy2(source_label_path, dest_label_path)
    
    return copied_count

def get_existing_images_in_splits(output_dir):
    splits = ['train', 'valid', 'test_freeze', 'test_accumulative', 'test_drift', 'test_new']
    existing_images = defaultdict(set)
    
    for split_name in splits:
        images_dir = os.path.join(output_dir, split_name, 'images')
        if os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    existing_images[split_name].add(filename)
    
    return existing_images

def get_all_existing_images(existing_images_dict):
    all_images = set()
    for images_set in existing_images_dict.values():
        all_images.update(images_set)
    return all_images

def is_first_run(output_dir):
    required_dirs = ['train/images', 'valid/images', 'test_freeze/images']
    
    for dir_path in required_dirs:
        full_path = os.path.join(output_dir, dir_path)
        if os.path.exists(full_path) and len(os.listdir(full_path)) > 0:
            return False
    return True

def load_existing_csv(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def save_processing_log(output_dir, stats, is_initial_run=False):
    log_file = os.path.join(output_dir, 'processing_log.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    mode = 'a' if os.path.exists(log_file) else 'w'
    
    with open(log_file, mode) as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Processamento: {timestamp}\n")
        f.write(f"Tipo: {'Criação Inicial' if is_initial_run else 'Adição Incremental'}\n")
        f.write("Estatísticas:\n")
        for split_name, count in stats.items():
            if count > 0:
                f.write(f"  {split_name}: {count} imagens\n")

def print_final_split_summary(output_dir):
    existing_images = get_existing_images_in_splits(output_dir)
    print("\n" + "="*60)
    print("RESUMO FINAL DOS SPLITS")
    print("="*60)

    total_images = 0
    for split_name, images_set in existing_images.items():
        count = len(images_set)
        if count > 0:
            print(f"  {split_name:<20}: {count:>6} imagens")
            total_images += count

    print("-"*60)
    print(f"  {'TOTAL':<20}: {total_images:>6} imagens")
    print("="*60)

    return existing_images

def save_final_summary_log(output_dir, final_stats):
    log_file = os.path.join(output_dir, 'processing_log.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"RESUMO FINAL: {timestamp}\n")
        f.write("Estado final dos splits:\n")
        
        total_images = 0
        for split_name, images_set in final_stats.items():
            count = len(images_set)
            if count > 0:
                f.write(f"  {split_name}: {count} imagens\n")
                total_images += count
        
        f.write(f"  TOTAL: {total_images} imagens\n")
        f.write(f"{'='*50}\n")

def recreate_splits_from_csv(csv_path, source_data_dir, output_dir, iteration=None):
    if not os.path.exists(csv_path):
        logging.error(f"Arquivo CSV não encontrado: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if iteration is not None:
        df = df[df['iteration'] == iteration]
        logging.info(f"Processando apenas iteração {iteration}")

    first_run = is_first_run(output_dir)

    if first_run:
        print("PRIMEIRA EXECUÇÃO - Criando splits iniciais")
        _create_initial_splits(df, source_data_dir, output_dir)
    else:
        print("EXECUÇÃO INCREMENTAL - Adicionando novos dados")
        _add_incremental_data(df, source_data_dir, output_dir)

def _create_initial_splits(df, source_data_dir, output_dir):
    print("Criando estrutura de diretórios...")
    create_directory_structure(output_dir, include_drift_accumulative=True)
    
    source_images_dir = os.path.join(source_data_dir, 'images')
    source_labels_dir = os.path.join(source_data_dir, 'labels')
    
    splits_dict = defaultdict(set)
    
    for _, row in df.iterrows():
        image_name = row['image_name']
        splits = row['splits'].split(',') if pd.notna(row['splits']) else []
        
        for split in splits:
            split = split.strip()
            if split:
                splits_dict[split].add(image_name)
    
    total_stats = {}
    for split_name, images in splits_dict.items():
        if images:
            dest_images_dir = os.path.join(output_dir, split_name, 'images')
            dest_labels_dir = os.path.join(output_dir, split_name, 'labels')
            
            copied_count = copy_files(list(images), source_images_dir, source_labels_dir, 
                                    dest_images_dir, dest_labels_dir)
            
            total_stats[split_name] = copied_count
            print(f"  {split_name}: {copied_count} imagens copiadas")
    
    create_yaml_files(output_dir)
    save_processing_log(output_dir, total_stats, is_initial_run=True)
    
    print(f"Splits iniciais criados com sucesso em: {output_dir}")

def _add_incremental_data(df, source_data_dir, output_dir):
    existing_images = get_existing_images_in_splits(output_dir)
    all_existing = get_all_existing_images(existing_images)
    
    print(f"Imagens já existentes nos splits: {len(all_existing)}")
    
    source_images_dir = os.path.join(source_data_dir, 'images')
    source_labels_dir = os.path.join(source_data_dir, 'labels')
    
    splits_dict = defaultdict(set)
    new_images_count = 0
    
    for _, row in df.iterrows():
        image_name = row['image_name']
        
        if image_name not in all_existing:
            new_images_count += 1
            splits = row['splits'].split(',') if pd.notna(row['splits']) else []
            
            for split in splits:
                split = split.strip()
                if split:
                    splits_dict[split].add(image_name)
    
    if new_images_count == 0:
        print("Nenhuma imagem nova encontrada. Splits já estão atualizados.")
        return
    
    print(f"Encontradas {new_images_count} imagens novas para adicionar")
    
    create_directory_structure(output_dir, include_drift_accumulative=True)
    
    incremental_stats = {}
    for split_name, images in splits_dict.items():
        if images:
            dest_images_dir = os.path.join(output_dir, split_name, 'images')
            dest_labels_dir = os.path.join(output_dir, split_name, 'labels')
            
            copied_count = copy_files(list(images), source_images_dir, source_labels_dir, 
                                    dest_images_dir, dest_labels_dir)
            
            current_total = len(existing_images[split_name]) + copied_count
            incremental_stats[split_name] = copied_count
            
            print(f"  {split_name}: +{copied_count} imagens (total: {current_total})")
    
    create_yaml_files(output_dir)
    save_processing_log(output_dir, incremental_stats, is_initial_run=False)
    
    print("Dados incrementais adicionados com sucesso!")

def recreate_specific_split(csv_path, source_data_dir, output_dir, split_name, iteration=None):
    df = pd.read_csv(csv_path)
    
    if iteration is not None:
        df = df[df['iteration'] == iteration]
    
    split_images = set()
    for _, row in df.iterrows():
        splits = row['splits'].split(',') if pd.notna(row['splits']) else []
        if split_name in [s.strip() for s in splits]:
            split_images.add(row['image_name'])
    
    if not split_images:
        logging.error(f"Nenhuma imagem encontrada para o split '{split_name}'")
        return
    
    dest_images_dir = os.path.join(output_dir, split_name, 'images')
    dest_labels_dir = os.path.join(output_dir, split_name, 'labels')
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)
    
    source_images_dir = os.path.join(source_data_dir, 'images')
    source_labels_dir = os.path.join(source_data_dir, 'labels')
    
    for filename in os.listdir(dest_images_dir):
        os.remove(os.path.join(dest_images_dir, filename))
    for filename in os.listdir(dest_labels_dir):
        os.remove(os.path.join(dest_labels_dir, filename))
    
    copied_count = copy_files(list(split_images), source_images_dir, source_labels_dir,
                             dest_images_dir, dest_labels_dir)
    
    logging.info(f"Split '{split_name}' recriado: {copied_count} imagens")

def get_splits_summary(output_dir):
    existing_images = get_existing_images_in_splits(output_dir)
    
    logging.info("RESUMO DOS SPLITS ATUAIS:")
    total_images = 0
    for split_name, images_set in existing_images.items():
        count = len(images_set)
        if count > 0:
            logging.info(f"{split_name:20}: {count:6d} imagens")
            total_images += count
    logging.info(f"TOTAL: {total_images} imagens")
    
    return existing_images

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    csv_path = os.path.join(project_root, 'dataset_info.csv')
    source_data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'dataset')

    print("INICIANDO PROCESSAMENTO DE SPLITS")
    print(f"CSV: {csv_path}")
    print(f"Dados fonte: {source_data_dir}")
    print(f"Saída: {output_dir}")

    if not os.path.exists(csv_path):
        print(f"Arquivo CSV não encontrado: {csv_path}")
        return

    if not os.path.exists(source_data_dir):
        print(f"Diretório de dados não encontrado: {source_data_dir}")
        return

    try:
        recreate_splits_from_csv(csv_path, source_data_dir, output_dir, iteration=None)

        final_stats = print_final_split_summary(output_dir)
        save_final_summary_log(output_dir, final_stats)

        print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")

    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        raise

if __name__ == "__main__":
    main()