import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path

def save_arr_to_dir(arr, dir):
    with open(dir, 'wb') as f:
        np.save(f, arr)

def get_id_embeddings(net, id_loader_dict, save_root, aggregate_layers=1):
    # save id (test & val) results
    net.eval()
    modes = ['train', 'test', 'val']
    for mode in modes:

        logits_path = os.path.join(save_root, f'{mode}_logits.npy')

        if Path(logits_path).is_file():
            print("Skipping", mode.capitalize(), "embeddings, already exists.")
            continue
        else:
            print("Extracting", mode.capitalize(), "embeddings...")

        dl = id_loader_dict[mode]
        dataiter = iter(dl)
        
        logits_list = []
        feature_list = []
        label_list = []
        
        for i in tqdm(range(1,
                        len(dataiter) + 1),
                        desc='Extracting reults...',
                        position=0,
                        leave=True):
            batch = next(dataiter)
            data = batch['data'].cuda()
            label = batch['label']
            with torch.no_grad():
                logits_cls, features_list = net(data, return_feature_list=True)
            feature = torch.concatenate([layer.mean(dim=(2, 3)) for layer in features_list[aggregate_layers:]], dim=1)
            logits_list.append(logits_cls.data.to('cpu').numpy())
            feature_list.append(feature.data.to('cpu').numpy())
            label_list.append(label.numpy())

        logits_arr = np.concatenate(logits_list)
        feature_arr = np.concatenate(feature_list)
        label_arr = np.concatenate(label_list)

        print(f"Base {mode.capitalize()} Accuracy:", (logits_arr.argmax(1) == (label_arr)).mean())

        os.makedirs(save_root, exist_ok=True)
        save_arr_to_dir(logits_arr, os.path.join(save_root, f'{mode}_logits.npy'))
        save_arr_to_dir(feature_arr, os.path.join(save_root, f'{mode}_feature.npy'))
        save_arr_to_dir(label_arr, os.path.join(save_root, f'{mode}_labels.npy'))

    # Load the embeddings
    data = dict()
    data['id'] = dict()
    for split in ['train', 'val', 'test']:
        data['id'][split] = dict()
        data['id'][split]['feature'] = np.load(os.path.join(save_root, f'{split}_feature.npy'))
        data['id'][split]['labels'] = np.load(os.path.join(save_root, f'{split}_labels.npy'))
        data['id'][split]['logits'] = np.load(os.path.join(save_root, f'{split}_logits.npy'))

    return data

def get_ood_embeddings(net, ood_loader_dict, save_root, aggregate_layers=1):
    # save ood results
    net.eval()
    ood_splits = ['nearood', 'farood']
    for ood_split in ood_splits:
        for dataset_name, ood_dl in ood_loader_dict[ood_split].items():
            dataiter = iter(ood_dl)

            logits_path = os.path.join(save_root, ood_split, f'{dataset_name}_logits.npy')

            if Path(logits_path).is_file():
                print("Skipping", ood_split.capitalize(), f"{dataset_name}_embeddings, already exists.")
                continue
            else:
                print("Extracting", ood_split.capitalize(), f"{dataset_name}_embeddings...")
        
            logits_list = []
            feature_list = []
            label_list = []

            for i in tqdm(range(1,
                            len(dataiter) + 1),
                            desc='Extracting reults...',
                            position=0,
                            leave=True):
                batch = next(dataiter)
                data = batch['data'].cuda()
                label = batch['label']

                with torch.no_grad():
                    logits_cls, features_list = net(data, return_feature_list=True)
                feature = torch.concatenate([layer.mean(dim=(2, 3)) for layer in features_list[aggregate_layers:]], dim=1)
                logits_list.append(logits_cls.data.to('cpu').numpy())
                feature_list.append(feature.data.to('cpu').numpy())
                label_list.append(label.numpy())

            logits_arr = np.concatenate(logits_list)
            feature_arr = np.concatenate(feature_list)
            label_arr = np.concatenate(label_list)

            os.makedirs(os.path.join(save_root, ood_split), exist_ok=True)
            save_arr_to_dir(logits_arr, os.path.join(save_root, ood_split, f'{dataset_name}_logits.npy'))
            save_arr_to_dir(feature_arr, os.path.join(save_root, ood_split, f'{dataset_name}_feature.npy'))
            save_arr_to_dir(label_arr, os.path.join(save_root, ood_split, f'{dataset_name}_labels.npy'))

    # Load the embeddings
    split_types = ['nearood', 'farood']
    data = dict()
    for split_type in split_types:
        data[split_type] = dict()
        for dataset_name in ood_loader_dict[split_type]:
            data[split_type][dataset_name] = dict()
            data[split_type][dataset_name]['feature'] = np.load(os.path.join(save_root, split_type, f'{dataset_name}_feature.npy'))
            data[split_type][dataset_name]['logits'] = np.load(os.path.join(save_root, split_type, f'{dataset_name}_logits.npy'))
            data[split_type][dataset_name]['labels'] = np.load(os.path.join(save_root, split_type, f'{dataset_name}_labels.npy'))

    return data

def resize_images_to_common_size(images, target_size=None, method='pad'):
    """
    Resize all images to a common size.
    
    Args:
        images: List of images (numpy arrays)
        target_size: Tuple (height, width). If None, uses the largest dimensions.
        method: 'resize', 'pad', or 'crop'
    """
    if not images:
        return images
    
    # Determine target size if not provided
    if target_size is None:
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        target_size = (max_h, max_w)
    
    processed_images = []
    
    for img in images:
        if method == 'resize':
            # Stretch/squash to fit target size
            resized = cv2.resize(img, (target_size[1], target_size[0]))
            processed_images.append(resized)
            
        elif method == 'pad':
            # Pad with zeros (black) to reach target size
            h, w = img.shape[:2]
            target_h, target_w = target_size
            
            # Calculate padding
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            
            # Center the image
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            
            if len(img.shape) == 3:  # Color image
                padded = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='constant')
            else:  # Grayscale
                padded = np.pad(img, ((top, bottom), (left, right)), mode='constant')
            
            processed_images.append(padded)
            
        elif method == 'crop':
            # Crop from center to target size
            h, w = img.shape[:2]
            target_h, target_w = target_size
            
            # Calculate crop boundaries
            start_h = max(0, (h - target_h) // 2)
            start_w = max(0, (w - target_w) // 2)
            end_h = min(h, start_h + target_h)
            end_w = min(w, start_w + target_w)
            
            cropped = img[start_h:end_h, start_w:end_w]
            
            # If still smaller than target, pad
            if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
                if len(cropped.shape) == 3:
                    padded = np.zeros((target_h, target_w, cropped.shape[2]), dtype=cropped.dtype)
                    padded[:cropped.shape[0], :cropped.shape[1]] = cropped
                else:
                    padded = np.zeros((target_h, target_w), dtype=cropped.dtype)
                    padded[:cropped.shape[0], :cropped.shape[1]] = cropped
                processed_images.append(padded)
            else:
                processed_images.append(cropped)
    
    return processed_images