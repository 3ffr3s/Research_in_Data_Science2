import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def make_dataset(directory, class_to_idx, extensions='.pkl'):
    instances = []
    directory = os.path.expanduser(directory)
    def is_valid_file(x):
        return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances

class PklsFolder(Dataset):
    def __init__(self, root_dir):
        classes, class_to_idx = self._find_classes(root_dir)
        samples = make_dataset(root_dir, class_to_idx)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort(key = lambda x : int(x))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = pickle.load(f)
        
        #flow = np.zeros([128, 1500])
        flow = np.zeros([80,350])
        #for i in range(len(sample)):
        #for i in range(min(128,len(sample))):
        for i in range(min(80,len(sample))):
            #flow[i, :len(sample[i])] = np.frombuffer(sample[i][:1500], dtype=np.uint8)
            flow[i, :len(sample[i])] = np.frombuffer(sample[i][:350], dtype=np.uint8)
            #randomize MAC,IP addresses
            flow[i, 0:12] = np.random.randint(256, size = 12, dtype = np.uint8)
            flow[i, 26:34] = np.random.randint(256, size = 8, dtype = np.uint8)
        
        if target > 0:
            target = 1
      
        return flow, target

    def __len__(self):
        return len(self.samples)
    
    def data_cnt_per_class(self):
        class_cnt = {label : 0 for label in self.classes}
        for i in range(len(self.targets)):
            class_cnt[str(self.targets[i])] += 1
        return class_cnt

def make_cls_idx(labels,classes,null_list_path = None):
    
    null_list = []
    
    if null_list_path:
        null_list = np.load(null_list_path)
    class_idx= {x : [] for x in classes}
    for idx, cls in enumerate(labels):
        if idx not in null_list:
            class_idx[cls].append(idx)
    return class_idx

def split_train_val_test(class_idx_list,train_size,val_size,test_size):
    np.random.shuffle(class_idx_list)
    train_idx = class_idx_list[:train_size]
    val_idx = class_idx_list[train_size:train_size+val_size]
    test_idx = class_idx_list[train_size+val_size:train_size+val_size+test_size]  
    return train_idx,val_idx,test_idx