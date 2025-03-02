import h5py
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, cache_path):
        self.cache_path = cache_path
        # 延迟加载文件句柄，避免在多进程情况下共享同一句柄
        self.file = None
        self.images = None
        self.masks = None

    def _init_file(self):
        if self.file is None:
            self.file = h5py.File(self.cache_path, 'r')
            self.images = self.file['images']
            self.masks = self.file['masks']

    def __len__(self):
        self._init_file()
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        self._init_file()
        image = self.images[idx]
        mask = self.masks[idx]
        # 转换为 PyTorch Tensor，注意保证 image 形状为 (H,W,C)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return image, mask
    
    def __del__(self):
        if self.file is not None:
            self.file.close() 