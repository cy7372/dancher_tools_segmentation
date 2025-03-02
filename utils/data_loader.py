# dancher_tools_segmentation/utils/data_loader.py

import os
import sys
import hashlib
import importlib
import traceback
from concurrent.futures import ProcessPoolExecutor
from threading import Thread

import cv2
import h5py
import numpy as np
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import torch
from dancher_tools_segmentation.utils.hdf5_dataset import HDF5Dataset

# ---------------------------------------------------------
# 1. 数据集注册表
# ---------------------------------------------------------
class DatasetRegistry:
    """
    数据集注册表，用于管理和动态加载不同的数据集。
    """
    _registry = {}

    @classmethod
    def register_dataset(cls, name):
        """
        注册数据集类。

        :param name: 数据集名称
        """
        def wrapper(dataset_class):
            cls._registry[name] = dataset_class
            return dataset_class
        return wrapper

    @classmethod
    def get_dataset(cls, name):
        """
        获取注册的数据集类。

        :param name: 数据集名称
        :return: 注册的数据集类
        """
        dataset_class = cls._registry.get(name)
        if not dataset_class:
            raise ValueError(f"Dataset '{name}' not registered.")
        return dataset_class

    @classmethod
    def load_dataset_module(cls, dataset_name):
        """
        动态导入指定名称的数据集模块。

        :param dataset_name: 数据集名称 (对应 Python 模块名)
        """
        try:
            importlib.import_module(f'datapacks.{dataset_name}')
            print(f"Successfully loaded dataset module: {dataset_name}")
        except ImportError as e:
            # 若需要查看更详细错误，可解注
            # traceback.print_exc()
            raise ValueError(
                f"Dataset type '{dataset_name}' is not recognized. "
                f"Original error: {str(e)}"
            )

# ---------------------------------------------------------
# 2. 通用工具函数
# ---------------------------------------------------------
def calculate_md5(file_path):
    """
    计算文件的 MD5 值。

    :param file_path: 文件路径
    :return: 文件 MD5 字符串
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def is_cache_valid(cache_path, module_paths):
    if not os.path.exists(cache_path):
        # print(f"[DEBUG] Cache file not found: {cache_path}")
        return False

    try:
        with h5py.File(cache_path, 'r') as f:
            cached_md5s = f.attrs.get('md5', None)
            if cached_md5s is None:
                print(f"[DEBUG] No MD5 found in cache: {cache_path}")
                return False
    except Exception as e:
        print(f"[ERROR] Failed to read cache file {cache_path}: {e}")
        return False

    current_md5s = [calculate_md5(path) for path in module_paths]
    cached_md5s = cached_md5s.split(',')

    # print(f"[DEBUG] Current MD5s: {current_md5s}")
    # print(f"[DEBUG] Cached MD5s: {cached_md5s}")

    if len(current_md5s) != len(cached_md5s):
        # print(f"[DEBUG] MD5 lengths do not match: {len(current_md5s)} != {len(cached_md5s)}")
        return False

    if not all(c1 == c2 for c1, c2 in zip(current_md5s, cached_md5s)):
        # print(f"[DEBUG] MD5 values do not match.")
        return False

    # print(f"[INFO] Cache is valid: {cache_path}")
    return True


def process_file(args):
    """
    修改后的 process_file：将 process_data 的生成器输出以分块方式返回，
    每块包含 chunk_size 个裁剪结果，以减少一次性内存占用。
    
    :param args: (dataset_name, filename, images_dir, masks_dir, img_size)
    :return: 一个包含多个元组的列表，每个元组为 (images_list, masks_list)
    """
    dataset_name, filename, images_dir, masks_dir, img_size = args

    # 获取数据集类
    dataset_class = DatasetRegistry.get_dataset(dataset_name)

    chunk_size = 100  # 根据实际内存情况调整块大小
    imgs = []
    masks = []
    chunks = []

    # 使用生成器逐个获取裁剪块
    for cropped_image, processed_mask in dataset_class.process_data(filename, images_dir, masks_dir, img_size):
        imgs.append(cropped_image)
        masks.append(processed_mask)
        if len(imgs) >= chunk_size:
            chunks.append((imgs, masks))
            imgs = []
            masks = []
    if imgs:
        chunks.append((imgs, masks))
    return chunks

# ---------------------------------------------------------
# 3. 缓存与数据构建逻辑
# ---------------------------------------------------------
def create_cache(datapack, path, img_size, cache_path, module_paths):
    """
    修改后的 create_cache：逐个处理图像文件，处理后直接写入 HDF5 缓存，
    避免一次性采集所有子进程返回的数据造成内存峰值，从而降低被 OOM killer 终止的风险。
    
    :param datapack: 数据集类
    :param path: 数据目录根路径
    :param img_size: 裁剪尺寸
    :param cache_path: 缓存文件保存路径
    :param module_paths: 用于生成 MD5 校验的模块文件路径列表
    """
    import gc
    images_dir = os.path.join(path, 'images')
    masks_dir = os.path.join(path, 'masks')
    images_files = sorted(os.listdir(images_dir))
    
    # 获取数据集名称，用于 process_file 调用
    dataset_name = datapack.dataset_name

    with h5py.File(cache_path, 'w') as f:
        dset_images = None
        dset_masks = None
        current_index = 0

        # 顺序处理每个图像文件，避免一次收集所有进程返回数据
        for filename in tqdm(images_files, desc=f"Processing images in {path}"):
            # 直接调用 process_file 处理单个图像文件
            file_chunks = process_file((dataset_name, filename, images_dir, masks_dir, img_size))
            # file_chunks 是列表，每个元素为 (images_list, masks_list)
            for images_list, masks_list in file_chunks:
                images_arr = np.array(images_list, dtype=np.float32)  # (n, H, W, C)
                masks_arr = np.array(masks_list, dtype=np.uint8)       # (n, H, W)
                if dset_images is None:
                    dset_images = f.create_dataset(
                        'images',
                        data=images_arr,
                        maxshape=(None,) + images_arr.shape[1:],
                        chunks=True,
                        dtype='float32'
                    )
                    dset_masks = f.create_dataset(
                        'masks',
                        data=masks_arr,
                        maxshape=(None,) + masks_arr.shape[1:],
                        chunks=True,
                        dtype='uint8'
                    )
                    current_index = images_arr.shape[0]
                else:
                    new_size = current_index + images_arr.shape[0]
                    dset_images.resize((new_size,) + images_arr.shape[1:])
                    dset_masks.resize((new_size,) + masks_arr.shape[1:])
                    dset_images[current_index:new_size] = images_arr
                    dset_masks[current_index:new_size] = masks_arr
                    current_index = new_size

                # 尽快释放中间变量
                del images_arr, masks_arr, images_list, masks_list
            # 每处理完一个文件调用垃圾回收
            gc.collect()

        # 写入当前用于 MD5 校验的文件属性
        module_md5 = ','.join([calculate_md5(p) for p in module_paths])
        f.attrs['md5'] = module_md5
        f.flush()

    # 这里不加载全部数据到内存，后续使用 HDF5Dataset 的懒加载方式
    return


# ---------------------------------------------------------
# 4. 核心数据加载接口
# ---------------------------------------------------------
def get_dataloaders(args):
    """
    通用数据加载器生成函数。可以加载指定数据集，并在首次加载时自动生成缓存。

    :param args: 包含训练配置和数据集配置的参数对象
    :return: (train_loader, test_loader)
    """
    batch_size = args.batch_size
    num_workers = args.num_workers
    img_size = args.img_size

    # 只加载配置中的单个数据集
    dataset_config = args.ds
    dataset_name = dataset_config['name']

    # 动态加载数据集模块 & 获取数据集类
    DatasetRegistry.load_dataset_module(dataset_name)
    datapack = DatasetRegistry.get_dataset(dataset_name)

    # 生成 module_paths 供 MD5 校验（包含数据集模块和 data_loader.py 本身）
    data_loader_path = os.path.abspath(__file__)
    dataset_module_path = os.path.join('datapacks', f'{dataset_name}.py')
    module_paths = [
        dataset_module_path,
        data_loader_path
    ]

    train_datasets, test_datasets = [], []

    # -- 1) 加载训练数据
    for train_path in dataset_config.get('train_paths', []):
        train_cache_path = os.path.join(train_path, "__cache__.h5")
        if not is_cache_valid(train_cache_path, module_paths):
            create_cache(datapack, train_path, img_size,
                         train_cache_path, module_paths)
        # 使用 HDF5Dataset, 避免一次性将全部数据加载内存
        train_dataset = HDF5Dataset(train_cache_path)
        train_datasets.append(train_dataset)

    # -- 2) 加载测试数据
    for test_path in dataset_config.get('test_paths', []):
        test_cache_path = os.path.join(test_path, "__cache__.h5")
        if not is_cache_valid(test_cache_path, module_paths):
            create_cache(datapack, test_path, img_size,
                         test_cache_path, module_paths)
        test_dataset = HDF5Dataset(test_cache_path)
        test_datasets.append(test_dataset)

    # 组装 ConcatDataset
    combined_train_dataset = ConcatDataset(train_datasets) if train_datasets else None
    combined_test_dataset = ConcatDataset(test_datasets) if test_datasets else None

    # 构建 Dataloader
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    ) if combined_train_dataset else None

    test_loader = DataLoader(
        combined_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if combined_test_dataset else None

    return train_loader, test_loader
