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
    处理单个文件并返回裁剪块或单图。
    """
    dataset_name, filename, images_dir, masks_dir, img_size = args

    # 获取数据集类
    dataset_class = DatasetRegistry.get_dataset(dataset_name)

    # 调用数据集类的方法
    try:
        result = dataset_class.process_data(filename, images_dir, masks_dir, img_size)
    except Exception as e:
        print(f"[ERROR] Processing failed for file: {filename} with error: {e}")
        raise e

    if isinstance(result[0], list):  # 检查是否是裁剪结果
        return result
    else:  # 单图包装为列表
        return [result[0]], [result[1]]

# ---------------------------------------------------------
# 3. 缓存与数据构建逻辑
# ---------------------------------------------------------
def create_cache(datapack, path, img_size, cache_path, module_paths):
    """
    根据指定的目录读取所有图像与掩码，处理后保存到缓存文件（异步存储）。
    兼容普通数据集和裁剪型数据集，自动适配裁剪逻辑。

    :param datapack: 该数据集的类（内含 color_map 等）
    :param path: 数据目录路径
    :param img_size: 输出图像大小
    :param cache_path: 缓存文件存放路径
    :param module_paths: list, 用于生成数据的模块文件路径
    :return: dict，包括 'images' 和 'masks'
    """
    images_dir = os.path.join(path, 'images')
    masks_dir = os.path.join(path, 'masks')
    images_files = sorted(os.listdir(images_dir))

    if hasattr(datapack, 'color_map'):
        color_map = getattr(datapack, 'color_map')
    else:
        raise ValueError(
            f"The dataset {datapack.__name__} must define a 'color_map' attribute."
        )

    if not isinstance(color_map, dict):
        raise ValueError("The dataset's color_map must be a dictionary.")

    # 初始化用于存储的图像和掩码列表
    all_images = []
    all_masks = []

    # 构建并行处理所需参数
    args_list = [
        (datapack.dataset_name, filename, images_dir, masks_dir, img_size)
        for filename in images_files
    ]

    # 并行处理图像与掩码
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_file, args_list),
            desc=f"Processing images in {path}",
            total=len(images_files)
        ))

    # 遍历结果并存储所有图像和掩码
    for i, result in enumerate(results):
        if isinstance(result[0], list) and isinstance(result[1], list):
            # 如果返回的是裁剪块列表，逐块添加
            all_images.extend(result[0])  # 裁剪后的多块图像
            all_masks.extend(result[1])  # 裁剪后的多块掩码
        else:
            # 普通数据集返回单个图像和掩码
            all_images.append(result[0])  # 单图像
            all_masks.append(result[1])  # 单掩码

    # 将所有图像和掩码转换为 NumPy 数组
    all_images = np.array(all_images, dtype=np.float32)  # (N, H, W, C)
    all_masks = np.array(all_masks, dtype=np.uint8)      # (N, H, W)

    # 打印最终的形状调试信息
    # print(f"[INFO] Total processed images: {all_images.shape}")
    # print(f"[INFO] Total processed masks: {all_masks.shape}")

    # 计算 MD5 并保存缓存
    module_md5 = ','.join([calculate_md5(p) for p in module_paths])

    def save():
        with h5py.File(cache_path, 'w') as f:
            f.create_dataset('images', data=all_images, dtype='float32')
            f.create_dataset('masks', data=all_masks, dtype='uint8')
            f.attrs['md5'] = module_md5

    Thread(target=save).start()

    return {'images': all_images, 'masks': all_masks}


# ---------------------------------------------------------
# 4. 核心数据加载接口
# ---------------------------------------------------------
def get_dataloaders(args):
    """
    通用数据加载器生成函数。可以加载指定数据集，并在首次加载时自动生成缓存。

    :param args: 包含训练配置和数据集配置的参数对象
                 - args.ds: 单个数据集的配置(dict)，包含 'name', 'train_paths', 'test_paths' 等
                 - args.batch_size, args.num_workers, args.img_size 等
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

    # 生成 module_paths 供 MD5 校验
    # 包含数据集模块和 data_loader.py 本身
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
        if is_cache_valid(train_cache_path, module_paths):
            with h5py.File(train_cache_path, 'r') as f:
                images = f['images'][:]
                masks = f['masks'][:]
                train_data = {'images': images, 'masks': masks}
        else:
            train_data = create_cache(
                datapack, train_path, img_size,
                train_cache_path, module_paths
            )
        train_dataset = datapack(train_data)  # IWDataset 实例化
        train_datasets.append(train_dataset)

    # -- 2) 加载测试数据
    for test_path in dataset_config.get('test_paths', []):
        test_cache_path = os.path.join(test_path, "__cache__.h5")
        if is_cache_valid(test_cache_path, module_paths):
            # print(f"[DEBUG] Cache valid for testing path: {test_path}")
            with h5py.File(test_cache_path, 'r') as f:
                images = f['images'][:]
                masks = f['masks'][:]
                test_data = {'images': images, 'masks': masks}
        else:
            test_data = create_cache(
                datapack, test_path, img_size,
                test_cache_path, module_paths
            )
        test_dataset = datapack(test_data)  # IWDataset 实例化
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
