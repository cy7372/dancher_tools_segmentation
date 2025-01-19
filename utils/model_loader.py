import importlib
import sys
import os
import torch
from dancher_tools_segmentation.core import SegModel

def get_model(args, device):
    model_name = args.model_name  # 直接使用传入的模型名称

    # 从 args 中读取所有参数，并准备一个 task_params 字典
    task_params = vars(args)  # 获取 args 中所有参数

    # 必需参数检查
    required_parameters = ['num_classes', 'img_size', 'in_channels', 'model_save_dir']
    for param in required_parameters:
        if param not in task_params or task_params[param] is None:
            raise ValueError(f"Missing required parameter '{param}'")

    # 尝试导入预设模型
    try:
        models_module = importlib.import_module('dancher_tools_segmentation.models')
        preset_models = getattr(models_module, 'PRESET_MODELS', {})
    except ImportError as e:
        raise ImportError(f"Failed to import preset models for task 'segmentation': {e}")

    # 加载并实例化模型
    try:
        if model_name in preset_models:
            model_class = preset_models[model_name]
            print(f"Loaded model '{model_name}' from presets.")
        else:
            # 加载自定义模型模块
            models_path = os.path.join(os.path.dirname(__file__), '../models')
            if models_path not in sys.path:
                sys.path.append(models_path)
            custom_model_module = f"models.{model_name}"
            model_module = importlib.import_module(custom_model_module)
            model_class = getattr(model_module, model_name)
            print(f"Loaded custom model '{model_name}' from 'models/{model_name}.py'.")

        # 获取模型构造函数的参数
        init_params = model_class.__init__.__code__.co_varnames

        # 过滤并传入模型所需的参数
        model_params = {k: v for k, v in task_params.items() if k in init_params}

        # 如果模型不是 SegModel，移除 `model_save_dir`
        if not issubclass(model_class, SegModel):
            model_params.pop('model_save_dir', None)

        # 实例化模型并传递给设备（CPU/GPU）
        model_instance = model_class(**model_params).to(device)

    except Exception as e:
        raise ValueError(f"Error instantiating model '{model_name}': {e}")

    # 定义封装的 WrappedModel
    class WrappedModel(SegModel):
        def __init__(self, **kwargs):
            # 从 kwargs 中提取 SegModel 的必需参数
            segmodel_kwargs = {param: kwargs[param] for param in ['num_classes', 'img_size', 'in_channels', 'model_save_dir'] if param in kwargs}
            # 调用 SegModel 的构造函数
            super(WrappedModel, self).__init__(**segmodel_kwargs)
            self.model_instance = model_instance
            self.model_name = model_name

        def forward(self, *args, **kwargs):
            return self.model_instance(*args, **kwargs)

    # 返回封装后的模型
    return WrappedModel(**task_params).to(device)


def load_weights(model, args):
    """
    根据配置加载或迁移模型权重。
    :param model: 初始化后的模型实例
    :param args: 配置对象
    """
    if args.transfer_weights:  # 如果提供了迁移权重路径
        print(f"Transferring model weights from {args.transfer_weights}")
        model.transfer(specified_path=args.transfer_weights, strict=False)
    else:
        model.load(model_dir=args.model_save_dir, mode=args.load_mode, specified_path=args.weights)
