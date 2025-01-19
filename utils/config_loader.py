import argparse
import yaml
import os
from .data_loader import DatasetRegistry

# 必需和可选的通用参数字典
required_common_parameters = {
    'model_name': str,
    'img_size': int,
    'num_classes': int,
}
optional_common_parameters = {
    'weights': (str, None),
    "transfer_weights": (str, None),
    'load_mode': (str, 'latest'),
    'learning_rate': (float, 0.001),
    'batch_size': (int, 16),
    'num_workers': (int, 4),
    'patience': (int, 10),
    'delta': (float, 0.01),
    'loss': (str, 'bce'),
    'loss_weights': (str, None),
    'num_epochs': (int, 100),
    'model_save_dir': (str, 'checkpoints'),
    'metrics': (str, None),
    'export': (bool, False),
    'conf_threshold': (float, None),
    'save_interval': (int, 5),
    'in_channels': (int, 3),
}

class Config:
    """支持通过属性访问配置参数的配置类"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def load_yaml_config(config_file):
    """加载并解析 YAML 配置文件"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def parse_loss_weights(loss_weights_str):
    """解析逗号分隔的损失权重字符串"""
    try:
        return [float(w) for w in loss_weights_str.split(",")]
    except ValueError:
        raise ValueError("Parameter 'loss_weights' should be a comma-separated list of numbers.")

def parse_metrics(metrics_str):
    """解析逗号分隔的指标字符串"""
    return [metric.strip() for metric in metrics_str.split(',')]

def get_color_maps(dataset_name):
    """从注册的数据集中获取 color_map 和 class_name（都可选）"""
    DatasetRegistry.load_dataset_module(dataset_name)
    dataset_class = DatasetRegistry.get_dataset(dataset_name)
    
    color_map = None
    class_name = None
    
    if hasattr(dataset_class, 'color_map'):
        color_map = dataset_class.color_map
        color_to_class = {color: class_id for color, class_id in color_map.items()}
        class_to_color = {class_id: color for color, class_id in color_map.items()}
    else:
        color_to_class = None
        class_to_color = None
    
    if hasattr(dataset_class, 'class_name'):
        class_name = dataset_class.class_name
    
    return color_to_class, class_to_color, class_name

def validate_and_set_defaults(config):
    """验证配置参数并设置默认值"""
    validated_config = {}
    errors = []

    # 检查必需的通用参数
    for param, param_type in required_common_parameters.items():
        if param in config and isinstance(config[param], param_type):
            validated_config[param] = config[param]
        else:
            errors.append(f"Missing or invalid required parameter '{param}'.")

    # 检查可选的通用参数并设置默认值
    for param, (param_type, default_value) in optional_common_parameters.items():
        if param in config:
            if param == "loss_weights" and isinstance(config[param], str):
                validated_config[param] = parse_loss_weights(config[param])
            elif param == "metrics" and isinstance(config[param], str):
                validated_config[param] = parse_metrics(config[param])
            elif isinstance(config[param], param_type):
                validated_config[param] = config[param]
            else:
                errors.append(f"Parameter '{param}' should be of type {param_type.__name__}.")
        else:
            validated_config[param] = default_value

    if 'datasets' in config:
        if len(config['datasets']) != 1:
            errors.append("Only one dataset is allowed in the 'datasets' field.")
        ds = config['datasets'][0]

        if 'name' not in ds or 'path' not in ds or 'train' not in ds or 'test' not in ds:
            errors.append("Each dataset entry must contain 'name', 'path', 'train', and 'test' fields.")
        else:
            dataset_config = {
                'name': ds['name'],
                'train_paths': [os.path.join(ds['path'], p) for p in ds['train']],
                'test_paths': [os.path.join(ds['path'], p) for p in ds['test']]
            }

            try:
                color_to_class, class_to_color, class_name = get_color_maps(ds['name'])
                if color_to_class is not None:
                    dataset_config['color_to_class'] = color_to_class
                if class_to_color is not None:
                    dataset_config['class_to_color'] = class_to_color
                if class_name is not None:
                    dataset_config['class_name'] = class_name
                print(f"Successfully loaded color maps and class names for dataset '{ds['name']}'")
            except ValueError as e:
                errors.append(str(e))

            validated_config['ds'] = dataset_config
    else:
        errors.append("Missing 'datasets' configuration.")

    if errors:
        raise ValueError("Configuration validation errors:\n" + "\n".join(errors))

    os.makedirs(validated_config['model_save_dir'], exist_ok=True)
    return validated_config

def get_config(config_file=None):
    """加载 YAML 配置文件并验证配置参数"""
    if config_file:
        config = load_yaml_config(config_file)
    else:
        parser = argparse.ArgumentParser(description='Load Configuration File')
        parser.add_argument('--config', type=str, required=True, help='Path to the config file')
        args = parser.parse_args()
        config = load_yaml_config(args.config)
        config_file = args.config

    validated_config = validate_and_set_defaults(config)
    print(f"Loaded and validated config from {config_file}: {validated_config}")
    return Config(validated_config)
