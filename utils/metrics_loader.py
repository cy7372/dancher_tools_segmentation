# dancher_tools/utils/metrics_loader.py

def get_metrics(args):
    """
    根据配置文件中的任务类型和指定的 metrics 名称加载对应的指标函数。
    :param args: 包含任务类型和 metrics 列表的配置对象
    :return: 指标名称到指标函数的字典
    """
    try:
        # 默认使用 segmentation 任务类型，直接导入对应的预设指标
        from ..metrics import PRESET_METRICS

        # 如果 args.metrics 为 None，则不使用任何指标
        if args.metrics is None:
            return {}

        # 确保配置文件中的 metrics 是一个列表
        metrics_list = args.metrics if isinstance(args.metrics, list) else [m.strip() for m in args.metrics.split(',')]

        # 筛选出有效的指标函数
        metrics = {name: PRESET_METRICS[name] for name in metrics_list if name in PRESET_METRICS}

        # 警告用户任何未找到的指标
        missing_metrics = set(metrics_list) - set(metrics.keys())
        if missing_metrics:
            print(f"Warning: Some metrics are missing from presets and will be ignored: {missing_metrics}")

        # 确认返回的 metrics 字典有效
        if not metrics:
            print(f"No valid metrics found for task 'segmentation' with names: {metrics_list}")

        return metrics
    except ImportError as e:
        raise ImportError(f"Failed to load metrics for task 'segmentation': {e}")
    except Exception as ex:
        raise ValueError(f"Error in loading metrics: {ex}")
