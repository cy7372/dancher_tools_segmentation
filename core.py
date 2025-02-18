import torch
import torch.nn as nn
import os
import glob
from datetime import datetime
import re
from tqdm import tqdm
from .utils import CombinedLoss, EarlyStopping
import numpy as np
import logging

class Core(nn.Module):
    def __init__(self):
        super(Core, self).__init__()
        self.model_name = None  # Default model name
        self.last_epoch = 0  # Initialize last_epoch
        self.best_val = 0
        self.optimizer = None
        self.criterion = None
        self.metrics = []

    def compile(self, criterion, optimizer=None, metrics=None, loss_weights=None):
        """
        设置模型的优化器、损失函数和评价指标。
        :param optimizer: 优化器实例
        :param criterion: 损失函数实例或损失函数列表
        :param metrics: 指标函数字典
        :param loss_weights: 损失函数对应的权重列表（如果 criterion 是列表）
        """
        self.optimizer = optimizer

        # 设置 criterion
        if isinstance(criterion, list):
            if len(criterion) > 1:
                self.criterion = CombinedLoss(losses=criterion, weights=loss_weights)
            elif len(criterion) == 1:
                self.criterion = criterion[0]  # 单一损失函数
            else:
                raise ValueError("Criterion list cannot be empty.")
        elif callable(criterion):
            self.criterion = criterion
        else:
            raise TypeError("Criterion should be a callable loss function or a list of callable loss functions.")

        # 设置指标函数字典
        self.metrics = {}
        if metrics is not None:
            for metric_name, metric_fn in metrics.items():
                if callable(metric_fn):
                    self.metrics[metric_name] = metric_fn
                else:
                    raise ValueError(f"Metric function '{metric_name}' is not callable.")

        print(f"Model compiled with metrics: {list(self.metrics.keys())}")

    def save(self, model_dir='./checkpoints', mode='latest'):
        """
        保存模型至指定目录。
        :param model_dir: 保存的文件夹路径
        :param mode: 保存模式，'latest'、'best' 或 'epoch'。
        """
        os.makedirs(model_dir, exist_ok=True)

        # 模式与文件名映射
        mode_to_filename = {
            'latest': f"{self.model_name}_latest.pth",
            'best': f"{self.model_name}_best.pth",
            'epoch': f"{self.model_name}_epoch_{self.last_epoch}.pth",
        }

        # 确认 mode 合法性
        if mode not in mode_to_filename:
            raise ValueError(f"Invalid mode '{mode}'. Valid options are: {list(mode_to_filename.keys())}")

        save_path = os.path.join(model_dir, mode_to_filename[mode])

        # 确定当前设备
        device = next(self.parameters()).device

        # 保存的状态字典
        save_dict = {
            'epoch': getattr(self, 'last_epoch', 0),
            'model_state_dict': self.state_dict(),  # 不改变当前设备，直接保存
            'best_val': getattr(self, 'best_val', None),
        }

        try:
            torch.save(save_dict, save_path, pickle_protocol=4)  # 高效保存
            print(f"Model successfully saved to {save_path}")
        except Exception as e:
            print(f"Failed to save model to {save_path}. Error: {str(e)}")

    def load(self, model_dir='./checkpoints', mode='latest', specified_path=None):
        """
        从指定路径加载模型。
        :param model_dir: 模型文件的目录
        :param mode: 加载模式，'latest'、'best' 或 'epoch'。
        :param specified_path: 直接指定的模型路径（若提供，将优先于 mode 和 model_dir）
        """
        if specified_path:
            load_path = specified_path
        else:
            if mode == 'latest':
                load_path = os.path.join(model_dir, f"{self.model_name}_latest.pth")
            elif mode == 'best':
                load_path = os.path.join(model_dir, f"{self.model_name}_best.pth")
            elif mode == 'epoch':
                load_path = os.path.join(model_dir, f"{self.model_name}_epoch_{self.last_epoch}.pth")
            else:
                raise ValueError("Invalid mode. Use 'latest', 'best', or 'epoch'.")

        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.last_epoch = checkpoint.get('epoch', 0)
            self.best_val = checkpoint.get('best_val', 0)
            print(f"Model successfully loaded from {load_path}, epoch: {self.last_epoch}, best_val: {self.best_val}.")
        else:
            print(f"No model found at {load_path}, starting from scratch.")
            self.last_epoch = 0
            self.best_val = 0

    def transfer(self, specified_path, strict=False):
        """
        使用指定路径加载预训练模型参数，并将符合条件的参数转移到当前模型（不改变训练状态）。
        :param specified_path: 指定的权重文件路径。
        :param strict: 是否严格匹配层结构。如果为False，将跳过不匹配的参数。
        """
        if not specified_path:
            raise ValueError("Transfer path is not specified. Please provide a valid path for transferring weights.")
        if not os.path.exists(specified_path):
            raise FileNotFoundError(f"The specified transfer path does not exist: {specified_path}")

        print(f"Transferring model parameters from {specified_path}")
        checkpoint = torch.load(specified_path, weights_only=False)
        checkpoint_state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_state_dict = self.state_dict()

        new_state_dict = {}
        missing_parameters = []
        extra_parameters = []

        for name, parameter in model_state_dict.items():
            if name in checkpoint_state_dict:
                if checkpoint_state_dict[name].size() == parameter.size():
                    new_state_dict[name] = checkpoint_state_dict[name]
                else:
                    extra_parameters.append(name)
            else:
                missing_parameters.append(name)

        self.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully transferred {len(new_state_dict)} parameters from {specified_path}")

        if missing_parameters:
            print(f"Parameters not found in checkpoint (using default): {missing_parameters}")
        if extra_parameters:
            print(f"Parameters in checkpoint but not used due to size mismatch: {extra_parameters}")

        print(f"Transfer completed. Matched: {len(new_state_dict)}, Missing: {len(missing_parameters)}, Size mismatch: {len(extra_parameters)}.")

class SegModel(Core):
    def __init__(self, num_classes, in_channels, img_size, model_save_dir, *args, **kwargs):
        super(SegModel, self).__init__(*args, **kwargs)
        self.model_name = None
        self.num_classes = num_classes
        self.img_size = img_size
        self.in_channels = in_channels

        self.logger = logging.getLogger('SegModel')
        self.logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        file_handler = logging.FileHandler(os.path.join(model_save_dir, 'training.log'))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def fit(self, train_loader, val_loader, class_names, num_epochs=500, model_save_dir='./checkpoints/', patience=15, delta=0.01, save_interval=1):
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        device = next(self.parameters()).device
        current_epoch = getattr(self, 'last_epoch', 0)
        total_epochs = num_epochs

        first_metric = list(self.metrics.keys())[0]


        for epoch in range(current_epoch + 1, total_epochs + 1):
            self.logger.info(f"\nStarting epoch {epoch}/{total_epochs}")
            self.last_epoch = epoch
            self.train()
            running_loss = 0.0

            for images, masks in tqdm(train_loader, desc='Training Batches', leave=False):
                images, masks = images.to(device), masks.to(device)

                self.optimizer.zero_grad()
                outputs = self(images)

                if isinstance(outputs, tuple):
                    outputs = outputs[-1]  # 获取最终输出 out0

                
                if outputs.size(1) != self.num_classes:
                    raise ValueError(f"Expected outputs with {self.num_classes} classes, got {outputs.size(1)}")



                loss = self.criterion(outputs, masks)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            self.logger.info(f'Epoch {epoch}/{total_epochs}, Loss: {epoch_loss:.4f}')

            if save_interval > 0 and epoch % save_interval == 0:
                self.save(model_dir=model_save_dir, mode='latest')
                self.logger.info(f'Model saved at epoch {epoch}.')

            val_loss, val_metrics, _ = self.evaluate(val_loader, class_names)
            val_first_metric = val_metrics.get(first_metric)

            if val_first_metric is not None and (self.best_val is None or val_first_metric > self.best_val):
                self.best_val = val_first_metric
                self.save(model_dir=model_save_dir, mode='best')
                self.logger.info(f"New best model saved with {first_metric}: {self.best_val:.4f}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break

        self.load(model_dir=model_save_dir, mode='best')
        self.logger.info(f'Training complete. Best {first_metric}: {self.best_val:.4f}')

    def predict(self, images):
        device = next(self.parameters()).device
        self.eval()
        images = images.to(device)

        with torch.no_grad():
            outputs = self(images)

        if outputs.size(1) != self.num_classes:
            raise ValueError(f"Expected outputs with {self.num_classes} classes, got {outputs.size(1)}")

        predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()

        if hasattr(self, 'class_to_color') and isinstance(self.class_to_color, dict):
            color_to_class = self.class_to_color
            color_image = np.zeros((predicted_labels.shape[0], predicted_labels.shape[1], 3), dtype=np.uint8)
            for cls, color in color_to_class.items():
                color_image[predicted_labels == cls] = color
            return color_image
        else:
            return predicted_labels

    def print_metrics(self, per_class_avg_metrics, class_names=None):
        if len(self.metrics) == 1:
            metric_name = list(self.metrics.keys())[0]
            header = f"{'Class':<15}"
            for cls in range(self.num_classes):
                if class_names:
                    header += f"{class_names[cls]:<15}"
                else:
                    header += f"{cls:<15}"
            header += f"{'mean':<15}"
            self.logger.info(header)

            metric_values = f"{metric_name:<15}"
            for cls in range(self.num_classes):
                metric_values += f"{per_class_avg_metrics[metric_name].get(cls, 0):<15.4f}"
            mean_value = round(np.mean([per_class_avg_metrics[metric_name].get(cls, 0) for cls in range(self.num_classes)]), 4)
            metric_values += f"{mean_value:<15.4f}"
            self.logger.info(metric_values)

        else:
            header = f"{'Class':<15}"
            for metric_name in self.metrics:
                header += f"{metric_name:<15}"
            self.logger.info(header)

            for cls in range(self.num_classes):
                class_name = class_names[cls] if class_names else cls
                class_metrics_str = f"{class_name:<15}"
                for metric_name in self.metrics:
                    class_metrics_str += f"{per_class_avg_metrics[metric_name].get(cls, 0):<15.4f}"
                self.logger.info(class_metrics_str)

            mean_row = f"{'mean':<15}"
            for metric_name in self.metrics:
                mean_value = round(np.mean([per_class_avg_metrics[metric_name].get(cls, 0) for cls in range(self.num_classes)]), 4)
                mean_row += f"{mean_value:<15.4f}"
            self.logger.info(mean_row)

    def evaluate(self, data_loader, class_names=None):
        device = next(self.parameters()).device
        self.eval()
        total_loss = 0.0
        all_predicted = []
        all_masks = []

        with torch.no_grad():
            for images, masks in tqdm(data_loader, desc='Evaluation Batches', leave=False):
                images = images.to(device)
                masks = masks.to(device)
                outputs = self(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[-1]  # 获取最终输出 out0


                if outputs.size(1) != self.num_classes:
                    raise ValueError(f"Expected outputs with {self.num_classes} classes, got {outputs.size(1)}")

                predicted_labels = torch.argmax(outputs, dim=1)
                all_predicted.append(predicted_labels.cpu())

                if masks is not None:
                    if masks.dim() > 3 and masks.size(1) > 1:
                        masks = masks.argmax(dim=1)
                    else:
                        masks = masks.squeeze(1)
                    all_masks.append(masks.cpu())

                    loss = self.criterion(outputs, masks)
                    total_loss += loss.item()

        avg_val_loss = total_loss / len(data_loader)

        all_predicted = torch.cat(all_predicted)
        all_masks = torch.cat(all_masks)

        per_class_avg_metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            scores = metric_fn(all_predicted, all_masks, self.num_classes)

            if isinstance(scores, (list, np.ndarray)):
                per_class_avg_metrics[metric_name] = {cls: scores[cls] for cls in range(self.num_classes)}
            else:
                per_class_avg_metrics[metric_name] = {cls: scores for cls in range(self.num_classes)}

        avg_metrics = {
            metric_name: np.mean(list(cls_metrics.values()))
            for metric_name, cls_metrics in per_class_avg_metrics.items()
        }

        self.logger.info(f'Validation Loss: {avg_val_loss:.4f}')

        self.print_metrics(per_class_avg_metrics, class_names)

        return avg_val_loss, avg_metrics, per_class_avg_metrics
