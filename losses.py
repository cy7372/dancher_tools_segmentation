import torch.nn as nn
import torch

class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        # 对于多分类问题，使用 CrossEntropyLoss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """输入: 
        - inputs: 模型输出（未经过 softmax 的 raw logits），形状为 [batch_size, num_classes, height, width]
        - targets: 目标标签，整数型标签，形状为 [batch_size, height, width]
        输出: 损失值
        """
        return self.loss_fn(inputs, targets)

# 自定义的 Dice 损失函数
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)  # 对于多分类，使用 softmax 激活

        # 如果 targets 是一个标签（单通道的），需要转化为 one-hot 编码
        if targets.dim() == 3:  # shape = [batch_size, height, width]
            targets = torch.nn.functional.one_hot(targets.long(), num_classes=self.num_classes)
            targets = targets.permute(0, 3, 1, 2).float()  # 转换成 shape = [batch_size, num_classes, height, width]

        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        total = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice.mean()


# 自定义的 Focal 损失函数
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        # 对于多分类，使用 softmax 激活
        inputs = torch.softmax(inputs, dim=1)

        # 计算交叉熵损失
        ce_loss = self.bce(inputs, targets)

        # 计算 focal loss
        pt = torch.exp(-ce_loss)  # pt 是预测为正确类别的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# 自定义的 Jaccard 损失函数
class JaccardLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 使用 softmax 激活来处理多分类任务
        inputs = torch.softmax(inputs, dim=1)

        # 计算交集
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        
        # 计算并集
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        
        # 计算 Jaccard 损失
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - jaccard.mean()


# 分割任务的预设损失函数
PRESET_LOSSES = {
    "ce": CrossEntropyLoss,               # 多类交叉熵损失
    "dice": DiceLoss,  # Dice 损失（自定义）
    "focal": FocalLoss,  # Focal 损失（自定义）
    "jaccard": JaccardLoss  # Jaccard 损失（自定义）
}
