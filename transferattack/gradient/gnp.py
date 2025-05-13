import torch
from ..utils import *
from ..attack import Attack


class GNP(Attack):
    """
    GNP (Gradient Norm Penalty)
    'GNP Attack: Transferable Adversarial Examples via Gradient Norm Penalty (ICIP 2023)' (https://ieeexplore.ieee.org/abstract/document/10223158)

    Arguments:
        model_name (str): 替代模型名称
        epsilon (float): 扰动预算（最大允许扰动）
        alpha (float): 单步步长
        epoch (int): 迭代次数
        decay (float): 动量衰减系数。
        r (float): 邻域步长（控制邻域点的偏移）。
        beta (float): 正则化系数（控制梯度范数惩罚的权重）。
        targeted (bool): 是否为目标攻击
        random_start (bool): 是否随机初始化扰动
        norm (str): 扰动范数类型（l2或linfty）
        loss (str): 损失函数类型
        device (torch.device): 计算设备

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1, r=0.01, beta=0.8.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/gnp/resnet18 --attack gnp --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/gnp/resnet18 --eval
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1., r=0.01, beta=0.8,
                 targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='GNP', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.r = r
        self.beta = beta

    def forward(self, data, label, **kwargs):
        """
        The GNP attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)  # 初始化扰动
        momentum = 0  # 初始化动量

        for _ in range(self.epoch):
            # 计算当前梯度
            logits = self.get_logits(self.transform(data + delta))
            loss = self.get_loss(logits, label)
            g1 = self.get_grad(loss, delta)

            # 计算归一化梯度方向 g_p
            g_p = g1 / (g1.abs().mean(dim=(1, 2, 3), keepdim=True))

            # 计算邻域点的梯度 g2
            logits = self.get_logits(self.transform(data + delta + self.r * g_p))
            loss = self.get_loss(logits, label)
            g2 = self.get_grad(loss, delta)

            # 融合梯度 gt = (1 + beta) * g1 + beta * g2
            gt = (1 + self.beta) * g1 + self.beta * g2

            # 更新动量
            momentum = self.get_momentum(gt, momentum)
            # 更新扰动
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
