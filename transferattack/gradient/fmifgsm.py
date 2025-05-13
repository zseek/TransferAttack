import torch

from ..utils import *
from ..attack import Attack


class FMIFGSM(Attack):
    """
    FMI-FGSM Attack

    Arguments:
        model_name (str): 替代模型名称
        epsilon (float): 扰动预算（最大允许扰动）
        alpha (float): 单步步长
        beta (float): 邻域扰动范围系数（控制邻域样本的分布范围）
        num_neighbor (int): 邻域样本数量（用于估计梯度方差）
        epoch (int): 迭代次数
        decay (float): 动量衰减系数
        targeted (bool): 是否为目标攻击
        random_start (bool): 是否随机初始化扰动
        norm (str): 扰动范数类型（l2或linfty）
        loss (str): 损失函数类型
        device (torch.device): 计算设备

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=3.0, num_neighbor=10, epoch=10, decay=1.

    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, beta=3.0, num_neighbor=10, epoch=10, decay=1.,
                 targeted=False,
                 random_start=False, norm='linfty', loss='crossentropy', device=None, attack='FMI-FGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = epsilon / epoch  # 单步步长
        self.zeta = beta * epsilon  # 邻域扰动范围（beta控制相对epsilon的大小）
        self.epoch = epoch  # 迭代次数
        self.decay = decay  # 动量衰减系数（设为1.0，即不衰减）
        self.num_neighbor = num_neighbor  # 邻域样本数量

    def forward(self, data, label, **kwargs):
        """
        FMPG攻击的核心流程

        Args:
            data: 输入图像（形状：NCHW）
            label: 标签（非目标攻击时为真实标签，目标攻击时为目标标签）

        Returns:
            delta: 最终的对抗扰动
        """
        if self.targeted:  # 若为目标攻击
            assert len(label) == 2
            label = label[1]  # 取目标标签

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # 初始化扰动
        delta = self.init_delta(data)
        momentum = 0  # 动量初始化

        # 主迭代循环
        for t in range(self.epoch):
            # Step 1: 计算当前梯度 g'
            logits = self.get_logits(self.transform(data + delta))
            loss = self.get_loss(logits, label)
            current_grad = self.get_grad(loss, delta)

            # Step 2: 计算预测点 x^*
            predicted_point = data + delta - self.alpha * (current_grad / (current_grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8))

            # Step 3: 在预测点 x^* 的邻域内采样并计算平均梯度
            averaged_gradient = 0
            for _ in range(self.num_neighbor):
                # 在预测点 x^* 的邻域内随机采样
                x_near = self.transform(
                    predicted_point + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))

                # 计算邻域样本的梯度
                logits = self.get_logits(x_near)
                loss = self.get_loss(logits, label)
                grad = self.get_grad(loss, delta)

                averaged_gradient += grad

            averaged_gradient /= self.num_neighbor  # 平均梯度

            # Step 4: 融合梯度
            fused_gradient = current_grad + averaged_gradient

            # Step 5: 更新动量
            momentum = self.decay * momentum + fused_gradient / (fused_gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)

            # Step 6: 更新扰动
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()  # 返回最终扰动
