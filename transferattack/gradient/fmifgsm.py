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
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=3.0, num_neighbor=10, epoch=10, decay=1., targeted=False,
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='FMI-FGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = epsilon / epoch  # 单步步长
        self.zeta = beta * epsilon  # 邻域扰动范围（beta控制相对epsilon的大小）
        self.epoch = epoch  # 迭代次数
        self.decay = decay  # 动量衰减系数（设为1.0，即不衰减）
        self.num_neighbor = num_neighbor  # 邻域样本数量

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        计算邻域平均梯度（融合当前梯度与邻域梯度）
        """
        # 初始化梯度
        averaged_gradient = 0

        # 邻域采样循环
        for _ in range(self.num_neighbor):
            # 1. 随机生成邻域样本（在当前扰动delta周围添加均匀噪声）
            x_near = self.transform(data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))

            # 2. 计算邻域样本的梯度grad（当前点的梯度）
            logits = self.get_logits(x_near)  # Calculate the output of the x_near
            loss = self.get_loss(logits, label)  # Calculate the loss of the x_near
            grad = self.get_grad(loss, delta)  # Calculate the gradient of the x_near

            averaged_gradient += grad

        averaged_gradient = averaged_gradient / self.num_neighbor
        return averaged_gradient


    def forward(self, data, label, **kwargs):
        """
        FMI-FGSM攻击的核心流程
        Args:
            data: 输入图像（形状：NCHW）
            label: 标签（非目标攻击时为真实标签，目标攻击时为目标标签）
        Returns:
            delta: 最终的对抗扰动
        """
        if self.targeted:  # 若为为定向攻击
            assert len(label) == 2
            label = label[1]  # 取目标标签, 第二个元素是目标标签张量。

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)  # 初始化扰动
        momentum = 0  # 动量初始化

        # 主迭代循环
        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))  # 获取模型输出
            loss = self.get_loss(logits, label)  # 计算损失
            grad0 = self.get_grad(loss, delta)  # 计算梯度

            # 1. 计算融合梯度
            grad = grad0 + self.get_averaged_gradient(data, delta, label)
            # 2. 动量更新
            momentum = self.get_momentum(grad, momentum)
            # 3. 更新扰动
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()  # 返回最终扰动