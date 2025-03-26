import torch
from ..utils import *
from ..attack import Attack

class PGN(Attack):
    """
    PGN (Penalizing Gradient Norm)
    'Boosting Adversarial Transferability by Achieving Flat Local Maxima (NeurIPS 2023)' (https://arxiv.org/abs/2306.05225)

    Arguments:
        model_name (str): 替代模型名称
        epsilon (float): 扰动预算（最大允许扰动）
        alpha (float): 单步步长
        beta (float): 邻域扰动范围系数（控制邻域样本的分布范围）
        gamma (float): 梯度平衡系数（控制当前梯度和预测梯度的权重）
        num_neighbor (int): 邻域样本数量（用于估计梯度方差）
        epoch (int): 迭代次数
        decay (float): 动量衰减系数
        targeted (bool): 是否为目标攻击
        random_start (bool): 是否随机初始化扰动
        norm (str): 扰动范数类型（l2或linfty）
        loss (str): 损失函数类型
        device (torch.device): 计算设备
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=3.0, gamma=0.5, num_neighbor=10, epoch=10, decay=1.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/pgn/resnet18 --attack pgn --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/pgn/resnet18 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=3.0, gamma=0.5, num_neighbor=10, epoch=10, decay=1., targeted=False,
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='PGN', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = epsilon / epoch  # 单步步长
        self.zeta = beta * epsilon  # 邻域扰动范围（beta控制相对epsilon的大小）
        self.gamma = gamma  # 平衡当前梯度和预测梯度的系数
        self.epoch = epoch  # 迭代次数
        self.decay = decay  # 动量衰减系数（设为1.0，即不衰减）
        self.num_neighbor = num_neighbor  # 邻域样本数量

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        计算邻域梯度的加权平均，核心步骤实现
        Args:
            data: 原始输入数据（未添加扰动）
            delta: 当前对抗扰动
            label: 真实标签（非目标攻击）或目标标签（目标攻击）
        Returns:
            averaged_gradient: 邻域梯度的加权平均值
        """
        averaged_gradient = 0
        for _ in range(self.num_neighbor):
            # 1. 随机生成邻域样本（在当前扰动delta周围添加均匀噪声）
            x_near = self.transform(data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))

            # 2. 计算邻域样本的梯度g1（当前点的梯度）
            logits = self.get_logits(x_near)  # Calculate the output of the x_near
            loss = self.get_loss(logits, label)  # Calculate the loss of the x_near
            g_1 = self.get_grad(loss, delta)  # Calculate the gradient of the x_near

            # 3. 预测下一步的候选样本x_next（基于梯度下降方向）
            x_next = self.transform(x_near + self.alpha*(-g_1 / (torch.abs(g_1).mean(dim=(1,2,3), keepdim=True))))

            # 4. 计算候选样本的梯度g2（预测点的梯度）
            logits = self.get_logits(x_next)  # Calculate the output of the x_next
            loss = self.get_loss(logits, label)  # Calculate the loss of the x_next
            g_2 = self.get_grad(loss, delta)  # Calculate the gradient of the x_next

            # 5. 累加加权梯度（gamma控制g1和g2的权重）
            averaged_gradient += (1-self.gamma)*g_1 + self.gamma*g_2  # Calculate the gradients

        return averaged_gradient / self.num_neighbor  # 返回平均梯度

    def forward(self, data, label, **kwargs):
        """
        PGN攻击的核心流程
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
        delta = self.init_delta(data)  # 初始化扰动（随机或零初始化）
        momentum, averaged_gradient = 0, 0  # 初始化动量和平均梯度

        # 迭代更新扰动
        for _ in range(self.epoch):

            # 计算邻域梯度的加权平均
            averaged_gradient = self.get_averaged_gradient(data, delta, label)

            # 更新动量（decay=1.0时为累积动量）
            momentum = self.get_momentum(averaged_gradient, momentum)

            # 根据动量更新扰动（并确保在epsilon范围内）
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()  # 返回最终扰动
