import torch

from ..utils import *
from ..attack import Attack

class MIFGSM(Attack):
    """
    MI-FGSM（动量迭代快速梯度符号法）
    'Boosting Adversarial Attacks with Momentum (CVPR 2018)'(https://arxiv.org/abs/1710.06081)
    通过引入动量项增强迭代攻击的稳定性和迁移性

    参数说明:
        model_name (str): 替代模型名称
        epsilon (float): 扰动预算（最大允许扰动）
        alpha (float): 单步步长（通常设为epsilon/epoch）
        epoch (int): 迭代次数
        decay (float): 动量衰减系数（论文中设为1.0，即累积动量）
        targeted (bool): 是否为目标攻击
        random_start (bool): 是否随机初始化扰动
        norm (str): 扰动范数类型（linfty或l2）
        loss (str): 损失函数类型（如交叉熵）
        device (torch.device): 计算设备

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --attack mifgsm --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='MI-FGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay