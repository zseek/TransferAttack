import torch
import torch.nn as nn

import numpy as np

from .utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attack(object):
    """
    所有对抗攻击算法的基类，攻击算法在 /transferattack/gradient 路径下
    """
    def __init__(self, attack, model_name, epsilon, targeted, random_start, norm, loss, device=None):
        """
        初始化超参数

        Arguments：
            attack (str): 攻击名称（如 'FGSM', 'PGD'）
            model_name (str/list): 代理模型名称或模型列表（用于集成攻击）
            epsilon (float): 最大扰动范围（控制攻击强度）
            targeted (bool): 是否为目标攻击
            random_start (bool): 是否使用随机初始化扰动
            norm (str): 扰动范数类型（'l2' 或 'linfty'）
            loss (str): 损失函数类型（如 'crossentropy'）
            device (torch.device): 计算设备（默认与模型一致）
        """
        if norm not in ['l2', 'linfty']:
            raise Exception("不支持的范数类型 {}".format(norm))
        self.attack = attack
        self.model = self.load_model(model_name)  # 加载代理模型
        self.epsilon = epsilon
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm

        # 设备管理：优先使用传入的device，否则使用模型所在设备
        if isinstance(self.model, EnsembleModel):
            self.device = self.model.device
        else:
            self.device = next(self.model.parameters()).device if device is None else device
        self.loss = self.loss_function(loss)  # 初始化损失函数

    def load_model(self, model_name):
        """
        加载代理模型（支持单模型或集成模型）
        参数:
            model_name (str/list): 模型名称或名称列表
        返回:
            model (torch.nn.Module): 包装后的模型（已加载预训练权重）
        """
        def load_single_model(model_name):
            # 优先从 torchvision 加载
            if model_name in models.__dict__.keys():
                print(f"=> 从 torchvision 加载模型: {model_name}")
                model = models.__dict__[model_name](weights="DEFAULT")
            elif model_name in timm.list_models():
                print(f"=> 从 timm 加载模型: {model_name}")
                model = timm.create_model(model_name, pretrained=True)
            else:
                raise ValueError(f"不支持的模型: {model_name}")
            return wrap_model(model.eval().to(device))  # 包装模型（添加预处理层）

        # 处理集成模型
        if isinstance(model_name, list):
            return EnsembleModel([load_single_model(name) for name in model_name])
        else:
            return load_single_model(model_name)

    def forward(self, data, label, **kwargs):
        """
        一般攻击流程

        Arguments:
            data (N, C, H, W): 输入图像的张量
            labels (N,): 真实标签的张量（如果是非定向攻击）
            labels (2,N): 包含 [真实标签, 定向标签] 的张量（如果是定向攻击）
        """
        if self.targeted:  # 处理有目标攻击的标签格式
            assert len(label) == 2
            label = label[1]  # 取目标标签, 第二个元素是目标标签张量。

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)  # 初始化扰动
        momentum = 0

        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))  # 获取模型输出
            loss = self.get_loss(logits, label)  # 计算损失
            grad = self.get_grad(loss, delta)  # 计算梯度

            momentum = self.get_momentum(grad, momentum)  # 更新动量

            delta = self.update_delta(delta, data, momentum, self.alpha)  # 更新扰动

        return delta.detach()

    def get_logits(self, x, **kwargs):
        """
        推理阶段，当攻击需要改变模型（例如，集成模型攻击、ghost 等）或输入（例如 DIM、SIM 等）时，应重写此方法。
        """
        return self.model(x)

    def get_loss(self, logits, label):
        """
        损失计算，当攻击需要改变损失计算时（例如 ATA 等），应重写此方法。
        """
        # Calculate the loss
        return -self.loss(logits, label) if self.targeted else self.loss(logits, label)


    def get_grad(self, loss, delta, **kwargs):
        """
        梯度计算，当攻击需要调整梯度时（例如 TIM、方差调节、增强动量等），应重写此方法。
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    def get_momentum(self, grad, momentum, **kwargs):
        """
        动量计算
        """
        return momentum * self.decay + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

    def init_delta(self, data, **kwargs):
        """
        初始化扰动（支持随机初始化）
        参数:
            data (Tensor): 原始输入图像
        返回:
            delta (Tensor): 初始化后的扰动
        """
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-self.epsilon, self.epsilon)  # L_infty 范数：均匀分布
            else:
                delta.normal_(-self.epsilon, self.epsilon)  # L2 范数：正态分布
                d_flat = delta.view(delta.size(0), -1)  # 归一化到 epsilon 范围内
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0,1).to(self.device)
                delta *= r/n*self.epsilon
            delta = clamp(delta, img_min-data, img_max-data)  # 确保扰动后的图像在有效范围内
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        """
        更新扰动（根据梯度和步长）
        参数:
            delta (Tensor): 当前扰动
            data (Tensor): 原始图像
            grad (Tensor): 梯度
            alpha (float): 更新步长
        返回:
            new_delta (Tensor): 更新后的扰动
        """
        # L_infty 范数：直接使用符号函数
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        # L2 范数：归一化梯度并缩放
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)  # 重新归一化到 epsilon 范围内

        delta = clamp(delta, img_min-data, img_max-data)  # 确保扰动后的图像在合法范围内
        return delta.detach().requires_grad_(True)

    def loss_function(self, loss):
        """
        获取损失函数
        """
        if loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise Exception("不支持的损失函数 {}".format(loss))

    def transform(self, data, **kwargs):
        return data

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.forward(*input, **kwargs)
