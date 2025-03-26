import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import pandas as pd
import timm
import os

img_height, img_width = 224, 224  # 定义图像的高度和宽度（224x224 是常见的输入尺寸）
img_max, img_min = 1., 0  # 像素值范围（归一化后）

# 论文中使用的 CNN 模型列表
# cnn_model_paper = ['resnet50', 'vgg16', 'mobilenet_v2', 'inception_v3']
cnn_model_paper = ['inception_v3', 'inception_v4', 'resnetv2_50', 'resnetv2_101', 'inception_resnet_v2']

# 论文中使用的 Vision Transformer 模型列表
vit_model_paper = ['vit_base_patch16_224', 'pit_b_224', 'visformer_small', 'swin_tiny_patch4_window7_224']

# 从 torchvision 加载的 CNN 模型列表
cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101', 'resnext50_32x4d', 'densenet121', 'mobilenet_v2']

# 从 timm 加载的 Vision Transformer 模型列表
vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small', 'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

# 特定任务（如目标攻击）的 ViT 模型列表
tgr_vit_model_list = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small', 'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base']

# 生成对抗样本时的目标类别
generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]


# def load_pretrained_model(cnn_model=[], vit_model=[]):
#     """
#     动态加载预训练模型
#     """
#     for model_name in cnn_model:
#         yield model_name, models.__dict__[model_name](weights="DEFAULT")
#         # yield model_name, models.__dict__[model_name](weights="IMAGENET1K_V1")
#
#     for model_name in vit_model:
#         yield model_name, timm.create_model(model_name, pretrained=True)


def load_pretrained_model(cnn_model=[], vit_model=[]):
    """
    动态加载预训练模型
    """
    for model_name in cnn_model:
        if model_name in models.__dict__:  # torchvision 模型
            yield model_name, models.__dict__[model_name](weights="DEFAULT")
        else:  # timm 模型
            try:
                yield model_name, timm.create_model(model_name, pretrained=True)
            except Exception as e:
                raise ValueError(f"Failed to load model '{model_name}'. Error: {e}")

    for model_name in vit_model:  # ViT 模型通常来自 timm
        yield model_name, timm.create_model(model_name, pretrained=True)

def wrap_model(model):
    """
    包装模型 ：在模型前添加预处理层（调整尺寸 + 归一化）。
    自动适配 ：根据模型类型（torchvision 或 timm）自动选择均值和标准差。处理 Inception 模型的特殊尺寸（299x299）。
    """
    model_name = model.__class__.__name__
    Resize = 224  # 默认尺寸
    
    if hasattr(model, 'default_cfg'):
        """timm.models：使用默认配置的均值和标准差"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
        Resize = model.default_cfg.get('input_size', (3, 224, 224))[1]  # 获取输入尺寸
    else:
        """torchvision.models：根据模型类型设置预处理参数"""
        if 'Inc' in model_name or 'inception' in model_name.lower():
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            Resize = 299
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            Resize = 224

    # 添加预处理层
    PreprocessModel = PreprocessingModel(Resize, mean, std)
    return torch.nn.Sequential(PreprocessModel, model)


def save_images(output_dir, adversaries, filenames):
    """
    将生成的对抗样本保存到指定目录。
    """
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


# 定义一个预处理模块，包含图像缩放和标准化操作。
class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.resize = transforms.Resize(resize)
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, x):
        return self.normalize(self.resize(x))


# 定义一个集成模型，将多个模型的输出进行组合。
class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


# 定义一个自定义数据集类，用于加载对抗样本及其标签。
class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, target_class=None, eval=False):
        self.targeted = targeted
        self.target_class = target_class
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))

        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/labels.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            if self.target_class:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], self.target_class] for i in range(len(dev))}
            else:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'],
                                             dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l


if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted', targeted=True, eval=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break

