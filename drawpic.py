import pandas as pd
import matplotlib.pyplot as plt
import re

# 原始数据（按你的输入格式整理）
data = {
    # Epsilon
    "adv_data/fmifgsm/inception_v3_eps8": [99.0, 50.6, 36.9, 31.3, 49.3, 13.7, 20.8, 26.6, 33.8],
    "adv_data/fmifgsm/inception_v3_eps10": [99.3, 60.2, 46.5, 41.2, 58.1, 19.5, 25.9, 34.6, 40.1],
    "adv_data/fmifgsm/inception_v3_eps12": [99.6, 68.5, 53.3, 49.3, 66.5, 22.5, 32.4, 39.7, 44.9],
    "adv_data/fmifgsm/inception_v3_eps14": [99.9, 73.8, 59.1, 56.6, 72.7, 27.9, 38.6, 46.4, 48.3],
    "adv_data/fmifgsm/inception_v3_eps16": [100.0, 77.1, 62.8, 61.1, 75.2, 33.0, 43.0, 52.0, 54.4],
    "adv_data/fmifgsm/inception_v3_eps18": [100.0, 81.5, 69.3, 66.2, 78.8, 37.6, 48.0, 56.4, 57.7],

    # Epoch
    "adv_data/fmifgsm/inception_v3_epoch5": [100.0, 78.8, 65.1, 62.5, 75.7, 33.4, 43.8, 51.6, 54.4],
    "adv_data/fmifgsm/inception_v3_epoch6": [100.0, 77.4, 63.6, 61.5, 74.8, 32.8, 44.0, 51.7, 53.2],
    "adv_data/fmifgsm/inception_v3_epoch7": [100.0, 78.3, 63.3, 61.3, 75.2, 31.4, 42.7, 51.5, 54.0],
    "adv_data/fmifgsm/inception_v3_epoch8": [100.0, 78.3, 65.0, 62.0, 75.7, 32.9, 43.5, 51.2, 52.5],
    "adv_data/fmifgsm/inception_v3_epoch9": [100.0, 77.3, 65.0, 60.9, 76.2, 34.0, 43.1, 50.5, 55.0],
    "adv_data/fmifgsm/inception_v3_epoch10": [100.0, 77.7, 64.6, 62.6, 76.1, 33.3, 44.5, 51.3, 53.8],

    # Number of Steps (num)
    "adv_data/fmifgsm/inception_v3_num5": [100.0, 78.4, 64.8, 61.4, 75.2, 31.3, 42.9, 51.2, 52.3],
    "adv_data/fmifgsm/inception_v3_num6": [100.0, 76.2, 63.8, 62.3, 74.7, 32.7, 43.3, 50.4, 52.2],
    "adv_data/fmifgsm/inception_v3_num7": [100.0, 78.6, 64.3, 61.2, 75.1, 31.4, 43.4, 50.8, 52.9],
    "adv_data/fmifgsm/inception_v3_num8": [100.0, 77.3, 65.0, 62.3, 74.6, 31.6, 43.3, 52.4, 53.0],
    "adv_data/fmifgsm/inception_v3_num9": [100.0, 77.9, 64.3, 61.1, 76.0, 31.4, 43.0, 50.9, 53.6],
    "adv_data/fmifgsm/inception_v3_num10": [100.0, 78.5, 64.0, 62.2, 75.7, 33.8, 43.7, 51.2, 55.0],

    # Beta
    "adv_data/fmifgsm/inception_v3_beta24": [100.0, 77.1, 64.7, 60.4, 74.7, 31.6, 41.8, 49.1, 53.0],
    "adv_data/fmifgsm/inception_v3_beta26": [100.0, 78.7, 64.4, 60.9, 75.6, 32.8, 42.7, 51.1, 52.9],
    "adv_data/fmifgsm/inception_v3_beta28": [100.0, 77.2, 63.8, 62.5, 75.4, 32.3, 43.4, 51.7, 54.7],
    "adv_data/fmifgsm/inception_v3_beta30": [100.0, 79.1, 64.1, 62.3, 76.2, 33.0, 44.0, 50.5, 53.6],
    "adv_data/fmifgsm/inception_v3_beta32": [100.0, 77.6, 64.7, 62.5, 76.6, 33.6, 44.4, 51.1, 54.3],
    "adv_data/fmifgsm/inception_v3_beta34": [100.0, 77.4, 65.0, 62.4, 75.5, 33.1, 42.4, 51.6, 54.3],
}

# 定义模型顺序和目标模型
all_models = [
    "inception_v3", "inception_v4", "resnetv2_50",
    "resnetv2_101", "inception_resnet_v2",
    "vit_base_patch16_224", "pit_b_224",
    "visformer_small", "swin_tiny_patch4_window7_224"
]
target_models = ["inception_v4", "resnetv2_50", "resnetv2_101", "inception_resnet_v2"]

# 提取目标模型的索引
target_indices = [all_models.index(m) for m in target_models]


# 解析数据到DataFrame
def parse_data(data):
    rows = []
    for key, values in data.items():
        # 提取参数（如 eps=8, epoch=5 等）
        param = re.search(r"(eps|epoch|num|beta)(\d+)", key)
        param_name = param.group(1)
        param_value = int(param.group(2))

        # 提取目标模型的成功率
        target_values = [values[i] for i in target_indices]
        rows.append({
            "param": param_name,
            "value": param_value,
            **{model: val for model, val in zip(target_models, target_values)}
        })
    return pd.DataFrame(rows)


df = parse_data(data)

# 绘制消融实验图表（以 eps 参数为例）
plt.figure(figsize=(10, 6))
for model in target_models:
    subset = df[df["param"] == "beta"]
    plt.plot(subset["value"], subset[model], marker="o", label=model)

plt.title("Ablation Study: Attack Success Rate vs. beta")
plt.xlabel("beta")
plt.ylabel("Success Rate (%)")
plt.legend()
plt.grid(True)
plt.show()