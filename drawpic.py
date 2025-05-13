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
    "adv_data/fmifgsm/inception_v3_num5": [100.0, 60.2, 48.5, 46.1, 57.8, 25.3, 35.1, 42.4, 45.7],
    "adv_data/fmifgsm/inception_v3_num10": [100.0, 71.8, 58.3, 55.9, 70.2, 30.5, 40.3, 48.1, 50.4],
    "adv_data/fmifgsm/inception_v3_num15": [100.0, 75.7, 61.8, 59.6, 73.9, 32.2, 42.5, 50.3, 52.8],
    "adv_data/fmifgsm/inception_v3_num20": [100.0, 77.9, 63.7, 61.8, 75.1, 33.1, 43.4, 51.0, 54.3],
    "adv_data/fmifgsm/inception_v3_num25": [100.0, 79.4, 65.1, 63.3, 76.6, 35.7, 45.9, 53.3, 56.6],
    "adv_data/fmifgsm/inception_v3_num30": [100.0, 79.6, 65.4, 63.4, 76.8, 35.9, 45.8, 53.4, 56.8],
    "adv_data/fmifgsm/inception_v3_num35": [100.0, 79.1, 65.7, 63.9, 76.7, 35.8, 46.0, 53.2, 56.9],
    "adv_data/fmifgsm/inception_v3_num40": [100.0, 79.5, 65.5, 64.0, 77.0, 36.0, 45.7, 52.4, 57.6],
    "adv_data/fmifgsm/inception_v3_num45": [100.0, 79.4, 65.3, 63.2, 76.7, 35.7, 45.8, 53.5, 56.4],
    "adv_data/fmifgsm/inception_v3_num50": [100.0, 79.5, 65.1, 63.3, 76.6, 35.8, 45.7, 53.2, 56.8],

    # Beta
    "adv_data/fmifgsm/inception_v3_beta2.2": [100.0, 68.3, 56.7, 54.9, 68.1, 29.8, 39.4, 46.1, 48.9],
    "adv_data/fmifgsm/inception_v3_beta2.4": [100.0, 72.5, 59.8, 58.2, 71.4, 31.2, 41.3, 47.8, 50.7],
    "adv_data/fmifgsm/inception_v3_beta2.6": [100.0, 75.9, 62.4, 60.7, 74.3, 32.5, 42.8, 49.2, 52.3],
    "adv_data/fmifgsm/inception_v3_beta2.8": [100.0, 78.3, 63.9, 61.8, 75.9, 32.8, 43.6, 50.1, 53.1],
    "adv_data/fmifgsm/inception_v3_beta3.0": [100.0, 79.1, 64.1, 62.3, 76.2, 33.0, 44.0, 50.5, 53.6],
    "adv_data/fmifgsm/inception_v3_beta3.2": [100.0, 78.8, 63.8, 61.7, 75.8, 32.9, 43.5, 50.0, 53.0],
    "adv_data/fmifgsm/inception_v3_beta3.4": [100.0, 78.0, 63.5, 61.5, 75.5, 32.7, 43.2, 49.8, 52.8],
    "adv_data/fmifgsm/inception_v3_beta3.6": [100.0, 77.0, 63.0, 61.0, 75.0, 32.5, 42.8, 49.5, 52.5],
    "adv_data/fmifgsm/inception_v3_beta3.8": [100.0, 76.0, 62.5, 60.5, 74.5, 32.3, 42.5, 49.2, 52.2],

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
        param = re.search(r"(eps|epoch|num|beta)(\d+(\.\d+)?)", key)
        param_name = param.group(1)
        param_value = float(param.group(2))

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