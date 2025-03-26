## Requirements:

- easydict for AT
- statsmodels for RS
- opencv-python for NRP

## Quick Start:
1. 请更改".sh "文件中的 "ATTACK_METHOD "名称，以评估您的方法！
2. 请按照以下说明直接运行'.sh'文件！

Notes: **我们已经设置了相对路径以便运行，所以您可以直接运行这些命令，无需做太多修改！**

## AT: https://github.com/locuslab/fast_adversarial/tree/master/ImageNet

```
sh at_defense.sh
```
默认使用 4 倍的扰动范围（4eps）进行评估。

## HGD: https://github.com/lfz/Guided-Denoise/tree/master/nips_deploy

```
sh hgd_defense.sh
```

## RS: https://github.com/locuslab/smoothing

```
sh rs_defense.sh
```
```
python defense/rs/predict.py /path/to/adv_data /path/to/noise_0.50/checkpoint.pth.tar  0.50 prediction_outupt --alpha 0.001 --N 1000 --skip 100 --batch 1
```

在单个 4090 GPU 上处理 1000 个样本大约需要 1 小时。

## NRP: https://github.com/Muzammal-Naseer/NRP

```
sh nrp_defense.sh
```

```
python defense/nrp/purify.py --dir=/path/to/adv_data --output=/path/to/save/purified_data --purifier NRP --model_pth /path/to/NRP.pth  --dynamic
```

然后，对净化后的数据进行评估，报告 ResNet101 目标模型的 ASR。

## DiffPure: https://github.com/NVlabs/DiffPure

```
sh diffpure_defense.sh
```
然后，评估净化后的数据，我们报告了在ResNet101目标模型上的语音识别率（ASR）。