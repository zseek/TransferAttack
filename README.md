<h1 align="center">TransferAttack</h1>



## 环境
+ Python >= 3.6
+ PyTorch >= 1.12.1
+ Torchvision >= 0.13.1
+ timm >= 0.6.12

```
pip install -r requirements.txt
```


## 使用

```
python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --attack mifgsm --model=resnet18
python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --eval
```
