# MSRA Detection

### PAN++ ResNet18 736 Training from Scratch
- Training
```
python train_msra.py --arch resnet18 --img_size 736 --short_size 736
```
- Test
```
python test_msra.py --arch resnet18 --resume [checkpoint path]
```

### PAN++ ResNet18 896 Training from Scratch
- Training
```
python train_msra.py --arch resnet18 --img_size 896 --short_size 896
```
- Test
```
python test_msra.py --arch resnet18 --resume [checkpoint path] --short_size 896
```

### PAN++ ResNet18 736 Pre-training + Fine-tuning
- Pre-training
```
python pretrain.py --dataset synth --arch resnet18 --img_size 640 --short_size 640 --epoch 1
```
- Training
```
python train_msra.py --arch resnet18 --img_size 736 --short_size 736 --pretrain [pretrained checkpoint path]
```
- Test
```
python test_msra.py --arch resnet18 --resume [checkpoint path]
```

### PAN++ ResNet18 896 Pre-training + Fine-tuning
- Pre-training
```
python pretrain.py --dataset synth --arch resnet18 --img_size 896 --short_size 896 --epoch 1
```
- Training
```
python train_msra.py --arch resnet18 --img_size 896 --short_size 896 --pretrain [pretrained checkpoint path]
```
- Test
```
python test_msra.py --resume  --short_size 896
```
- Result
```
p: 0.8928, r: 0.8596, f: 0.8759
```