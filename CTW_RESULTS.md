# CTW1500 Detection

### PAN++ ResNet18 320 Training from Scratch
- Training
```
python train_ctw.py --arch resnet18 --short_size 320 --img_size 320
```
- Test
```
python test_ctw.py --arch resnet18 --short_size 320 --min_kernel_area 0.5 --min_area 50 --min_score 0.86 --resume [checkpoint path]
```
- Result
```
p: 0.8072, r: 0.7396, f: 0.7719
```

### PAN++ ResNet18 512 Training from Scratch
- Training
```
python train_ctw.py --arch resnet18 --short_size 512 --img_size 512
```
- Test
```
python test_ctw.py --arch resnet18 --short_size 512 --min_kernel_area 1.3 --min_area 130 --min_score 0.88 --resume [checkpoint path]
```

### PAN++ ResNet18 640 Training from Scratch
- Training
```
python train_ctw.py --arch resnet18 --short_size 640 --img_size 640
```
- Test
```
python test_ctw.py --arch resnet18 --short_size 640 --min_score 0.88 --resume [pretrained checkpoint path]
```

### PAN++ ResNet18 320 Pre-training + Fine-tuning
- Pre-training
```
python pretrain.py --dataset synth --arch resnet18 --img_size 640 --short_size 640 --epoch 1
```
- Training
```
python train_ctw.py --arch resnet18 --short_size 320 --img_size 320 --pretrain [pretrained checkpoint path]
```
- Test
```
python test_ctw.py --arch resnet18 --short_size 320 --min_kernel_area 0.5 --min_area 50 --min_score 0.86 --resume [checkpoint path]
```

### PAN++ ResNet18 512 Pre-training + Fine-tuning
- Pre-training
```
python pretrain.py --dataset synth --arch resnet18 --img_size 640 --short_size 640 --epoch 1
```
- Training
```
python train_ctw.py --arch resnet18 --short_size 512 --img_size 512 --pretrain [pretrained checkpoint path]
```
- Test
```
python test_ctw.py --arch resnet18 --short_size 512 --min_kernel_area 1.3 --min_area 130 --min_score 0.88 --resume [checkpoint path]
```

### PAN++ ResNet18 640 Pre-training + Fine-tuning
- Pre-training
```
python pretrain.py --dataset synth --arch resnet18 --img_size 640 --short_size 640 --epoch 1
```
- Training
```
python train_ctw.py --arch resnet18 --short_size 640 --img_size 640 --pretrain [pretrained checkpoint path]
```
- Test
```
python test_ctw.py --arch resnet18 --short_size 640 --min_score 0.88 --resume [pretrained checkpoint path]
```