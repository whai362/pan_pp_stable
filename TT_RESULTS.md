# Total-Text Detection

### PAN++ ResNet18 320 Training from Scratch
- Training
```
python train_tt.py --arch resnet18 --short_size 320 --img_size 320
```
- Test
```
python test_tt.py --arch resnet18 --short_size 320 --min_kernel_area 0.5 --min_area 50 --min_score 0.84 --resume [checkpoint path]
```

### PAN++ ResNet18 512 Training from Scratch
- Training
```
python train_tt.py --arch resnet18 --short_size 512 --img_size 512
```
- Test
```
python test_tt.py --arch resnet18 --short_size 512 --min_kernel_area 1.3 --min_area 130 --min_score 0.86 --resume [checkpoint path]
```

### PAN++ ResNet18 640 Training from Scratch
- Training
```
python train_tt.py --arch resnet18 --short_size 640 --img_size 640
```
- Test
```
python test_tt.py --arch resnet18 --short_size 640 --min_score 0.88 --resume [pretrained checkpoint path]
```

### PAN++ ResNet18 320 Pre-training + Fine-tuning
- Pre-training
```
python pretrain.py --dataset synth --arch resnet18 --img_size 640 --short_size 640 --epoch 1
```
- Training
```
python train_tt.py --arch resnet18 --short_size 320 --img_size 320 --pretrain [pretrained checkpoint path]
```
- Test
```
python test_tt.py --arch resnet18 --short_size 320 --min_kernel_area 0.5 --min_area 50 --min_score 0.84 --resume [checkpoint path]
```

### PAN++ ResNet18 512 Pre-training + Fine-tuning
- Pre-training
```
python pretrain.py --dataset synth --arch resnet18 --img_size 640 --short_size 640 --epoch 1
```
- Training
```
python train_tt.py --arch resnet18 --short_size 512 --img_size 512 --pretrain [pretrained checkpoint path]
```
- Test
```
python test_tt.py --arch resnet18 --short_size 512 --min_kernel_area 1.3 --min_area 130 --min_score 0.86 --resume [checkpoint path]
```

### PAN++ ResNet18 640 Pre-training + Fine-tuning
- Pre-training
```
python pretrain.py --dataset synth --arch resnet18 --img_size 640 --short_size 640 --epoch 1
```
- Training
```
python train_tt.py --arch resnet18 --short_size 640 --img_size 640 --pretrain [pretrained checkpoint path]
```
- Test
```
python test_tt.py --arch resnet18 --short_size 640 --min_score 0.88 --resume [pretrained checkpoint path]
```

# Total-Text End-to-End Recognition

### PAN++ ResNet18 640 Joint Training
- Training
```
python pretrain.py --dataset joint --arch resnet18 --with_rec True --epoch 3 --img_size 640 --short_size 640
```
- Test
```
python test_tt.py --arch resnet18 --resume release_models/pan_pp_joint_resnet18_640_with_rec.pth.tar --with_rec True  --short_size 640 --min_score 0.8
```
- Result
```
Calculated!{"recall": 0.5810383747178329, "precision": 0.7838002436053593, "hmean": 0.6673580502981591, "AP": 0}
```
- Test VOC F
```
python test_tt.py --arch resnet18 --resume release_models/pan_pp_joint_resnet18_640_with_rec.pth.tar --with_rec True --short_size 640 --min_score 0.8 --voc f --rec_ignore_score 0.8
```
- Result
```
Calculated!{"recall": 0.6799097065462754, "precision": 0.9001793185893604, "hmean": 0.7746913580246912, "AP": 0}
```

### PAN++ ResNet18 736 Joint Training
- Training
```
python pretrain.py --dataset joint --arch resnet18 --with_rec True --epoch 3 --img_size 736 --short_size 736
```
- Test
```
python test_tt.py --arch resnet18 --resume release_models/pan_pp_joint_resnet18_736_with_rec.pth.tar --with_rec True  --short_size 736 --min_score 0.8
```
- Result
```
Calculated!{"recall": 0.6009029345372461, "precision": 0.8003607937462417, "hmean": 0.6864363073749354, "AP": 0}
```
- Test VOC F
```
python test_tt.py --arch resnet18 --resume release_models/pan_pp_joint_resnet18_736_with_rec.pth.tar --with_rec True --short_size 736 --min_score 0.8 --voc f --rec_ignore_score 0.8
```
- Result
```
Calculated!{"recall": 0.6966139954853273, "precision": 0.900758902510216, "hmean": 0.7856415478615071, "AP": 0}
```

### PAN++ ResNet50 736 Joint Training
- Training
```
python pretrain.py --dataset joint --arch resnet50 --with_rec True --epoch 3 --img_size 736 --short_size 736
```
- Test
```
python test_tt.py --arch resnet50 --resume [checkpoint path] --with_rec True  --short_size 736 --min_score 0.8
```
- Test VOC F
```
python test_tt.py --arch resnet50 --resume [checkpoint path] --with_rec True --short_size 736 --min_score 0.8 --voc f --rec_ignore_score 0.8
```


### PAN++ ResNet18 640 Pre-training + Fine-tune
- Pre-training
```
python pretrain.py --dataset pretrain --arch resnet18 --with_rec True --epoch 3 --img_size 640 --short_size 640
```
- Training
```
python train_tt.py --arch resnet18 --with_rec True --epoch 100 --lr 1e-4 --kernel_scale 0.5 --img_size 640 --short_size 640 --pretrain [pretrained checkpoint path]

```
- Test
```
python test_tt.py --arch resnet18 --resume [checkpoint path] --with_rec True  --short_size 640 --min_score 0.8
```
- Test VOC F
```
python test_tt.py --arch resnet18 --resume [checkpoint path] --with_rec True --short_size 640 --min_score 0.8 --voc f --rec_ignore_score 0.8
```

### PAN++ ResNet18 736 Pre-training + Fine-tune
- Pre-training
```
python pretrain.py --dataset pretrain --arch resnet18 --with_rec True --epoch 3 --img_size 640 --short_size 640
```
- Training
```
python train_tt.py --arch resnet18 --with_rec True --epoch 100 --lr 1e-4 --kernel_scale 0.5 --img_size 736 --short_size 736 --pretrain [pretrained checkpoint path]
```
- Test
```
python test_tt.py --arch resnet18 --resume [checkpoint path] --with_rec True  --short_size 736 --min_score 0.8
```
- Test VOC F
```
python test_tt.py --arch resnet18 --resume [checkpoint path] --with_rec True  --short_size 736 --min_score 0.8 --voc f --rec_ignore_score 0.8
```

### PAN++ ResNet50 736 Pre-training + Fine-tune
- Pre-training
```
python pretrain.py --dataset pretrain --arch resnet50 --with_rec True --epoch 3 --img_size 640 --short_size 640
```
- Training
```
python train_tt.py --arch resnet50 --with_rec True --epoch 100 --lr 1e-4 --kernel_scale 0.5 --img_size 736 --short_size 736 --pretrain [pretrained checkpoint path]
```
- Test
```
python test_tt.py --arch resnet50 --resume [checkpoint path] --with_rec True  --short_size 736 --min_score 0.8
```
- Test VOC F
```
python test_tt.py --arch resnet50 --resume [checkpoint path] --with_rec True  --short_size 736 --min_score 0.8 --voc f --rec_ignore_score 0.8
```