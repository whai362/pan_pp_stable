# RCTW-17 End-to-End Recognition

### PAN++ ResNet18 736 Pre-training + Fine-tune
- Pre-training
```
python pretrain.py --dataset joint --arch resnet18 --with_rec True --epoch 3 --img_size 640 --short_size 640 --rec_cscale 4
```
- Training
```
python train_rctw.py --arch resnet18 --img_size 736 --short_size 736 --with_rec True --pretrain [pretrained checkpoint path]
```
- Test
```
python test_rctw.py --arch resnet18 --with_rec True --resume [checkpoint path]
```

### PAN++ ResNet50 896 Pre-training + Fine-tune
- Pre-training
```
python pretrain.py --dataset joint --arch resnet50 --with_rec True --epoch 3 --img_size 896 --short_size 896 --rec_cscale 4
```
- Training
```
python train_rctw.py --arch resnet50 --img_size 896 --short_size 896 --with_rec True --pretrain [pretrained checkpoint path]
```
- Test
```
python test_rctw.py --arch resnet50 --with_rec True --short_size 896 --resume [checkpoint path]
```