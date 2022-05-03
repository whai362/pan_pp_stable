#### ICDAR 2015 Detection

- PAN++ ResNet18 736 Training from Scratch
Training
```
python train_ic15.py --arch resnet18 --img_size 736 --short_size 736
```
Test
```
python test_ic15.py --resume release_models/pan_pp_ic15_resnet18_736.pth.tar
```
Result
```
Calculated!{"recall": 0.7852672123254695, "precision": 0.8437661665804449, "hmean": 0.8134663341645886, "AP": 0}
```

- PAN++ ResNet18 896 Training from Scratch
Training
```
python train_ic15.py --arch resnet18 --img_size 896 --short_size 896
```
Test
```
python test_ic15.py --resume [checkpoint path] --short_size 896 --min_score 0.87
```
Result
```
Calculated!{"recall": 0.7857486759749639, "precision": 0.864406779661017, "hmean": 0.8232030264817151, "AP": 0}
```

- PAN++ ResNet18 736 Pre-training + Fine-tuning
Pre-training
```
python pretrain.py --dataset synth --arch resnet18 --img_size 640 --short_size 640 --epoch 1
```
Training
```
python train_ic15.py --arch resnet18 --img_size 736 --short_size 736 --pretrain [pretrain checkpoint path]
```
Test
```
python test_ic15.py --resume [checkpoint path]
```
Result
```
Calculated!{"recall": 0.8040442946557534, "precision": 0.8594956253216676, "hmean": 0.8308457711442786, "AP": 0}
```

- PAN++ ResNet18 896 Pre-training + Fine-tuning
Pre-training
```
python pretrain.py --dataset synth --arch resnet18 --img_size 896 --short_size 896 --epoch 1
```
Training
```
python train_ic15.py --arch resnet18 --img_size 896 --short_size 896 --pretrain [pretrain checkpoint path]
```
Test
```
python test_ic15.py --resume [checkpoint path] --short_size 896 --min_score 0.86
```
Result
```
Calculated!{"recall": 0.8074145402022147, "precision": 0.8873015873015873, "hmean": 0.8454751701537686, "AP": 0}
```

#### ICDAR 2015 End-to-End Recognition

- PAN++ ResNet18 736 Joint Training
Training
```
python pretrain.py --dataset joint --arch resnet18 --with_rec True --epoch 3 --img_size 736 --short_size 736
```
Test
```
python test_ic15.py --resume release_models/pan_pp_joint_resnet18_736_with_rec.pth.tar --with_rec True --min_score 0.8 --rec_ignore_score 0.93
```
Result
```
Calculated!{"recall": 0.5402022147327876, "precision": 0.8342007434944237, "hmean": 0.6557568673290474, "AP": 0}
```
Test VOC G
```
python test_ic15.py --resume release_models/pan_pp_joint_resnet18_736_with_rec.pth.tar --with_rec True --min_score 0.8 --rec_ignore_score 0.89 --voc g
```
Result VOC G
```
Calculated!{"recall": 0.5546461242176216, "precision": 0.8193456614509246, "hmean": 0.661498708010336, "AP": 0}
```
Test VOC W
```
python test_ic15.py --resume release_models/pan_pp_joint_resnet18_736_with_rec.pth.tar --with_rec True --min_score 0.8 --rec_ignore_score 0.79 --unalpha_score 0.995 --voc w
```
Result VOC W
```
Calculated!{"recall": 0.6389022628791526, "precision": 0.9014945652173914, "hmean": 0.7478162862778248, "AP": 0}
```
Test VOC S
```
python test_ic15.py --resume release_models/pan_pp_joint_resnet18_736_with_rec.pth.tar --with_rec True --min_score 0.5 --rec_ignore_score 0.68 --unalpha_score 0.995 --edit_dist_score 0.5 --voc s
```
Result VOC S
```
Calculated!{"recall": 0.7029369282619162, "precision": 0.9119300437226733, "hmean": 0.7939097335508428, "AP": 0}
```

- PAN++ ResNet18 896 Joint Training
Training
```
python pretrain.py --dataset joint --arch resnet18 --with_rec True --epoch 3 --img_size 896 --short_size 896
```
Test
```
python test_ic15.py --resume release_models/pan_pp_joint_resnet18_896_with_rec.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.93
```
Result
```
Calculated!{"recall": 0.5604236880115552, "precision": 0.8202959830866807, "hmean": 0.665903890160183, "AP": 0}
```
Test VOC G
```
python test_ic15.py --resume release_models/pan_pp_joint_resnet18_896_with_rec.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.90 --voc g
```
Result VOC G
```
Calculated!{"recall": 0.574867597496389, "precision": 0.8139059304703476, "hmean": 0.6738148984198646, "AP": 0}
```
Test VOC W
```
python test_ic15.py --resume release_models/pan_pp_joint_resnet18_896_with_rec.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.8 --unalpha_score 0.995 --voc w
```
Result VOC W
```
Calculated!{"recall": 0.6639383726528647, "precision": 0.8989569752281616, "hmean": 0.7637773469952922, "AP": 0}
```
Test VOC S
```
python test_ic15.py --resume release_models/pan_pp_joint_resnet18_896_with_rec.pth.tar --with_rec True --short_size 896 --min_score 0.5 --rec_ignore_score 0.65 --unalpha_score 0.995 --edit_dist_score 0.5 --voc s
```
Result VOC S
```
Calculated!{"recall": 0.731824747231584, "precision": 0.918429003021148, "hmean": 0.8145766345123258, "AP": 0}
```

- PAN++ ResNet50 896 Joint Training
Training
```
python pretrain.py --dataset joint --arch resnet50 --with_rec True --epoch 3 --img_size 896 --short_size 896
```
Test
```
python test_ic15.py --arch resnet50 --resume release_models/pan_pp_joint_resnet50_896_with_rec.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.93
```
Result
```
Calculated!{"recall": 0.5763119884448724, "precision": 0.8289473684210527, "hmean": 0.679920477137177, "AP": 0}
```
Test VOC G
```
python test_ic15.py --arch resnet50 --resume release_models/pan_pp_joint_resnet50_896_with_rec.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.88 --voc g
```
Result VOC G
```
Calculated!{"recall": 0.6023110255175734, "precision": 0.8160469667318982, "hmean": 0.6930747922437673, "AP": 0}
```
Test VOC W
```
python test_ic15.py --arch resnet50 --resume release_models/pan_pp_joint_resnet50_896_with_rec.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.78 --unalpha_score 0.995 --voc w
```
Result VOC W
```
Calculated!{"recall": 0.6836783822821377, "precision": 0.8981657179000633, "hmean": 0.7763805358119191, "AP": 0}
```
Test VOC S
```
python test_ic15.py --arch resnet50 --resume release_models/pan_pp_joint_resnet50_896_with_rec.pth.tar --with_rec True --short_size 896 --min_score 0.5 --rec_ignore_score 0.72 --unalpha_score 0.995 --edit_dist_score 0.5 --voc s
```
Result VOC S
```
Calculated!{"recall": 0.7400096292729899, "precision": 0.9220155968806238, "hmean": 0.8210470085470085, "AP": 0}
```

- PAN++ ResNet18 736 Pre-training + Fine-tune
Pre-training
```
python pretrain.py --dataset pretrain --arch resnet18 --with_rec True --epoch 3 --img_size 640 --short_size 640
```
Training
```
python train_ic15.py --arch resnet18 --with_rec True --epoch 100 --lr 1e-4 --img_size 736 --short_size 736 --pretrain [pretrain checkpoint path]
```
Test
```
python test_ic15.py --resume release_models/pan_pp_ic15_resnet18_736_with_rec_finetune.pth.tar --with_rec True --min_score 0.8 --rec_ignore_score 0.88
```
Result
```
Calculated!{"recall": 0.5483870967741935, "precision": 0.7855172413793103, "hmean": 0.6458746810320386, "AP": 0}
```
Test VOC G
```
python test_ic15.py --resume release_models/pan_pp_ic15_resnet18_736_with_rec_finetune.pth.tar --with_rec True --min_score 0.8 --rec_ignore_score 0.85 --voc g
```
Result VOC G
```
Calculated!{"recall": 0.5589792970630717, "precision": 0.7946611909650924, "hmean": 0.656302996042962, "AP": 0}
```
Test VOC W
```
python test_ic15.py --resume release_models/pan_pp_ic15_resnet18_736_with_rec_finetune.pth.tar --with_rec True --min_score 0.8 --rec_ignore_score 0.73 --unalpha_score 0.995 --voc w
```
Result VOC W
```
Calculated!{"recall": 0.6490129995185363, "precision": 0.8921244209133025, "hmean": 0.7513935340022296, "AP": 0}
```
Test VOC S
```
python test_ic15.py --resume release_models/pan_pp_ic15_resnet18_736_with_rec_finetune.pth.tar --with_rec True --min_score 0.5 --rec_ignore_score 0.66 --unalpha_score 0.995 --edit_dist_score 0.5 --voc s
```
Result VOC S
```
Calculated!{"recall": 0.7024554646124218, "precision": 0.9164572864321608, "hmean": 0.7953120741346416, "AP": 0}
```

- PAN++ ResNet18 896 Pre-training + Fine-tune
Pre-training
```
python pretrain.py --dataset pretrain --arch resnet18 --with_rec True --epoch 3 --img_size 896 --short_size 896
```
Training
```
python train_ic15.py --arch resnet18 --with_rec True --epoch 100 --lr 1e-4 --img_size 896 --short_size 896 --pretrain [pretrain checkpoint path]
```
Test
```
python test_ic15.py --resume release_models/pan_pp_ic15_resnet18_896_with_rec_finetune.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.93
```
Result
```
Calculated!{"recall": 0.5435724602792489, "precision": 0.8222869628550619, "hmean": 0.6544927536231884, "AP": 0}
```
Test VOC G
```
python test_ic15.py --resume release_models/pan_pp_ic15_resnet18_896_with_rec_finetune.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.85 --voc g
```
Result VOC G
```
Calculated!{"recall": 0.5734232065479057, "precision": 0.7856200527704486, "hmean": 0.6629557472863903, "AP": 0}
```
Test VOC W
```
python test_ic15.py --resume release_models/pan_pp_ic15_resnet18_896_with_rec_finetune.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.78 --unalpha_score 0.995 --voc w
```
Result VOC W
```
Calculated!{"recall": 0.6624939817043813, "precision": 0.890038809831824, "hmean": 0.7595914987579354, "AP": 0}
```
Test VOC S
```
python test_ic15.py --resume release_models/pan_pp_ic15_resnet18_896_with_rec_finetune.pth.tar --with_rec True --short_size 896 --min_score 0.5 --rec_ignore_score 0.71 --unalpha_score 0.995 --edit_dist_score 0.5 --voc s
```
Result VOC S
```
Calculated!{"recall": 0.7231584015406837, "precision": 0.9136253041362531, "hmean": 0.8073098629400699, "AP": 0}
```

- PAN++ ResNet50 896 Pre-training + Fine-tune
Pre-training
```
python pretrain.py --dataset pretrain --arch resnet50 --with_rec True --epoch 3 --img_size 896 --short_size 896
```
Training
```
python train_ic15.py --arch resnet50 --with_rec True --epoch 100 --lr 1e-4 --img_size 896 --short_size 896 --pretrain [pretrain checkpoint path]
```
Test
```
python test_ic15.py --arch resnet50 --resume release_models/pan_pp_ic15_resnet50_896_with_rec_finetune.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.9
```
Result
```
Calculated!{"recall": 0.5844968704862783, "precision": 0.8007915567282322, "hmean": 0.675758419148344, "AP": 0}
```
Test VOC G
```
python test_ic15.py --arch resnet50 --resume release_models/pan_pp_ic15_resnet50_896_with_rec_finetune.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.85 --voc g
```
Result VOC G
```
Calculated!{"recall": 0.6008666345690901, "precision": 0.8041237113402062, "hmean": 0.68779278038027, "AP": 0}
```
Test VOC W
```
python test_ic15.py --arch resnet50 --resume release_models/pan_pp_ic15_resnet50_896_with_rec_finetune.pth.tar --with_rec True --short_size 896 --min_score 0.8 --rec_ignore_score 0.76 --unalpha_score 0.995 --voc w
```
Result VOC W
```
Calculated!{"recall": 0.6788637457871931, "precision": 0.8980891719745223, "hmean": 0.7732382780367425, "AP": 0}
```
Test VOC S
```
python test_ic15.py --arch resnet50 --resume release_models/pan_pp_ic15_resnet50_896_with_rec_finetune.pth.tar --with_rec True --short_size 896 --min_score 0.5 --rec_ignore_score 0.7 --unalpha_score 0.995 --edit_dist_score 0.5 --voc s
```
Result VOC S
```
Calculated!{"recall": 0.7424169475204622, "precision": 0.9311594202898551, "hmean": 0.8261451915349585, "AP": 0}
```
