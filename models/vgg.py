import os
import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from .detection_head import DetectionHead
from .recognition_head import RecognitionHead
import time

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class Convkxk(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super(Convkxk, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPEM(nn.Module):
    def __init__(self, planes):
        super(FPEM, self).__init__()
        self.dwconv3_1 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer3_1 = Convkxk(planes, planes)

        self.dwconv2_1 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3, stride=1,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer2_1 = Convkxk(planes, planes)

        self.dwconv1_1 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer1_1 = Convkxk(planes, planes)

        self.dwconv2_2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer2_2 = Convkxk(planes, planes)

        self.dwconv3_2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer3_2 = Convkxk(planes, planes)

        self.dwconv4_2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer4_2 = Convkxk(planes, planes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):
        f3_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2_ = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_, f2)))
        f1_ = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_, f1)))

        f2_ = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_, f1_)))
        f3_ = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_, f2_)))
        f4_ = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3_)))

        f1 = f1 + f1_
        f2 = f2 + f2_
        f3 = f3 + f3_
        f4 = f4 + f4_

        return f1, f2, f3, f4


class VGG(nn.Module):

    def __init__(self,
                 features,
                 num_classes=6,
                 init_weights=True,
                 scale=1,
                 rec_cfg=None,
                 rec_cscale=1):
        super(VGG, self).__init__()
        self.features = features

        # reduce dim layer
        self.reduce_layer4 = Convkxk(512, 128)
        self.reduce_layer3 = Convkxk(512, 128)
        self.reduce_layer2 = Convkxk(512, 128)
        self.reduce_layer1 = Convkxk(256, 128)

        # FPEM
        self.fpem1 = FPEM(128)
        self.fpem2 = FPEM(128)

        # detection head
        self.detection_head = DetectionHead(
            512, 128, num_classes)

        self.recognition_head = None
        if rec_cfg is not None:
            rec_cfg['input_dim'] = 512
            rec_cfg['hidden_dim'] = int(128 * rec_cscale)
            self.recognition_head = RecognitionHead(**rec_cfg)

        self.scale = scale

        if init_weights:
            self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                gt_words=None,
                word_masks=None,
                args=None):
        if args.report_speed:
            time_cost = {}
            torch.cuda.synchronize()
            start = time.time()

        h = imgs

        h = self.features[:23](h)
        f1 = self.reduce_layer1(h)

        h = self.features[23:33](h)
        f2 = self.reduce_layer2(h)

        h = self.features[33:43](h)
        f3 = self.reduce_layer3(h)

        h = self.features[43:](h)
        f4 = self.reduce_layer4(h)

        if args.report_speed:
            torch.cuda.synchronize()
            time_cost['backbone_time'] = time.time() - start
            start = time.time()

        # FPEM
        f1, f2, f3, f4 = self.fpem1(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem2(f1, f2, f3, f4)

        f2 = self._upsample(f2, f1)
        f3 = self._upsample(f3, f1)
        f4 = self._upsample(f4, f1)
        f = torch.cat((f1, f2, f3, f4), 1)

        if args.report_speed:
            torch.cuda.synchronize()
            time_cost['neck_time'] = time.time() - start
            start = time.time()

        out_det = self.detection_head(
            f, (imgs.size(2) // self.scale, imgs.size(3) // self.scale))

        if args.report_speed:
            torch.cuda.synchronize()
            time_cost['det_head_time'] = time.time() - start
            start = time.time()

        outputs = {}
        if self.training:
            loss_det = self.detection_head.loss(
                out_det, gt_texts, gt_kernels, training_masks, gt_instances,
                gt_bboxes, args=args)
            outputs.update(loss_det)
        else:
            res_det = self.detection_head.get_results(imgs, out_det, args=args)
            outputs.update(res_det)

        if self.recognition_head is not None:
            if self.training:
                x_crops, gt_words = self.recognition_head.extract_feature(
                    f,
                    (imgs.size(2), imgs.size(3)),
                    gt_instances * training_masks,
                    gt_bboxes,
                    gt_words,
                    word_masks)

                if x_crops is not None:
                    out_rec = self.recognition_head(x_crops, gt_words)
                    loss_rec = self.recognition_head.loss(
                        out_rec, gt_words, reduce=False)
                else:
                    loss_rec = {
                        'loss_rec': f.new_full((1,), -1, dtype=torch.float32),
                        'acc_rec': f.new_full((1,), -1, dtype=torch.float32)
                    }

                outputs.update(loss_rec)
            else:
                if args.report_speed:
                    start = time.time()
                if len(res_det['bboxes']) > 0:
                    x_crops, _ = self.recognition_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        f.new_tensor(
                            res_det['label'],
                            dtype=torch.long).unsqueeze(0),
                        bboxes=f.new_tensor(
                            res_det['bboxes_h'],
                            dtype=torch.long),
                        unique_labels=res_det['instances'])
                    words, word_scores = self.recognition_head.forward(
                        x_crops,
                        args=args)
                else:
                    words = []
                    word_scores = []
                    if args.report_speed:
                        start = time.time()

                if args.report_speed:
                    torch.cuda.synchronize()
                    time_cost['rec_time'] = time.time() - start
                    outputs.update(time_cost)

                outputs['words'] = words
                outputs['word_scores'] = word_scores
                outputs['label'] = ''

        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,
          'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
          512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
          512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # debug
    # pretrained = False
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
