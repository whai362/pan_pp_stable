import os
import os.path as osp
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
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = ['resnet18', 'resnet50', 'resnet101']

model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Convkxk(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1,
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
            kernel_size=3,
            stride=1,
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


class ResNet(nn.Module):

    def __init__(
            self,
            block,
            layers,
            num_classes=6,
            scale=1,
            rec_cfg=None,
            rec_cscale=1):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        hidden_dim = 128

        # reduce dim layer
        self.reduce_layer4 = Convkxk(512 * block.expansion, hidden_dim)
        self.reduce_layer3 = Convkxk(256 * block.expansion, hidden_dim)
        self.reduce_layer2 = Convkxk(128 * block.expansion, hidden_dim)
        self.reduce_layer1 = Convkxk(64 * block.expansion, hidden_dim)

        # FPEM
        self.fpem1 = FPEM(hidden_dim)
        self.fpem2 = FPEM(hidden_dim)

        # detection head
        self.detection_head = DetectionHead(
            hidden_dim * 4,
            hidden_dim,
            num_classes)

        self.recognition_head = None
        if rec_cfg is not None:
            rec_cfg['input_dim'] = 512
            rec_cfg['hidden_dim'] = int(128 * rec_cscale)
            self.recognition_head = RecognitionHead(**rec_cfg)

        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
        h = self.relu1(self.bn1(self.conv1(h)))
        h = self.relu2(self.bn2(self.conv2(h)))
        h = self.relu3(self.bn3(self.conv3(h)))
        h = self.mp(h)

        h = self.layer1(h)
        f1 = self.reduce_layer1(h)

        h = self.layer2(h)
        f2 = self.reduce_layer2(h)

        h = self.layer3(h)
        f3 = self.reduce_layer3(h)

        h = self.layer4(h)
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
            f,
            (imgs.size(2) // self.scale, imgs.size(3) // self.scale))

        if args.report_speed:
            torch.cuda.synchronize()
            time_cost['det_head_time'] = time.time() - start
            start = time.time()

        outputs = {}
        if self.training:
            loss_det = self.detection_head.loss(
                out_det,
                gt_texts,
                gt_kernels,
                training_masks,
                gt_instances,
                gt_bboxes, args=args)
            outputs.update(loss_det)
        else:
            res_det = self.detection_head.get_results(imgs, out_det, args=args)
            outputs.update(res_det)

        if self.recognition_head is not None:
            if self.training:
                x_crops, gt_words = self.recognition_head.extract_feature(
                    f, (imgs.size(2), imgs.size(3)),
                    gt_instances * training_masks, gt_bboxes, gt_words,
                    word_masks)
                if x_crops is not None:
                    out_rec = self.recognition_head(x_crops, gt_words)
                    loss_rec = self.recognition_head.loss(
                        out_rec,
                        gt_words,
                        reduce=False)
                else:
                    loss_rec = {
                        'loss_rec': f.new_full((1,), -1, dtype=torch.float32),
                        'acc_rec': f.new_full((1,), -1, dtype=torch.float32)
                    }
                outputs.update(loss_rec)
            else:
                if args.report_speed:
                    torch.cuda.synchronize()
                    start = time.time()
                if len(res_det['bboxes']) > 0:
                    x_crops = self.recognition_head.extract_feature_test(
                        f,
                        (imgs.size(2), imgs.size(3)),
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
                        torch.cuda.synchronize()
                        start = time.time()

                if args.report_speed:
                    torch.cuda.synchronize()
                    time_cost['rec_time'] = time.time() - start

                outputs['words'] = words
                outputs['word_scores'] = word_scores
                outputs['label'] = ''

        if args.report_speed:
            outputs.update(time_cost)

        return outputs


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
    return model


def load_url(url, model_dir='./pretrained', map_location=None):
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = osp.join(model_dir, filename)
    if not osp.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)
