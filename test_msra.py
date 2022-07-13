import os
import os.path as osp
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mmcv
import collections
import editdistance

from dataset import MSRALoader
import models
from utils import Logger, AverageMeter


def report_speed(outputs, fps, time_cost):
    time_cost_ = 0
    for key in outputs:
        if 'time' in key:
            time_cost_ += outputs[key]
            time_cost[key].update(outputs[key])
            print(key, time_cost[key].avg)

    fps.update(time_cost_)

    print('FPS: %.2f' % (1.0 / fps.avg))


def write_result_as_txt(image_name, bboxes, path):
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = path + '%s.txt' % (image_name)
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d" % values[0]
        for v_id in range(1, len(values)):
            line += ", %d" % values[v_id]
        line += '\n'
        lines.append(line)
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line)


def test(args):
    n_classes = 2 + args.emb_dim

    data_loader = MSRALoader(
        split='test', short_size=args.short_size,
        read_type=args.read_type,
        report_speed=args.report_speed)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2)

    # Setup Model
    if args.arch == 'resnet18':
        model = models.resnet18(
            pretrained=False,
            num_classes=n_classes,
            scale=args.scale)
    elif args.arch == 'resnet50':
        model = models.resnet50(
            pretrained=True,
            num_classes=n_classes,
            scale=args.scale)
    elif args.arch == 'vgg':
        model = models.vgg16_bn(
            pretrained=False,
            num_classes=n_classes,
            scale=args.scale)
    model = model.cuda()

    if args.resume is not None:
        print("Loading model and optimizer from checkpoint '{}'".format(
            args.resume))
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['state_dict']
        d = collections.OrderedDict()
        for key, value in state_dict.items():
            if 'module' in key:
                key = key[7:]
            d[key] = value

        model.load_state_dict(d)

    model = models.fuse_module(model)
    model.eval()
    if args.report_speed:
        fps = AverageMeter(max_len=500)
        time_cost = {
            'backbone_time': AverageMeter(max_len=500),
            'neck_time': AverageMeter(max_len=500),
            'det_head_time': AverageMeter(max_len=500),
            'det_post_time': AverageMeter(max_len=500)
        }

    for idx, (org_img, img) in enumerate(test_loader):
        print('Testing %d / %d' % (idx, len(test_loader)))
        # print(data_loader.img_paths[idx])
        sys.stdout.flush()

        img = img.cuda()
        org_img = org_img.numpy().astype('uint8')[0]
        args.org_img_size = org_img.shape
        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]

        with torch.no_grad():
            outputs = model(img, args=args)

        if args.report_speed:
            report_speed(outputs, fps, time_cost)

        bboxes = outputs['bboxes']
        words = [None] * len(bboxes)
        write_result_as_txt(image_name, bboxes, 'outputs/submit_msra/')

        # visualize results
        if args.vis:
            vis_root = 'outputs/vis_msra/'
            if not osp.exists(vis_root):
                os.makedirs(vis_root)
            vis_res = org_img.copy()
            for bbox, word in zip(bboxes, words):
                cv2.drawContours(
                    vis_res,
                    [bbox.reshape(4, 2)], -1,
                    (0, 255, 0), 2)
                if word is not None:
                    text_pos = np.min(bbox.reshape(-1, 2), axis=0)
                    cv2.putText(
                        text_pos,
                        word,
                        (tl[0], tl[1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 1)
            cv2.imwrite(
                osp.join(vis_root, image_name + '.png'),
                vis_res[:, :, ::-1])


def str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet18')
    parser.add_argument('--scale', nargs='?', type=int, default=4)
    parser.add_argument('--short_size', nargs='?', type=int, default=720,
                        help='image short size')
    parser.add_argument('--emb_dim', nargs='?', type=int, default=4)
    parser.add_argument('--return_poly_bbox', nargs='?', type=str2bool,
                        default=False)
    parser.add_argument('--feature_size', type=int, nargs='+', default=[8, 32])
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=2.6)
    parser.add_argument('--min_area', nargs='?', type=float, default=260)
    parser.add_argument('--min_score', nargs='?', type=float, default=0.86)
    parser.add_argument('--beam_size', nargs='?', type=int, default=0)
    parser.add_argument('--voc', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', nargs='?', type=str2bool,
                        default=False)
    parser.add_argument('--read_type', nargs='?', type=str, default='cv2')
    parser.add_argument('--vis', nargs='?', type=str2bool, default=False)

    # recognition post-processing hyper-parameters
    parser.add_argument('--with_rec', nargs='?', type=str2bool, default=False)
    args = parser.parse_args()
    print(args)
    test(args)
