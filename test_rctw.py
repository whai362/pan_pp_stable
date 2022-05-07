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

from dataset import RCTWLoader
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


def write_result_as_txt(image_name, bboxes, path, words=None):
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = path + '%s.txt' % (image_name)
    lines = []
    for i, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        if words is None:
            line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
            lines.append(line)
        elif words[i] is not None:
            line = "%d,%d,%d,%d,%d,%d,%d,%d" % tuple(values) + \
                   ",%s\n" % words[i]
            lines.append(line)
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line)


def correct(word, score, voc=None):
    return word.replace('\"', "")


def test(args):
    n_classes = 2 + args.emb_dim

    data_loader = RCTWLoader(
        split='test',
        short_size=args.short_size,
        read_type=args.read_type,
        report_speed=args.report_speed)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2)

    rec_cfg = None
    if args.with_rec:
        rec_cfg = {
            'voc': data_loader.voc,
            'char2id': data_loader.char2id,
            'id2char': data_loader.id2char,
            'feature_size': args.feature_size
        }

    # Setup Model
    if args.arch == 'resnet18':
        model = models.resnet18(
            pretrained=False,
            num_classes=n_classes,
            rec_cfg=rec_cfg,
            scale=args.scale,
            rec_cscale=args.rec_cscale)
    elif args.arch == 'resnet50':
        model = models.resnet50(
            pretrained=False,
            num_classes=n_classes,
            rec_cfg=rec_cfg,
            scale=args.scale,
            rec_cscale=args.rec_cscale)
    elif args.arch == 'vgg':
        model = models.vgg16_bn(
            pretrained=False,
            num_classes=n_classes,
            use_coordconv=args.use_coordconv,
            scale=args.scale,
            rec_cfg=rec_cfg)
    model = model.cuda()

    print('Total params: %.2fM' % (
            sum(p.numel() for p in model.parameters()) / 1e6))

    if args.resume is not None:
        if os.path.isfile(args.resume):
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
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()

    model = models.fuse_module(model)
    model.eval()

    if args.report_speed:
        fps = AverageMeter(max_len=500)
        time_cost = {
            'backbone_time': AverageMeter(max_len=500),
            'neck_time': AverageMeter(max_len=500),
            'det_head_time': AverageMeter(max_len=500),
            'det_post_time': AverageMeter(max_len=500),
            'rec_time': AverageMeter(max_len=500),
        }

    json_out = {}
    for idx, (org_img, img) in enumerate(test_loader):
        print('Testing %d / %d' % (idx, len(test_loader)), flush=True)

        img = img.cuda()
        org_img = org_img.numpy().astype('uint8')[0]
        args.org_img_size = org_img.shape
        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]

        with torch.no_grad():
            outputs = model(img, args=args)

        if args.report_speed:
            report_speed(outputs, fps, time_cost)

        bboxes = outputs['bboxes']

        if args.with_rec:
            words = outputs['words']
            word_scores = outputs['word_scores']
            words = [correct(word, score) for word, score in
                     zip(words, word_scores)]

        if args.with_rec:
            write_result_as_txt(
                image_name, bboxes, 'outputs/submit_rctw_rec/', words)
        else:
            write_result_as_txt(image_name, bboxes, 'outputs/submit_rctw/')

        json_out[image_name] = {
            'bboxes': np.array(outputs['bboxes']).astype(np.int).tolist(),
            'scores': np.array(outputs['scores']).astype(
                np.float32).tolist()}
        if args.with_rec:
            json_out[image_name]['words'] = words
            json_out[image_name]['word_scores'] = np.array(
                word_scores).astype(np.float32).tolist()
        mmcv.dump(
            json_out,
            './outputs/rctw.json',
            file_format='json',
            ensure_ascii=False)

        if args.vis:
            output_root = 'outputs/vis_rctw/'
            if not os.path.exists(output_root):
                os.makedirs(output_root)
            text_box = org_img.copy()
            for bbox in bboxes:
                cv2.drawContours(
                    text_box,
                    [bbox.reshape(4, 2)],
                    -1, (0, 255, 0), 4)
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
    parser.add_argument('--short_size', nargs='?', type=int, default=736,
                        help='image short size')
    parser.add_argument('--emb_dim', nargs='?', type=int, default=4)
    parser.add_argument('--return_poly_bbox', nargs='?', type=str2bool,
                        default=False)
    parser.add_argument('--with_rec', nargs='?', type=str2bool, default=False)
    parser.add_argument('--feature_size', type=int, nargs='+', default=[8, 32])
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=2.6)
    parser.add_argument('--min_area', nargs='?', type=float, default=260)
    parser.add_argument('--min_score', nargs='?', type=float, default=0.7)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', nargs='?', type=str2bool,
                        default=False)
    parser.add_argument('--json_out', nargs='?', type=str2bool, default=True)
    parser.add_argument('--read_type', nargs='?', type=str, default='pil')
    parser.add_argument('--rec_cscale', nargs='?', type=float, default=4)
    parser.add_argument('--vis', nargs='?', type=str2bool, default=False)
    args = parser.parse_args()
    print(args)

    test(args)
