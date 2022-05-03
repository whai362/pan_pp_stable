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

from dataset import IC15Loader
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
    if not osp.exists(path):
        os.makedirs(path)

    file_path = path + 'res_%s.txt' % (image_name)
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


def correct_n_g(args, word, score, voc=None):
    EPS = 1e-6
    if len(word) < 3:
        return None
    if score > args.rec_score:
        return word
    if not word.isalpha():
        if score > args.unalpha_score:
            return word
        return None

    if score < args.rec_ignore_score:
        return None

    if voc is not None:
        min_d = 1e10
        matched = ''
        for voc_word in voc:
            d = editdistance.eval(word, voc_word)
            if d < min_d:
                matched = voc_word
                min_d = d
            if min_d == 0:
                break
        if min_d < args.edit_dist:
            return matched
        else:
            return None

    return word


def correct_w(args, word, score, voc=None):
    if len(word) < 3:
        return None

    if score < args.rec_ignore_score:
        return None

    if not word.isalpha():
        if score > args.unalpha_score:
            return word
        return None

    min_d = 1e10
    matched = None
    for voc_word in voc:
        d = editdistance.eval(word, voc_word)
        if d < min_d:
            matched = voc_word
            min_d = d
        if min_d == 0:
            break
    if float(min_d) / len(word) < args.edit_dist_score:
        return matched

    return None


def prefix_score(a, b):
    score = 0
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            score += 1.0 / (i + 1)

    return score


def correct_s(args, word, score, voc=None):
    if len(word) < 3:
        return None

    if score < args.rec_ignore_score:
        return None

    if not word.isalpha():
        if score > args.unalpha_score:
            return word

    min_d = 1e10
    max_prefix_s = 0
    matched = None
    for voc_word in voc:
        d = editdistance.eval(word, voc_word)
        prefix_s = prefix_score(word, voc_word)
        if d < min_d:
            matched = voc_word
            min_d = d
            max_prefix_s = prefix_s
        elif d == min_d and prefix_s > max_prefix_s:
            matched = voc_word
            max_prefix_s = prefix_s

        if min_d == 0:
            break

    if float(min_d) / len(word) < args.edit_dist_score:
        return matched

    return None


def read_voc(voc_path):
    lines = mmcv.list_from_file(voc_path)
    voc = []
    for line in lines:
        if len(line) == 0:
            continue
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        line = line.replace('\r', '').replace('\n', '')
        if line[0] == '#':
            continue
        voc.append(line.lower())

    return voc


def test(args):
    n_classes = 2 + args.emb_dim

    data_loader = IC15Loader(
        split='test', short_size=args.short_size,
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

    voc = None
    correct = correct_n_g
    if args.voc == 'g':
        voc_path = './data/ICDAR2015/Challenge4/GenericVocabulary.txt'
        voc = read_voc(voc_path)
        correct = correct_n_g

    if args.voc == 'w':
        voc_path = './data/ICDAR2015/Challenge4/ch4_test_vocabulary.txt'
        voc = read_voc(voc_path)
        correct = correct_w

    if args.voc == 's':
        path = './data/ICDAR2015/Challenge4/ch4_test_vocabularies_per_image/'
        voc_names = [voc_name for voc_name in mmcv.utils.scandir(path, '.txt')]
        voc = {}
        for voc_name in voc_names:
            voc_path = osp.join(path, voc_name)
            voc[voc_name] = read_voc(voc_path)
        correct = correct_s

    # Setup Model
    if args.arch == 'resnet18':
        model = models.resnet18(
            pretrained=False,
            num_classes=n_classes,
            scale=args.scale,
            rec_cfg=rec_cfg)
    elif args.arch == 'resnet50':
        model = models.resnet50(
            pretrained=True,
            num_classes=n_classes,
            scale=args.scale,
            rec_cfg=rec_cfg)
    elif args.arch == 'vgg':
        model = models.vgg16_bn(
            pretrained=False,
            num_classes=n_classes,
            scale=args.scale,
            rec_cfg=rec_cfg)
    model = model.cuda()

    print('Total params: %.2fM' % (
            sum(p.numel() for p in model.parameters()) / 1e6))

    if args.resume is not None:
        if osp.isfile(args.resume):
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
            print("Loaded checkpoint '{}'".format(args.resume))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

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
        words = [None] * len(bboxes)
        if args.with_rec:
            words = outputs['words']
            word_scores = outputs['word_scores']
            if args.voc == 's':
                voc_name = '_'.join([
                    'voc',
                    image_name + '.txt'])
                words = [correct(args, word, np.mean([score]), voc[voc_name])
                         for word, score in zip(words, word_scores)]
            else:
                words = [correct(args, word, np.mean([score]), voc)
                         for word, score in zip(words, word_scores)]

        if args.with_rec:
            write_result_as_txt(
                image_name, bboxes, 'outputs/submit_ic15_rec/', words)
        else:
            write_result_as_txt(image_name, bboxes, 'outputs/submit_ic15/')

        # visualize results
        if args.vis:
            vis_root = 'outputs/vis_ic15/'
            if not osp.exists(vis_root):
                os.makedirs(vis_root)
            vis_res = org_img.copy()
            for bbox, word in zip(bboxes, words):
                cv2.drawContours(
                    vis_res, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)
                text_pos = np.min(bbox.reshape(4, 2), axis=0)
                cv2.putText(
                    vis_res,
                    word,
                    (text_pos[0], text_pos[1]),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (255, 0, 0), 1)
            cv2.imwrite(osp.join(vis_root, image_name + '.png'),
                        vis_res[:, :, ::-1])

    # package results
    if args.with_rec:
        eval_cmd = 'cd %s;zip -j %s %s/*' % (
            './outputs/', 'submit_ic15_rec.zip', 'submit_ic15_rec')
    else:
        eval_cmd = 'cd %s;zip -j %s %s/*' % (
            './outputs/', 'submit_ic15.zip', 'submit_ic15')
    print(eval_cmd)
    sys.stdout.flush()
    _ = os.popen(eval_cmd)
    _.read()


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
    parser.add_argument('--min_score', nargs='?', type=float, default=0.85)
    parser.add_argument('--beam_size', nargs='?', type=int, default=0)
    parser.add_argument('--voc', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', nargs='?', type=str2bool,
                        default=False)
    parser.add_argument('--read_type', nargs='?', type=str, default='pil')
    parser.add_argument('--vis', nargs='?', type=str2bool, default=False)

    # recognition post-processing hyper-parameters
    parser.add_argument('--with_rec', nargs='?', type=str2bool, default=False)
    parser.add_argument('--rec_score', nargs='?', type=float, default=0.95)
    parser.add_argument('--unalpha_score', nargs='?', type=float, default=0.9)
    parser.add_argument('--rec_ignore_score', nargs='?', type=float,
                        default=0.9)
    parser.add_argument('--edit_dist', nargs='?', type=int, default=2)
    parser.add_argument('--edit_dist_score', nargs='?', type=float,
                        default=1.0 / 3.0)

    args = parser.parse_args()
    print(args)

    test(args)
