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

from dataset import SynthLoader, PretrainDatasetLoader, JointDatasetLoader
import models
from utils import Logger, AverageMeter

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)
EPS = 1e-6


def train(args, train_loader, model, optimizer, epoch, start_iter):
    model.train()

    batch_time = AverageMeter(max_len=500)
    data_time = AverageMeter(max_len=500)
    losses = AverageMeter(max_len=500)
    losses_text = AverageMeter(max_len=500)
    losses_kernels = AverageMeter(max_len=500)
    losses_dis = AverageMeter(max_len=500)
    losses_rec = AverageMeter(max_len=500)
    ious_text = AverageMeter(max_len=500)
    ious_kernel = AverageMeter(max_len=500)
    accs_rec = AverageMeter(max_len=500)

    end = time.time()
    for batch_idx, (imgs, gt_texts, gt_kernels, training_masks, gt_instances,
                    gt_bboxes, gt_words, word_masks) in enumerate(train_loader):
        if batch_idx < start_iter:
            print('skipping iter: %d' % batch_idx)
            sys.stdout.flush()
            continue
        data_time.update(time.time() - end)

        iter_num = epoch * len(train_loader) + batch_idx
        max_iter_num = args.epoch * len(train_loader)
        adjust_learning_rate(args, optimizer, epoch, iter_num, max_iter_num)

        input = {
            'imgs': imgs,
            'gt_texts': gt_texts,
            'gt_kernels': gt_kernels,
            'training_masks': training_masks,
            'gt_instances': gt_instances,
            'gt_bboxes': gt_bboxes,
            'args': args
        }
        if args.with_rec:
            input['gt_words'] = gt_words
            input['word_masks'] = word_masks

        outputs = model(**input)

        loss_text = torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item(), imgs.size(0))

        loss_kernels = torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item(), imgs.size(0))

        if 'loss_dis' in outputs:
            loss_dis = torch.mean(outputs['loss_dis'])
            losses_dis.update(loss_dis.item(), imgs.size(0))
        else:
            loss_dis = 0

        loss = args.loss_w[0] * loss_text + args.loss_w[1] * loss_kernels + \
               args.loss_w[2] * loss_dis

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item(), imgs.size(0))

        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item(), imgs.size(0))

        if 'loss_rec' in outputs:
            loss_rec = outputs['loss_rec']
            valid = loss_rec > -EPS
            if torch.sum(valid) > 0:
                loss_rec = torch.mean(loss_rec[valid])
                losses_rec.update(loss_rec.item(), imgs.size(0))
                loss = loss + loss_rec

                acc_rec = outputs['acc_rec']
                acc_rec = torch.mean(acc_rec[valid])
                accs_rec.update(acc_rec.item(), torch.sum(valid).item())

        losses.update(loss.item(), imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 20 == 0:
            output_log = '({batch}/{size}) LR: {lr:.6f} | Batch: {bt:.3f}s ' \
                         '| Total: {total:.0f}min | ETA: {eta:.0f}min ' \
                         '| Loss: {loss:.3f} ' \
                         '| Loss text/kernel/dis/rec: {loss_text:.3f}/{loss_kernel:.3f}/{loss_dis:.3f}/{loss_rec:.3f} ' \
                         '| IoU text/kernel: {iou_text:.3f}/{iou_kernel:.3f} ' \
                         '| Acc rec: {acc_rec:.3f}'.format(
                batch=batch_idx + 1,
                size=len(train_loader),
                lr=optimizer.param_groups[0]['lr'],
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                loss_text=losses_text.avg,
                loss_kernel=losses_kernels.avg,
                loss_dis=losses_dis.avg,
                loss_rec=losses_rec.avg,
                loss=losses.avg,
                iou_text=ious_text.avg,
                iou_kernel=ious_kernel.avg,
                acc_rec=accs_rec.avg
            )
            print(output_log)
            sys.stdout.flush()

        if batch_idx > 0 and batch_idx % 1000 == 0:
            save_checkpoint({
                'epoch': epoch,
                'iter': batch_idx + 1,
                'state_dict': model.state_dict(),
                'lr': args.lr,
                'optimizer': optimizer.state_dict()},
                checkpoint=args.checkpoint,
                filename='checkpoint.pth.tar')

    return losses.avg, ious_text.avg, ious_kernel.avg, accs_rec.avg


def adjust_learning_rate(args, optimizer, epoch, iter_num, max_iter_num):
    if args.use_polylr:
        lr = args.lr * (1.0 - iter_num * 1.0 / max_iter_num) ** 0.9
    else:
        lr = args.lr
        for i in range(len(args.schedule)):
            if epoch < args.schedule[i]:
                break
            lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = osp.join(checkpoint, filename)
    torch.save(state, filepath)


def main(args):
    if args.checkpoint == '':
        args.checkpoint = 'checkpoints/{dataset}_{arch}_{img_size}'.format(
            dataset=args.dataset,
            arch=args.arch,
            img_size=args.img_size)
        if args.with_rec:
            args.checkpoint += '_with_rec'
            print('with recognition')
    print('checkpoint path: %s' % args.checkpoint)
    if not osp.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    kernel_num = 2
    emb_dim = args.emb_dim
    n_classes = kernel_num + emb_dim
    start_epoch = 0
    start_iter = 0

    if args.dataset == 'synth':
        data_loader = SynthLoader(
            split='train',
            is_transform=True,
            img_size=args.img_size,
            kernel_scale=args.kernel_scale,
            short_size=args.short_size,
            for_rec=args.with_rec,
            read_type=args.read_type)
    elif args.dataset == 'pretrain':
        data_loader = PretrainDatasetLoader(
            split='train',
            is_transform=True,
            img_size=args.img_size,
            kernel_scale=args.kernel_scale,
            short_size=args.short_size,
            for_rec=args.with_rec,
            read_type=args.read_type)
    elif args.dataset == 'joint':
        data_loader = JointDatasetLoader(
            split='train',
            is_transform=True,
            img_size=args.img_size,
            kernel_scale=args.kernel_scale,
            short_size=args.short_size,
            for_rec=args.with_rec,
            read_type=args.read_type)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True)

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
            pretrained=True,
            num_classes=n_classes,
            rec_cfg=rec_cfg)
    elif args.arch == 'resnet50':
        model = models.resnet50(
            pretrained=True,
            num_classes=n_classes,
            rec_cfg=rec_cfg)
    elif args.arch == 'resnet101':
        model = models.resnet101(
            pretrained=True,
            num_classes=n_classes,
            rec_cfg=rec_cfg)
    elif args.arch == 'vgg':
        model = models.vgg16_bn(
            pretrained=True,
            num_classes=n_classes,
            rec_cfg=rec_cfg)

    model = torch.nn.DataParallel(model).cuda()

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        if args.use_adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=0.99,
                weight_decay=5e-4)

    title = 'pretrain'
    logger = None
    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint.')
        assert osp.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iter']
        args.lr = checkpoint['lr']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if osp.isfile(osp.join(args.checkpoint, 'log.txt')):
            logger = Logger(
                osp.join(args.checkpoint, 'log.txt'),
                title=title,
                resume=True)
    if logger is None:
        logger = Logger(osp.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['LR', 'Loss', 'IoU Text/Kernel', 'Acc Rec'])

    for epoch in range(start_epoch, args.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epoch), flush=True)

        train_loss, train_iou_text, train_iou_kernel, train_acc_rec = train(
            args, train_loader, model, optimizer, epoch, start_iter)

        save_checkpoint({
            'epoch': epoch + 1,
            'iter': 0,
            'state_dict': model.state_dict(),
            'lr': args.lr,
            'optimizer': optimizer.state_dict()},
            checkpoint=args.checkpoint,
            filename='checkpoint_%dep.pth.tar' % epoch \
                if epoch > args.epoch - 10 else 'checkpoint.pth.tar')
        logger.append([
            args.lr,
            train_loss,
            '%.3f/%.3f' % (train_iou_text, train_iou_kernel),
            train_acc_rec])

    logger.close()


def str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet18')
    parser.add_argument('--dataset', nargs='?', type=str, default='synth')
    parser.add_argument('--batch', nargs='?', type=int, default=16)
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3)
    parser.add_argument('--use_polylr', nargs='?', type=str2bool, default=True)
    parser.add_argument('--use_adam', nargs='?', type=str2bool, default=True)
    parser.add_argument('--epoch', nargs='?', type=int, default=3)
    parser.add_argument('--schedule', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--img_size', nargs='?', type=int, default=640)
    parser.add_argument('--short_size', nargs='?', type=int, default=640)
    parser.add_argument('--kernel_scale', nargs='?', type=float, default=0.5)
    parser.add_argument('--emb_dim', nargs='?', type=int, default=4)
    parser.add_argument('--with_rec', nargs='?', type=str2bool, default=False)
    parser.add_argument('--feature_size', type=int, nargs='+', default=[8, 32])
    parser.add_argument('--loss_w', type=float, nargs='+',
                        default=[1, 0.5, 0.25])
    parser.add_argument('--loss_dis_w', type=float, nargs='+',
                        default=[1, 1, 0.1])
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    parser.add_argument('--checkpoint', nargs='?', type=str, default='')
    parser.add_argument('--report_speed', nargs='?', type=str2bool,
                        default=False)
    parser.add_argument('--read_type', nargs='?', type=str, default='pil')
    args = parser.parse_args()
    print(args)

    main(args)
