import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .losses import dice_loss, ohem_batch, discriminative_loss, iou
from .pa_pyx import pa


class DetectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DetectionHead, self).__init__()

        self.conv1 = nn.Conv2d(
            input_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            hidden_dim,
            output_dim,
            kernel_size=1,
            stride=1,
            padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, output_size):
        return F.upsample(x, size=output_size, mode='bilinear')

    def forward(self, f, output_size):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)
        out = self._upsample(out, output_size)

        return out

    def get_results(self, img, out, args):
        results = {}
        if args.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score = torch.sigmoid(out[:, 0, :, :])

        kernels = out[:, :2, :, :] > 0
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask

        emb = out[:, 2:, :, :]
        emb = emb * text_mask.float()

        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        emb = emb.cpu().numpy()[0].astype(np.float32)

        label = pa(
            kernels,
            emb,
            args.min_kernel_area / (args.scale * args.scale))

        if args.report_speed:
            torch.cuda.synchronize()
            results['det_post_time'] = time.time() - start

        label_num = np.max(label) + 1
        scale = (args.org_img_size[1] * 1.0 / img.shape[-1],
                 args.org_img_size[0] * 1.0 / img.shape[-2])
        label = cv2.resize(
            label,
            (img.shape[-1], img.shape[-2]),
            interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(
            score,
            (img.shape[-1], img.shape[-2]),
            interpolation=cv2.INTER_NEAREST)

        if args.with_rec:
            bboxes_h = np.zeros((1, label_num, 4), dtype=np.int32)
            instances = [[]]

        bboxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))

            if points.shape[0] < args.min_area / (args.scale * args.scale):
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < args.min_score:
                label[ind] = 0
                continue

            if args.with_rec:
                tl = np.min(points, axis=0)
                br = np.max(points, axis=0) + 1
                bboxes_h[0, i] = (tl[0], tl[1], br[0], br[1])
                instances[0].append(i)

            if args.return_poly_bbox:
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                _, contours, _ = cv2.findContours(
                    binary,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale
            else:
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        results['bboxes'] = bboxes
        results['scores'] = scores
        if args.with_rec:
            results['label'] = label
            results['bboxes_h'] = bboxes_h
            results['instances'] = instances

        return results

    def loss(self,
             out,
             gt_texts,
             gt_kernels,
             training_masks,
             gt_instances,
             gt_bboxes,
             args):
        texts = out[:, 0, :, :]
        kernels = out[:, 1:2, :, :]
        embs = out[:, 2:, :, :]

        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = dice_loss(texts, gt_texts, selected_masks, reduce=False)
        iou_text = iou(
            (texts > 0).long(),
            gt_texts,
            training_masks,
            reduce=False)
        losses = {'loss_text': loss_text, 'iou_text': iou_text}

        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = dice_loss(
                kernel_i,
                gt_kernel_i,
                selected_masks,
                reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou(
            (kernels[:, -1, :, :] > 0).long(),
            gt_kernels[:, -1, :, :],
            training_masks * gt_texts,
            reduce=False)
        losses['loss_kernels'] = loss_kernels
        losses['iou_kernel'] = iou_kernel

        loss_dis = discriminative_loss(
            embs,
            gt_instances,
            gt_kernels[:, -1, :, :],
            training_masks,
            gt_bboxes,
            feature_dim=args.emb_dim,
            delta_v=0.5,
            delta_d=1.5,
            weights=args.loss_dis_w,
            reduce=False)
        losses['loss_dis'] = loss_dis

        return losses
