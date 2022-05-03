# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function, Variable


def discriminative_loss_single(emb, instance, kernel, training_mask, bboxes, feature_dim, delta_v, delta_d, weights):
    training_mask = (training_mask > 0.5).long()
    kernel = (kernel > 0.5).long()
    instance = instance * training_mask
    instance_kernel = (instance * kernel).view(-1)
    instance = instance.view(-1)
    emb = emb.view(feature_dim, -1)

    unique_labels, unique_ids = torch.unique(instance_kernel, sorted=True, return_inverse=True)
    num_instance = unique_labels.size(0)
    if num_instance <= 1:
        return 0

    emb_mean = emb.new_zeros((feature_dim, num_instance), dtype=torch.float32)
    for i, lb in enumerate(unique_labels):
        if lb == 0:
            continue
        ind_k = instance_kernel == lb
        emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)

    l_agg = emb.new_zeros(num_instance, dtype=torch.float32) # bug
    for i, lb in enumerate(unique_labels):
        if lb == 0:
            continue
        ind = instance == lb
        emb_ = emb[:, ind]
        dist = (emb_ - emb_mean[:, i:i+1]).norm(p=2, dim=0)
        dist = F.relu(dist - delta_v) ** 2
        l_agg[i] = torch.mean(torch.log(dist + 1.0))
    l_agg = torch.mean(l_agg[1:])

    if num_instance > 2:
        emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
        emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, feature_dim)
        # print(seg_band)

        mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, feature_dim)
        mask = mask.view(num_instance, num_instance, -1)
        mask[0, :, :] = 0
        mask[:, 0, :] = 0
        mask = mask.view(num_instance * num_instance, -1)
        # print(mask)

        dist = emb_interleave - emb_band
        dist = dist[mask > 0].view(-1, feature_dim).norm(p=2, dim=1)
        dist = F.relu(2 * delta_d - dist) ** 2
        # l_dis = torch.mean(torch.log(dist + 1.0))

        l_dis = [torch.log(dist + 1.0)]
        emb_bg = emb[:, instance == 0].view(feature_dim, -1)
        if emb_bg.size(1) > 100:
            rand_ind = np.random.permutation(emb_bg.size(1))[:100]
            emb_bg = emb_bg[:, rand_ind]
        if emb_bg.size(1) > 0:
            for i, lb in enumerate(unique_labels):
                if lb == 0:
                    continue
                dist = (emb_bg - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
                dist = F.relu(2 * delta_d - dist) ** 2
                l_dis_bg = torch.mean(torch.log(dist + 1.0), 0, keepdim=True)
                l_dis.append(l_dis_bg)
        l_dis = torch.mean(torch.cat(l_dis))
    else:
        l_dis = 0

    l_agg = weights[0] * l_agg
    l_dis = weights[1] * l_dis
    l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
    loss = l_agg + l_dis + l_reg
    return loss


def discriminative_loss(emb, instance, kernel, training_mask, bboxes, feature_dim, delta_v, delta_d, weights,
                        reduce=True):
    loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

    for i in range(loss_batch.size(0)):
        loss_batch[i] = discriminative_loss_single(emb[i], instance[i], kernel[i], training_mask[i], bboxes[i],
                                                   feature_dim, delta_v, delta_d, weights)

    if reduce:
        loss_batch = torch.mean(loss_batch)

    return loss_batch


if __name__ == '__main__':
    pred = [
        [0, 0, 0, 0, 0],
        [0, 0.1, 0.1, 0, 0],
        [0, 0.1, 0.2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1.5, 1.5],
        [0, 0, 0, 1.3, 1.5],
    ]

    label = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]

    feature_dim, delta_v, delta_d, param_var, param_dist, param_reg = 1, 0.01, 1.5, 1.0, 1.0, 0.001

    pred = np.array(pred, dtype='float32')
    pred = pred.reshape(1, 1, 6, 5)
    pred = torch.from_numpy(pred).cuda()

    label = np.array(label)
    label = label.reshape(1, 1, 6, 5)
    label = torch.from_numpy(label).cuda().long()

    # print(pred)
    # print(label)

    # print(pred.shape)
    # print(label.shape)

    discriminative_loss(pred, label, feature_dim, delta_v, delta_d, param_var, param_dist, param_reg)
