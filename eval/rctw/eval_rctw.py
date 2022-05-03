import os
import argparse
import numpy as np

import mmcv
from collections import defaultdict
import Polygon as plg
import editdistance
import zipfile
import json

data_root = '/mnt/lustre/wangwenhai/workspace/pan_ppv2.pytorch/data/RCTW-17/'
dictmap_to_lower = mmcv.load(data_root + 'dictmap_to_lower.json')


def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups


def get_union(pa, pb):
    pa_area = pa.area()
    pb_area = pb.area()
    return pa_area + pb_area - get_intersection(pa, pb)


def get_intersection(pa, pb):
    pInt = pa & pb
    if len(pInt) == 0:
        return 0
    else:
        return pInt.area()


def cat_best_hmean(gt, predictions, thresholds):
    num_gts = len([g for g in gt if g['ignore'] == False])
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([b['bbox'] for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_trans = {k: np.array([b['trans'] for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_ignored = {k: np.array([b['ignore'] for b in boxes])
                        for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    ed = np.zeros((nd, len(thresholds)))
    gt_ed = np.zeros((nd, len(thresholds)))
    sum_gt_ed = 0
    for g in gt:
        if g['ignore']:
            continue
        sum_gt_ed += len(g['trans'])

    for i, p in enumerate(predictions):
        pred_polygon = plg.Polygon(np.array(p['bbox']).reshape(-1, 2))
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_ignored = image_gt_ignored[p['name']]
            gt_checked = image_gt_checked[p['name']]
            gt_trans = image_gt_trans[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            ovmax = 0
            jmax = 0
            for j, gt_box in enumerate(gt_boxes):
                gt_polygon = plg.Polygon(np.array(gt_box).reshape(-1, 2))
                union = get_union(pred_polygon, gt_polygon)
                inter = get_intersection(pred_polygon, gt_polygon)
                overlap = inter / (union + 1e-6)
                if overlap > ovmax:
                    ovmax = overlap
                    jmax = j

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    if gt_ignored[jmax]:
                        tp[i, t] = 0.
                        ed[i, t] = 0
                    else:
                        tp[i, t] = 1.
                        ed[i, t] = editdistance.eval(p['trans'], gt_trans[jmax])
                        gt_ed[i, t] = len(gt_trans[jmax])
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.
                    ed[i, t] = len(p['trans'])
            else:
                fp[i, t] = 1.
                ed[i, t] = len(p['trans'])

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    ed = np.cumsum(ed, axis=0)
    gt_ed = np.cumsum(gt_ed, axis=0)

    ed = (ed - gt_ed + sum_gt_ed) / len(image_gts)

    recalls = tp / float(num_gts)
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    fmeasures = 2 * precisions * recalls / (precisions + recalls + 1e-6)

    best_i = np.argmax(fmeasures)
    print('[Best F-Measure] p: {:.2f}, r: {:.2f}, f: {:.2f}, 1-ned: {:.2f}, best_score_th: {:.3f}'.format(
        float(precisions[best_i]) * 100, float(recalls[best_i]) * 100, float(fmeasures[best_i]) * 100,
        float(ed[best_i]), predictions[best_i]['score']))

    best_i = np.argmin(ed)
    print('[Best 1-NED]     p: {:.2f}, r: {:.2f}, f: {:.2f}, 1-ned: {:.2f}, best_score_th: {:.3f}'.format(
        float(precisions[best_i]) * 100, float(recalls[best_i]) * 100, float(fmeasures[best_i]) * 100,
        float(ed[best_i]), predictions[best_i]['score']))


def trans_pred_format(img_name, pred):
    bdd = []
    bboxes = pred['bboxes']
    scores = pred['scores']
    words = pred['words']
    for i in range(len(bboxes)):
        bdd_i = {
            'category': 'text',
            'timestamp': 1000,
            'name': img_name,
            'bbox': np.array(bboxes[i]).reshape(-1).tolist(),
            'score': scores[i],
            'trans': words[i]
        }
        bdd.append(bdd_i)
    return bdd


def trans_gt_format(gt_path):
    bdd = []

    img_name = gt_path.split('/')[-1].replace('.txt', '')
    lines = mmcv.list_from_file(gt_path)
    for line in lines:
        line = line.replace('\xef\xbb\xbf', '')
        line = line.replace('\ufeff', '')
        gt = line.split(',')
        gt[-1] = gt[-1].replace("\"", "")
        bbox = [np.int(gt[i]) for i in range(8)]

        word = ''
        for c in gt[9]:
            if c in dictmap_to_lower:
                c = dictmap_to_lower[c]
            word += c

        difficult = int(gt[8])

        bdd_i = {
            'category': 'text',
            'timestamp': 1000,
            'name': img_name,
            'bbox': bbox,
            'score': 1,
            'ignore': difficult == 1,
            'trans': word
        }
        bdd.append(bdd_i)

    return bdd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', nargs='?', type=str)
    args = parser.parse_args()

    if args.pred is None:
        raise NotImplementedError('prediction file is required.')

    gt_list = []
    val_img_list = mmcv.list_from_file(data_root + 'val_list.txt')
    for img_name in val_img_list:
        gt_path = data_root + 'train/' + img_name.replace('.jpg', '.txt')
        gt_list.extend(trans_gt_format(gt_path))
    cat_gt = group_by_key(gt_list, 'category')

    preds = mmcv.load(args.pred)
    pred_list = []
    for img_name, pred in preds.items():
        pred_list.extend(trans_pred_format(img_name, pred))
    cat_pred = group_by_key(pred_list, 'category')

    thresholds = [0.5]
    cat_best_hmean(cat_gt['text'], cat_pred['text'], thresholds)


if __name__ == '__main__':
    main()
