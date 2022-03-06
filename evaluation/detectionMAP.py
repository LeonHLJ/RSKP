import os
import numpy as np
import time
from scipy.signal import savgol_filter
import sys
import scipy.io as sio
import utils.utils as utils


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i]][0]


def filter_segments(segment_predict, videonames, ambilist, factor):
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        vn = videonames[int(segment_predict[i, 0])]
        for a in ambilist:
            if a[0] == vn:
                gt = range(int(round(float(a[2]) * factor)), int(round(float(a[3]) * factor)))
                pd = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(len(set(gt).union(set(pd))))
                if IoU > 0:
                    ind[i] = 1
    s = [segment_predict[i, :] for i in range(np.shape(segment_predict)[0]) if ind[i] == 0]
    return np.array(s)


def getActLoc(vid_preds, frm_preds, vid_lens, act_thresh_cas, annotation_path, args):
    gtsegments = np.load(annotation_path + '/segments.npy')
    gtlabels = np.load(annotation_path + '/labels.npy')
    videoname = np.load(annotation_path + '/videoname.npy')
    videoname = np.array([v.decode('utf-8') for v in videoname])
    subset = np.load(annotation_path + '/subset.npy')
    subset = np.array([s.decode('utf-8') for s in subset])
    classlist = np.load(annotation_path + '/classlist.npy')
    classlist = np.array([c.decode('utf-8') for c in classlist])
    ambilist = annotation_path + '/Ambiguous_test.txt'
    if os.path.isfile(ambilist):
        ambilist = list(open(ambilist, 'r'))
        ambilist = [a.strip('\n').split(' ') for a in ambilist]
    else:
        ambilist = None
    if args.feature_type == 'UNT':
        factor = 10.0 / 4.0
    else:
        factor = 25.0 / 16.0

    # Keep only the test subset annotations
    gts, gtl, vn, vp, fp, vl = [], [], [], [], [], []
    for i, s in enumerate(subset):
        if subset[i] == 'test':
            gts.append(gtsegments[i])
            gtl.append(gtlabels[i])
            vn.append(videoname[i])

    gtsegments = gts
    gtlabels = gtl
    videoname = vn

    # keep ground truth and predictions for instances with temporal annotations
    gtl, vn, vp, fp, vl = [], [], [], [], []
    for i, s in enumerate(gtsegments):
        if len(s):
            gtl.append(gtlabels[i])
            vn.append(videoname[i])
            vp.append(vid_preds[i])
            fp.append(frm_preds[i])
            vl.append(vid_lens[i])
    gtlabels = gtl
    videoname = vn

    # which categories have temporal labels ?
    templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

    # the number index for those categories.
    templabelidx = []
    for t in templabelcategories:
        templabelidx.append(str2ind(t, classlist))

    dataset_segment_predict = []
    class_threshold = args.class_threshold
    for c in templabelidx:
        c_temp = []
        # Get list of all predictions for class c
        for i in range(len(fp)):
            vid_cls_score = vp[i][c]
            vid_cas = fp[i][:, c]
            vid_cls_proposal = []
            if vid_cls_score < class_threshold:
                continue
            for t in range(len(act_thresh_cas)):
                thres = act_thresh_cas[t]
                vid_pred = np.concatenate([np.zeros(1), (vid_cas > thres).astype('float32'), np.zeros(1)], axis=0)
                vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]
                s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
                e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
                for j in range(len(s)):
                    len_proposal = e[j] - s[j]
                    if len_proposal >= 2:
                        inner_score = np.mean(vid_cas[s[j]:e[j] + 1])
                        outer_s = max(0, int(s[j]- 0.25 * len_proposal))
                        outer_e = min(int(vid_cas.shape[0]-1), int(e[j] + 0.25 * len_proposal + 1))
                        outer_temp_list = list(range(outer_s, int(s[j]))) + list(range(int(e[j] + 1), outer_e))
                        if len(outer_temp_list) == 0:
                            outer_score = 0
                        else:
                            outer_score = np.mean(vid_cas[outer_temp_list])
                        c_score = inner_score - outer_score
                        vid_cls_proposal.append([i, s[j], e[j] + 1, c_score])
            pick_idx = NonMaximumSuppression(np.array(vid_cls_proposal), 0.5)
            nms_vid_cls_proposal = [vid_cls_proposal[k] for k in pick_idx]
            c_temp += nms_vid_cls_proposal
        if len(c_temp) > 0:
            c_temp = np.array(c_temp)
            c_temp = c_temp[np.argsort(-c_temp[:,3])]
            if ambilist is not None:
                c_temp = filter_segments(c_temp, videoname, ambilist, factor)  # filtering segment in ambilist
        #import pdb; pdb.set_trace()
        dataset_segment_predict.append(c_temp)
    return dataset_segment_predict


def NonMaximumSuppression(segs, overlapThresh):
    # if there are no boxes, return an empty list
    if len(segs) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if segs.dtype.kind == "i":
        segs = segs.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the segments
    s = segs[:, 1]
    e = segs[:, 2]
    scores = segs[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = e - s + 1
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest coordinates for the start of
        # the segments and the smallest coordinates
        # for the end of the segments
        maxs = np.maximum(s[i], s[idxs[:last]])
        mine = np.minimum(e[i], e[idxs[:last]])

        # compute the length of the overlapping area
        l = np.maximum(0, mine - maxs + 1)
        # compute the ratio of overlap
        overlap = l / area[idxs[:last]]

        # delete segments beyond the threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return pick


def getLocMAP(seg_preds, th, annotation_path, args):
    gtsegments = np.load(annotation_path + '/segments.npy')
    gtlabels = np.load(annotation_path + '/labels.npy')
    videoname = np.load(annotation_path + '/videoname.npy')
    videoname = np.array([v.decode('utf-8') for v in videoname])
    subset = np.load(annotation_path + '/subset.npy')
    subset = np.array([s.decode('utf-8') for s in subset])
    classlist = np.load(annotation_path + '/classlist.npy')
    classlist = np.array([c.decode('utf-8') for c in classlist])
    if args.feature_type == 'UNT': 
        factor = 10.0 / 4.0
    else:
        factor = 25.0 / 16.0

    # Keep only the test subset annotations
    gts, gtl, vn = [], [], []
    for i, s in enumerate(subset):
        if subset[i] == 'test':
            gts.append(gtsegments[i])
            gtl.append(gtlabels[i])
            vn.append(videoname[i])

    gtsegments = gts
    gtlabels = gtl
    videoname = vn

    # keep ground truth and predictions for instances with temporal annotations
    gts, gtl, vn = [], [], []
    for i, s in enumerate(gtsegments):
        if len(s):
            gts.append(gtsegments[i])
            gtl.append(gtlabels[i])
            vn.append(videoname[i])
    gtsegments = gts
    gtlabels = gtl
    videoname = vn

    # which categories have temporal labels ?
    templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

    # the number index for those categories.
    templabelidx = []
    for t in templabelcategories:
        templabelidx.append(str2ind(t, classlist))

    ap = []
    for c in templabelidx:
        segment_predict = seg_preds[c]
        # Sort the list of predictions for class c based on score
        if len(segment_predict) == 0:
            return 0
        segment_predict = segment_predict[np.argsort(-segment_predict[:, 3])]

        # Create gt list
        segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments)) for j in
                      range(len(gtsegments[i])) if str2ind(gtlabels[i][j], classlist) == c]
        gtpos = len(segment_gt)

        # Compare predictions and gt
        tp, fp = [], []
        for i in range(len(segment_predict)):
            matched = False
            best_iou = 0
            for j in range(len(segment_gt)):
                if segment_predict[i][0] == segment_gt[j][0]:
                    gt = range(int(round(segment_gt[j][1] * factor)), int(round(segment_gt[j][2] * factor)))
                    p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                    IoU = float(len(set(gt).intersection(set(p)))) / float(len(set(gt).union(set(p))))
                    if IoU >= th:
                        matched = True
                        if IoU > best_iou:
                            best_iou = IoU
                            best_j = j
            if matched:
                del segment_gt[best_j]
            tp.append(float(matched))
            fp.append(1. - float(matched))
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        if sum(tp) == 0:
            prc = 0.
        else:
            cur_prec = tp_c / (fp_c + tp_c)
            cur_rec = tp_c / gtpos
            prc = _ap_from_pr(cur_prec, cur_rec)
        ap.append(prc)

    if ap:
        return 100 * np.mean(ap)
    else:
        return 0


# Inspired by Pascal VOC evaluation tool.
def _ap_from_pr(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])

    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])

    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])

    return ap


def compute_iou(dur1, dur2):
    # find the each edge of intersect rectangle
    left_line = max(dur1[0], dur2[0])
    right_line = min(dur1[1], dur2[1])

    # judge if there is an intersect
    if left_line >= right_line:
        return 0
    else:
        intersect = right_line - left_line
        union = max(dur1[1], dur2[1]) - min(dur1[0], dur2[0])
        return intersect / union


def getDetectionMAP(vid_preds, frm_preds, vid_lens, annotation_path, args):
    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dmap_list = []
    seg = getActLoc(vid_preds, frm_preds, vid_lens,
                    np.arange(args.start_threshold, args.end_threshold, args.threshold_interval), annotation_path, args)
    for iou in iou_list:
        print('Testing for IoU %f' % iou)
        dmap_list.append(getLocMAP(seg, iou, annotation_path, args))
    return dmap_list, iou_list

