# Refs: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py

import torch

from typing import Tuple

def _boxes_intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> bool:
    """
    二つのbounding_boxが重なっているかを判定

    Parameters
    ----------
    box_a, box_b: torch.Tensor

    Returns
    -------
    flg: bool
    """

    if box_a[0] > box_b[2]:
        return False  # box_a is right of box_b
    if box_b[0] > box_a[2]:
        return False  # box_a is left of box_b
    if box_a[3] < box_b[1]:
        return False  # box_a is above box_b
    if box_a[1] > box_b[3]:
        return False  # box_a is below box_b

    return True

def _get_intersect_area(box_a: torch.Tensor, box_b: torch.Tensor):
    """
    重なっている面積を計算(積集合)

    Parameters
    ----------
    box_a, box_b: torch.Tensor

    Returns
    -------
    area:
    """
    assert _boxes_intersect(box_a, box_b)

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    # intersection area
    # return (x_b - x_a + 1) * (y_b - y_a + 1)
    return (x_b - x_a) * (y_b - y_a)

def _get_union_area(box_a: torch.Tensor, box_b: torch.Tensor):
    """
    box_aとbox_bを合わせた面積を計算(和集合)

    Parameters
    ----------
    box_a, box_b: torch.Tensor

    Returns
    -------
    area:
    """
    assert _boxes_intersect(box_a, box_b)

    s_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    s_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    area = s_a + s_b - _get_intersect_area(box_a, box_b)

    return area

def iou(box_gt: torch.Tensor, box_pred: torch.Tensor):
    """
    IOUを計算

    Parameters
    ----------
    box_gt:
        正解bounding_box
    
    box_pred:
        予測bounding_box
    
    Returns
    -------
    iou_metrics:
        IOU
    """

    if _boxes_intersect(box_gt, box_pred):
        return _get_intersect_area(box_gt, box_pred) / _get_union_area(box_gt, box_pred)
    else:
        return 0

def _ap_per_class_base(boxes_gt: torch.Tensor, labels_gt: torch.Tensor, boxes_pred: torch.Tensor, labels_pred: torch.Tensor, scores_pred: torch.Tensor, iou_threshold: float=0.5) -> Tuple[torch.Tensor]:
    """
    一つのクラスに対するPrecisionとRecallを計算

    Parameters
    ----------
    boxes_gt: torch.Tensor
        正解bounding_box (2 dims)
        バッチ内のすべてのbounding_boxを指定(クラスによらない)

    labels_gt: torch.Tensor
        正解ラベル (1 dims)
        バッチ内のすべてのlabels指定(クラスによらない)

    boxes_pred: torch.Tensor
        予測bounding_box (2 dims)
        一種類の予測クラスのみの予測bounding_boxを指定する

    labels_pred: torch.Tensor
        予測ラベル (1 dims)
        一種類の予測クラスのみのlabelsを指定する
        (便宜的にテンソルを指定するようにしている)
    
    scores_pred: torch.Tensor
        予測信頼度 (1 dims)
        一種類の予測クラスのみのscoresを指定する
    
    iou_threshold: float
        iou threshold

    Returns
    -------
    precisin, recall, is_correct: Tuple[torch.Tensor]
        予測信頼度順で並べられたPrecisionとRecallと正否
        (補完実行前)
    """

    if len(labels_pred) == 0:
        # print(' >> Hit: first checkpoint')
        return torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float), torch.ones(1, dtype=torch.bool)

    assert torch.all(labels_pred == labels_pred[0])

    p = torch.argsort(scores_pred)
    mask = labels_gt == labels_pred[0]
    is_correct = torch.zeros(len(labels_pred), dtype=torch.bool)
    for bb_gt in boxes_gt[mask]:
        pre_indice = None
        for i, bb_pred in enumerate(boxes_pred):
            iou_t = iou(bb_gt, bb_pred)
            assert iou_t >= 0 and iou_t <= 1, 'IoU must be in [0, 1].'
            if pre_indice is None and iou_t >= iou_threshold:
                is_correct[i] = True
                pre_indice = i
            elif pre_indice is not None and iou_t >= iou_threshold:
                is_correct[pre_indice] = False
                break
            # else:
            #     # These cases do not happen.

    # ious = []
    # for bb_pred in boxes_pred:
    #     max_iou = 0
    #     for bb_gt in boxes_gt[mask]:
    #         out_t = iou(bb_gt, bb_pred)
    #         if max_iou < out_t:
    #             max_iou = out_t
    #     ious += [max_iou]
    # ious = torch.tensor(ious, dtype=torch.float)
    # is_correct = ious > iou_threshold

    corrects = (is_correct).long()

    masks = [torch.tensor([j for j in range(i+1)], dtype=torch.long) for i in range(len(labels_pred))]
    n_gts = (labels_gt == labels_pred[0]).long().sum().item()
    # print(' > n_groud_truths: {}'.format(n_gts))

    precisions = []
    recalls = []
    for i, mask in enumerate(masks):
        assert len(mask) == i+1
        precisions += [torch.sum(corrects[p][mask]).item() / len(mask)]

        # ここの処理はどうすればいいのか？
        if n_gts == 0:
            # print((' >> Hit: second checkpoint'))
            recalls += [0]
        else:
            t = torch.sum(corrects[p][mask]).item()
            # print(' > n_corrects: {}'.format(t))
            recalls += [t / n_gts]

    precisions = torch.tensor(precisions)
    recalls = torch.tensor(recalls)
    assert len(precisions) == len(labels_pred)
    assert len(recalls) == len(labels_pred)
    assert recalls[-1] <= 1

    return precisions, recalls, is_correct[p]


def ap_per_class(boxes_gt: torch.Tensor, labels_gt: torch.Tensor, boxes_pred: torch.Tensor, labels_pred: torch.Tensor, scores_pred: torch.Tensor, iou_threshold: float=0.5) -> float:
    """
    一つのクラスに対するAverage Precisionを計算
    not 11-point interpolated average precision

    Parameters
    ----------
    boxes_gt: torch.Tensor
        正解bounding_box (2 dims)
        バッチ内のすべてのbounding_boxを指定(クラスによらない)

    labels_gt: torch.Tensor
        正解ラベル (1 dims)
        バッチ内のすべてのlabels指定(クラスによらない)

    boxes_pred: torch.Tensor
        予測bounding_box (2 dims)
        一種類の予測クラスのみの予測bounding_boxを指定する

    labels_pred: torch.Tensor
        予測ラベル (1 dims)
        一種類の予測クラスのみのlabelsを指定する
        (便宜的にテンソルを指定するようにしている)
    
    scores_pred: torch.Tensor
        予測信頼度 (1 dims)
        一種類の予測クラスのみのscoresを指定する

    iou_threshold: float
        iou threshold

    Returns
    -------
    ap: float
        Average Precision
    """

    precisions, _, is_correct = _ap_per_class_base(boxes_gt, labels_gt, boxes_pred, labels_pred, scores_pred, iou_threshold)
    # print('> precisinos: {}'.format(precisions))
    # print('> recalls: {}'.format(_))
    # print('> is_correct: {}'.format(is_correct))
    # print('='*50)
    assert len(precisions) != 0
    ap = (precisions[is_correct].sum() / len(precisions)).item()
    return ap

def ap_interpolated_per_class(boxes_gt: torch.Tensor, labels_gt: torch.Tensor, boxes_pred: torch.Tensor, labels_pred: torch.Tensor, scores_pred: torch.Tensor, iou_threshold: float=0.5) -> float:
    """
    一つのクラスに対するAverage Precisionを計算
    11-point interpolated average precision

    Parameters
    ----------
    boxes_gt: torch.Tensor
        正解bounding_box (2 dims)
        バッチ内のすべてのbounding_boxを指定(クラスによらない)

    labels_gt: torch.Tensor
        正解ラベル (1 dims)
        バッチ内のすべてのlabels指定(クラスによらない)

    boxes_pred: torch.Tensor
        予測bounding_box (2 dims)
        一種類の予測クラスのみの予測bounding_boxを指定する

    labels_pred: torch.Tensor
        予測ラベル (1 dims)
        一種類の予測クラスのみのlabelsを指定する
        (便宜的にテンソルを指定するようにしている)
    
    scores_pred: torch.Tensor
        予測信頼度 (1 dims)
        一種類の予測クラスのみのscoresを指定する

    iou_threshold: float
        iou threshold

    Returns
    -------
    ap_interpolated: float
        Interpolated Average Precision
    """

    precisions, recalls, _= _ap_per_class_base(boxes_gt, labels_gt, boxes_pred, labels_pred, scores_pred, iou_threshold)

    # Interpolate
    precisions_interpolated = precisions.clone()
    if torch.all(recalls == recalls[0]):
        precisions_interpolated = torch.zeros(len(precisions), dtype=torch.float)
    else:
        max_prec = precisions[-1]
        for i in range(len(recalls)-1, -1, -1):
            precisions_interpolated[i] = torch.maximum(precisions[i], max_prec)
            max_prec = precisions_interpolated[i]
    
    eleven_points_prec = []
    for i, r in enumerate(torch.arange(0, 1.1, 0.1)):
        for j, rec in enumerate(recalls):
            if r <= rec:
                eleven_points_prec += [precisions_interpolated[j]]
                break
            if j == len(recalls)-1:
                eleven_points_prec += [0]
    eleven_points_prec = torch.tensor(eleven_points_prec)
    ap_interpolated = eleven_points_prec.mean().item()

    return ap_interpolated

def AP(boxes_gt: torch.Tensor, labels_gt: torch.Tensor, boxes_pred: torch.Tensor, labels_pred: torch.Tensor, scores_pred: torch.Tensor, num_classes: int, iou_threshold: float=0.5, interpolate: bool=False) -> torch.Tensor:
    """
    全クラスに対するAverage Precisionを計算

    Parameters
    ----------
    boxes_gt: torch.Tensor
        正解bounding_box (2 dims)
        バッチ内のすべてのbounding_boxを指定(クラスによらない)

    labels_gt: torch.Tensor
        正解ラベル (1 dims)
        バッチ内のすべてのlabels指定(クラスによらない)

    boxes_pred: torch.Tensor
        予測bounding_box (2 dims)

    labels_pred: torch.Tensor
        予測ラベル (1 dims)
    
    scores_pred: torch.Tensor
        予測信頼度 (1 dims)
    
    iou_threshold: float
        iou threshold
    
    interpolate: bool
        11点補完をするかどうか

    Returns
    -------
    aps: torch.Tensor
        Average Precisions for every classes
    """

    aps = []
    for c in range(num_classes):
        mask = labels_pred == c
        if interpolate:
            ap_c = ap_interpolated_per_class(boxes_gt, labels_gt, boxes_pred[mask], labels_pred[mask], scores_pred[mask], iou_threshold)
        else:
            ap_c = ap_per_class(boxes_gt, labels_gt, boxes_pred[mask], labels_pred[mask], scores_pred[mask], iou_threshold)
        aps += [ap_c]
    
    assert len(aps) == num_classes

    aps = torch.tensor(aps, dtype=torch.float)

    return aps

def mAP(boxes_gt: torch.Tensor, labels_gt: torch.Tensor, boxes_pred: torch.Tensor, labels_pred: torch.Tensor, scores_pred: torch.Tensor, num_classes: int, iou_threshold: float=0.5, interpolate: bool=False) -> float:
    """
    mean Average Precisionを計算

    Parameters
    ----------
    boxes_gt: torch.Tensor
        正解bounding_box (2 dims)
        バッチ内のすべてのbounding_boxを指定(クラスによらない)

    labels_gt: torch.Tensor
        正解ラベル (1 dims)
        バッチ内のすべてのlabels指定(クラスによらない)

    boxes_pred: torch.Tensor
        予測bounding_box (2 dims)

    labels_pred: torch.Tensor
        予測ラベル (1 dims)
    
    scores_pred: torch.Tensor
        予測信頼度 (1 dims)
    
    iou_threshold: float
        iou threshold
    
    interpolate: bool
        11点補完をするかどうか

    Returns
    -------
    map: float
        mean Average Precision
    """

    aps = AP(boxes_gt, labels_gt, boxes_pred, labels_pred, scores_pred, num_classes, iou_threshold, interpolate)
    return torch.mean(aps)
