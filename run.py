# Refs
# **Task**
# - https://signate.jp/competitions/403/data
#
# ** COCO format**
# - https://qiita.com/harmegiddo/items/da131ae5bcddbbbde41f
# - https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/cd-coco-overview.html
#
# **Train code**
# - https://qiita.com/ImR0305/items/22c199f85c44890e0ff8
# - https://torch.classcat.com/category/object-detection/

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils.data_loader import coco2dataframe, OBDataset
from utils.metrics import mAP
from utils.transforms import ToTensor, Resize, RandomHorizontalFlip, Compose

import json
import time
import numpy as np

import os
os.environ['http_proxy'] = 'http://ufproxy.b.cii.u-fukui.ac.jp:8080'
os.environ['https_proxy'] = 'http://ufproxy.b.cii.u-fukui.ac.jp:8080'

def collate_fn(batch):
    return tuple(zip(*batch))

transforms_train = Compose([Resize(1/8), RandomHorizontalFlip(0.5)])
transforms_test = Compose([Resize(1/8)])

if __name__ == '__main__':
    # Dataset
    with open('data/coco_annotations.json', 'r') as fp:
        coco_annot = json.load(fp)

    images, annots = coco2dataframe(coco_annot)

    # bounding_boxが存在しない画像があるっぽいのでそれらを除去
    img_ids = annots['image_id'].tolist()
    images = images.query('id in {}'.format(img_ids))

    # 予測対象のJumper SchoolとBreezer Schoolに限定
    # 'bg': 0, 'Jumper School': 1, 'Breezer School': 2,
    annots = annots.query('category_id in [0, 1, 2]')
    img_ids = annots['image_id'].tolist()
    images = images.query('id in {}'.format(img_ids))

    # shuffle
    p = np.random.permutation(len(images))
    N = 200
    train_ds = OBDataset(images.take(p).iloc[:N], annots, transforms=transforms_train)
    test_ds = OBDataset(images.take(p).iloc[N:N+N], annots, transforms=transforms_test)
    print(f'Train: {len(train_ds)}')
    print(f'Test: {len(test_ds)}')

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=2, shuffle=False, num_workers=1,
        collate_fn=collate_fn)

    
    # Model
    num_classes = 3  # 2 class (targets) + background

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # backbone.out_channels = 1280
    # anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    # # anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512),), aspect_ratios=((0.5, 1.0, 2.0),)*5)
    # roi_pooler =torchvision.ops.MultiScaleRoIAlign(
    #     featmap_names=['0','1','2','3'],
    #     output_size=7,
    #     sampling_ratio=2)
    # model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)


    # Train
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    n_epochs = 10

    torch.cuda.empty_cache()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    model = model.to(device)

    for epoch in range(n_epochs):
        stime = time.perf_counter()
        train_loss = 0
        model.train()
        for i, batch in enumerate(train_loader):
            images, targets = batch

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss += loss_value * len(images)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)

        model.eval()
        for i, batch in enumerate(train_loader):
            images, targets = batch

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                output = model(images)
                mAPs = []
                for out, gt in zip(output, targets):
                    boxes_gt = gt['boxes']
                    labels_gt = gt['labels']
                    boxes_pred = out['boxes']
                    labels_pred = out['labels']
                    scores_pred = out['scores']
                    threshold = 0.5
                    interpolated = True
                    mAPs += [mAP(boxes_gt, labels_gt, boxes_pred, labels_pred, scores_pred, num_classes, threshold, interpolated)]
                mAPs = np.array(mAPs)
                mmAP = np.mean(mAPs)

        print('Epoch[{}/{}] time: {:.4f}, loss: {}, test_mean_mAP: {}, test_mAP: {}'.format(epoch+1, n_epochs, time.perf_counter()-stime, train_loss, mmAP, mAPs))
