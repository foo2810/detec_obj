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

import json
import time
import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    # Dataset
    with open('data/coco_annotations.json', 'r') as fp:
        coco_annot = json.load(fp)

    images, annots = coco2dataframe(coco_annot)
    # bounding_boxが存在しない画像があるっぽいのでそれらを除去
    img_ids = annots['image_id'].tolist()
    images = images.query('id in {}'.format(img_ids))
    p = np.random.permutation(len(images))
    train_ds = OBDataset(images.take(p).iloc[:50], annots)
    test_ds = OBDataset(images.take(p).iloc[50:60], annots)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=2, shuffle=False, num_workers=1,
        collate_fn=collate_fn)

    
    # Model
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 12  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    # Train
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.005)
    n_epochs = 10

    torch.cuda.empty_cache()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    model = model.to(device)

    model.train()
    for epoch in range(n_epochs):

        stime = time.perf_counter()
        train_loss = 0
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

        print('Epoch[{}/{}] time: {:.4f}, loss: {}'.format(epoch+1, n_epochs, time.perf_counter()-stime, train_loss))


    # model.eval()
    # for i, batch in enumerate(test_loader):
    #     images, targets = batch

    #     images = list(image.to(device) for image in images)
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #     with torch.no_grad():
    #         output = model(images)
    #         print(output)