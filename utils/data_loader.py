import torch
import numpy as np
import pandas as pd
from PIL import Image

import os

__all__ = ['OBDataset', 'coco2dataframe']


def coco2dataframe(coco_annot: dict) -> pd.DataFrame:
    """
    COCO形式のデータからimagesとannotationsを取り出しDataFrameに変換

    Parameters
    ----------
    coco_annot: dict
        COCO形式のアノテーションデータ
    
    Returns
    -------
    images, annotations: tuple
        imagesとannotatinosのDataFrame
    """

    images = pd.DataFrame(coco_annot['images'])
    annotations = pd.DataFrame(coco_annot['annotations'])
    return images, annotations

class OBDataset(torch.utils.data.Dataset):
    """
    物体検出向けのカスタムデータセット
    """

    def __init__(self, images: pd.DataFrame, annotations: pd.DataFrame, transforms=None):
        super().__init__()
        self.img_infos = images
        self.annots = annotations
        self.transforms = transforms
    
    def __getitem__(self, idx):
        img_info = self.img_infos.iloc[idx]
        img_id = img_info['id']
        img_fname = img_info['file_name']
        img = Image.open(img_fname)
        target_annots = self.annots.query(f'image_id == {img_id}')
        if len(target_annots) == 0:
            print(' >>> Warning: target_annots is empty (img_id: {})'.format(img_id))
        iscrowd = torch.tensor(target_annots['iscrowd'].to_numpy(), dtype=torch.uint8)
        bbox = torch.tensor(target_annots['bbox'].tolist(), dtype=torch.float)
        labels = torch.tensor(target_annots['category_id'].to_numpy(), dtype=torch.long)
        area = torch.tensor(target_annots['area'].to_numpy(), dtype=torch.float)

        target = {
            'boxes': bbox, 'labels': labels,
            'image_id': torch.tensor([img_id], dtype=torch.long),
            'area': area, 'iscrowd': iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.img_infos)


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
 
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
 
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
 
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    import json
    with open('data/coco_annotations.json', 'r') as fp:
        coco_annot = json.load(fp)
    
    images, annots = coco2dataframe(coco_annot)
    print('images:', images.dtypes)
    print('annots:', annots.dtypes)
    images.to_csv('images.csv')
    annots.to_csv('annots.csv')

    ds = OBDataset(images, annots)

    for i in range(len(ds)):
        img, target = ds[i]
        print(img.size())
        print(target)
        break
