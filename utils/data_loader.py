import torch
import numpy as np
import pandas as pd
from PIL import Image


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
        img = torch.tensor(np.array(Image.open(img_fname)).transpose([2, 1, 0]) / 255, dtype=torch.float)

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
