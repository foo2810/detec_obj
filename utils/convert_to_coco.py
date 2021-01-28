import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path

from typing import List



def load(annot_dir: Path) -> List[dict]:
    """
    指定したディレクトリ下のアノテーションファイルをすべてロードする

    Parameters
    ----------
    annot_dir: Path
        アノテーションファイルのあるディレクトリ
    
    Returns
    -------
    org_data: List[dict]
        アノテーション情報のリスト
    """

    cache_path = Path('./data/org_annot.dat')
    if not cache_path.exists():
        org_data = []
        for path in annot_dir.glob('*.json'):
            # print(path)
            with path.open('r') as fp:
                org_data += [json.load(fp)]
        # org_data = pd.DataFrame(org_data)

        with cache_path.open('wb') as fp:
            pickle.dump(org_data, fp, protocol=4)
    else:
        with cache_path.open('rb') as fp:
            org_data = pickle.load(fp)
    
    return org_data

def gen_coco_info() -> dict:
    """
    一つのアノテーションデータからCOCOフォーマットのinfoを生成

    Parameters
    ----------
    annot: dict
        一つのjsonファイルに格納されたアノテーションデータ
    
    Returns
    -------
    info: dict
        COCOフォーマットのinfo
    """

    info = {}

    return info

def gen_coco_licenses(annots: List[dict]) -> List[dict]:
    """
    複数のアノテーションデータからCOCOフォーマットのlicensesを生成

    Parameters
    ----------
    annots: List[dict]
        複数のjsonファイルに格納されたアノテーションデータ
    
    Returns
    -------
    licenses: list
        COCOフォーマットのlicenses
    """

    licenses = []
    return licenses

def gen_coco_images(annots: List[dict], file_names: List[str]) -> List[dict]:
    """
    複数のアノテーションデータからCOCOフォーマットのimagesを生成

    Parameters
    ----------
    annots: List[dict]
        複数のjsonファイルに格納されたアノテーションデータ
    
    file_names: List[str]
        annotsに対応した画像ファイルリスト
    
    Returns
    -------
    images: List[dict]
        COCOフォーマットのimages
    """

    images = []
    for i, annot, fname in zip(list(range(len(annots))), annots, file_names):
        img_annot = {
            # 'license': ..., 
            'file_name': fname, 
            # 'coco_url': ...,
            # 'height': ...,
            # 'width': ...,
            # 'date_captured': ...,
            # 'flickr_url': ...,
            'id': i,
        }
        images += [img_annot]
    return images

def gen_coco_annotations(annots: List[dict], cat2id: dict) -> List[dict]:
    """
    複数のアノテーションデータからCOCOフォーマットのannotationsを生成

    Parameters
    ----------
    annots: List[dict]
        複数のjsonファイルに格納されたアノテーションデータ
    
    cat2id: dict
        カテゴリ名とカテゴリIDの対応表
    
    Returns
    -------
    annotations: dict
        COCOフォーマットのannotations
    """

    annotations = []

    for i, annot in zip(list(range(len(annots))), annots):
        j = 0
        for class_name in annot['labels'].keys():
            for bb in annot['labels'][class_name]:
                ann = {
                    # 'segmentation': ...,
                    'area': (bb[2]-bb[0])*(bb[3]-bb[1]),
                    'iscrowd': 0,
                    'image_id': i,
                    'bbox': bb,
                    'category_id': cat2id[class_name],
                    'id': f'img{i}_{j}'
                }
                annotations += [ann]
                j += 1

    return annotations
        
def gen_coco_categories(cat2id: dict) -> list:
    """
    COCOフォーマットのcategoriesを生成

    Parameters
    ----------
    cat2id: dict
        カテゴリ名とカテゴリIDの対応表
    
    Returns
    -------
    categories: list
        COCOフォーマットのcategoriesを生成
    """

    categories = []
    for class_name in cat2id.keys():
        categories += [
            {
                'supercategory': 'sea',
                'id': cat2id[class_name],
                'name': class_name,
            }
        ]

    return categories
        

def convert_to_coco(annots: List[dict], file_names: List[str], cat2id: dict) -> dict:
    """
    オリジナルのアノテーションデータをCOCOフォーマットに変換

    Parameters
    ----------
    annots: List[dict]
        複数のjsonファイルに格納されたアノテーションデータ
    
    file_names: List[str]
        annotsに対応した画像ファイルリスト

    cat2id: dict
        カテゴリ名とカテゴリIDの対応表
    
    Returns
    -------
    coco: dict 
        COCOフォーマットに変換したアノテーションデータ
    """
    coco = {
        'info': gen_coco_info(),
        'licenses': gen_coco_licenses(annots),
        'images': gen_coco_images(annots, file_names),
        'annotations': gen_coco_annotations(annots, cat2id),
        'categories': gen_coco_categories(cat2id),
    }

    return coco


if __name__ == '__main__':
    annot_dir = Path('./data/train_annotations/')
    # cat2id = {
    #     'Jumper School': 0, 'Breezer School': 1,
    #     'Dolphin': 2, 'Bird': 3, 'Object': 4, 'Cloud': 5, 'Ripple': 6, 'Smooth Surface': 7, 'Wake': 8,
    #     'Each Fish': 9, 'w': 10,
    # }
    cat2id = {
        'bg': 0,
        'Jumper School': 1, 'Breezer School': 2,
        'Dolphin': 3, 'Bird': 4, 'Object': 5, 'Cloud': 6, 'Ripple': 7, 'Smooth Surface': 8, 'Wake': 9,
        'Each Fish': 10, 'w': 11,
    }
    file_names = ['data/train_images/{}.jpg'.format(path.stem) for path in annot_dir.glob('*.json')]

    org_data = load(annot_dir)

    coco = convert_to_coco(org_data, file_names, cat2id)

    with open('coco_annotations.json', 'w') as fp:
        json.dump(coco, fp)

