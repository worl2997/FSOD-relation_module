import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fsdet.utils.file_io import PathManager
import pickle
__all__ = ["register_meta_pascal_voc"]

# 커스텀 데이로더 정의
def load_filtered_voc_instances(name: str, dirname: str, split: str, classnames: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"

    name :  "voc_{}_trainval_{}{}_{}shot{}"
    dirname :  root : datasets , dirname: VOC2007 / VOC2012
    split: {}_{}shot_split_{}_trainval  ex) all_1shot_split_1_trainval, novel_1shot_split_1_trainval
    classnames : 정의된 base 및 novel 클래스 목록
    """

    # name -> "voc_{}_trainval_{}{}_{}shot{}
    # few-shot 일경우
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", "vocsplit")
        if "seed" in name:
            shot = name.split("_")[-2].split("shot")[0]  #
            seed = int(name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = name.split("_")[-1].split("shot")[0]
        for cls in classnames:
            with PathManager.open(
                os.path.join(
                    split_dir, "box_{}shot_{}_train.txt".format(shot, cls)
                )
            ) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [
                    fid.split("/")[-1].split(".jpg")[0] for fid in fileids_
                ]
                fileids[cls] = fileids_

    else:
        # base data
        with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
            fileids = np.loadtxt(f, dtype=np.str)


    dicts = []

    # 새로 추가한 glove data
    glove_path = "/home/jacky/바탕화면/FSOD-relation_module/datasets/word2vec_base_split1.pkl"
    with open(glove_path, 'rb') as f:
        voc_glove = pickle.load(f)
        f.close()
    #voc_glove = torch.Tensor(voc_glove)

    if is_shots:
        for cls, fileids_ in fileids.items():

            dicts_ = []
            for fileid in fileids_:
                year = "2012" if "_" in fileid else "2007"
                dirname = os.path.join("datasets", "VOC{}".format(year))
                anno_file = os.path.join(
                    dirname, "Annotations", fileid + ".xml"
                )
                jpeg_file = os.path.join(
                    dirname, "JPEGImages", fileid + ".jpg"
                )

                tree = ET.parse(anno_file)

                for obj in tree.findall("object"):
                    r = {
                        "file_name": jpeg_file,
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                        "voc_glove": voc_glove  # 추가된 부분

                    }
                    cls_ = obj.find("name").text
                    if cls != cls_:
                        continue
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances = [
                        {
                            "category_id": classnames.index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            # "glove": 데이터 로드하는 코드 추가
                        }
                    ]
                    r["annotations"] = instances
                    dicts_.append(r)


            if len(dicts_) > int(shot):
                dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)

    else:
        # base trainig 에서는 Glove 데이터를 뻄
        for fileid in fileids:
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

            tree = ET.parse(anno_file)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
                "voc_glove" : voc_glove # 추가된 부분
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if not (cls in classnames):
                    continue
                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances.append(
                    {
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                )
            r["annotations"] = instances
            dicts.append(r)
    return dicts


def register_meta_pascal_voc(name, metadata, dirname, split, year, keepclasses, sid):
    if keepclasses.startswith("base_novel"):
        thing_classes = metadata["thing_classes"][sid]
    elif keepclasses.startswith("base"):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith("novel"):
        thing_classes = metadata["novel_classes"][sid]

    DatasetCatalog.register(name, lambda: load_filtered_voc_instances(name, dirname, split, thing_classes),)

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        year=year,
        split=split,
        base_classes=metadata["base_classes"][sid],
        novel_classes=metadata["novel_classes"][sid]
    )
