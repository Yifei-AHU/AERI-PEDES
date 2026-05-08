import os.path as op
import random
from typing import List
from utils.iotools import read_json, write_json
from .bases import BaseDataset
import json
import itertools
from itertools import cycle, islice

class AERIPEDES(BaseDataset):
    """
    AERI-PEDES Dataset (with Ground Images)
    ---------------------------------------

    Training samples: (pid, image_id, aerial_img_path, ground_img_path, caption)
    Testing samples:  (caption, aerial_img_path)

    JSON annotation example:
    {
        "id": 1,
        "split": "train",
        "captions": [...],
        "file_paths": [...],           # aerial images
        "ground_file_paths": [...]     # ground images
    }
    """

    dataset_name = 'AERI-PEDES'

    def __init__(self, root='', verbose=True, force_rebuild=False):
        super(AERIPEDES, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_name)
        self.img_dir = self.dataset_dir

        self.train_anno_path = op.join(self.dataset_dir, 'train_caption.json')
        self.val_anno_path = op.join(self.dataset_dir, 'test_caption.json')

        self._check_before_run()

        # split by "split" field
        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.train_anno_path,self.val_anno_path)

        # build dataset lists
        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos, training=False)
        self.val, self.val_id_container = self._process_anno(self.val_annos, training=False)

        if verbose:
            self.logger.info("=> AERI-PEDES dataset loaded successfully.")
            self.show_dataset_info()
        
    def read_json(self):
        train_aerial_count = 0
        train_ground_count = 0
        train_text_count = 0
        test_aerial_count = 0
        test_text_count = 0
        with open(self.anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            split = item.get("split", "").lower()
            captions = item.get("captions", [])
            file_paths = item.get("file_paths", [])
            ground_file_paths = item.get("ground_file_paths", [])

            if split == "train":
                train_aerial_count += len(file_paths)
                train_ground_count += len(ground_file_paths)
                train_text_count += len(captions)

            elif split == "test":
                test_aerial_count += len(file_paths)
                test_text_count += len(captions)

        return train_aerial_count, train_text_count, test_aerial_count, test_text_count

    def _split_anno(self, train_anno_path, val_anno_path):
        train_annos = read_json(train_anno_path)
        val_annos = read_json(val_anno_path)
        train, test, val = [], [], []
        for a in train_annos:
            if a['split'] == 'train':
                train.append(a)
        for b in val_annos:
            if b['split'] == 'test':
                test.append(b)
                val.append(b)
        return train, test, val

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()

        # ==========================================================
        # TRAINING PHASE — with cache for reproducibility
        # ==========================================================
        if training:
            dataset = []

            image_id = 0
            for anno in annos:
                pid = int(anno['pid']) - 1
                pid_container.add(pid)

                # 航空图 & 地面图路径
                aerial_imgs = op.join(self.img_dir, anno['aerial_img'][11:])
                ground_imgs = op.join(self.img_dir, anno['ground_img'][11:])
                captions = anno['caption']
                dataset.append((pid, aerial_imgs, ground_imgs, captions))

            return dataset, pid_container

        # ==========================================================
        # TEST / VALIDATION — Text + Aerial Only
        # ==========================================================
        else:
            img_paths, image_pids, captions, caption_pids = [], [], [], []
            ground_img_paths, ground_image_pids = [], []
            for anno in annos:
                pid = int(anno['pid'])
                pid_container.add(pid)
                aerial_imgs = [op.join(self.img_dir, p) for p in anno['file_paths']]
                ground_imgs = [op.join(self.img_dir, p) for p in anno['ground_file_paths']]
                captions_list = anno['captions']

                if not aerial_imgs or not captions_list:
                    continue

                # 所有文本
                for cap in captions_list:
                    captions.append(cap)
                    caption_pids.append(pid)

                # 所有航空图像
                for aerial_img in aerial_imgs:
                    img_paths.append(aerial_img)
                    image_pids.append(pid)
                
                for ground_img in ground_imgs:
                    ground_img_paths.append(ground_img)
                    ground_image_pids.append(pid)

            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions,
                "ground_image_pids": ground_image_pids,
                "ground_img_paths": ground_img_paths
            }
            return dataset, pid_container
        
    def _check_before_run(self):
        if not op.exists(self.dataset_dir):
            raise RuntimeError(f"Dataset directory not found: {self.dataset_dir}")
        if not op.exists(self.train_anno_path):
            raise RuntimeError(f"Annotation file not found: {self.train_anno_path}")
