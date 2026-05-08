# encoding: utf-8
"""
@author:  huynguyen792
@contact: nguyet91@qut.edu.au
"""

import logging

import json

import os.path as osp

from prettytable import PrettyTable


class G2APS(object):
    logger = logging.getLogger("IRRA.dataset")
    dataset_dir = 'G2APS-AG'

    def __init__(self, root='./data',
                 verbose=True, **kwargs):
        super(G2APS, self).__init__()

        camera = [1]

        camera = 'a'
        self.dataset_dir = osp.join(root, self.dataset_dir)

        with open(osp.join(self.dataset_dir,"g2aps_train.json"),"r") as f:
            self.data_train = json.load(f)

        with open(osp.join(self.dataset_dir,"g2aps_test.json"),"r") as f:
            self.data_test = json.load(f)
        
        with open(osp.join(self.dataset_dir,"g2aps_val.json"),"r") as f:
            self.data_val = json.load(f)

        # train data
        train = []
        train_id_container = set()  
        for item in self.data_train:
            train.append((item[0],item[1],osp.join(self.dataset_dir,item[2][0],item[2][1]),osp.join(self.dataset_dir,item[3][0],item[3][1]),item[4]))
            train_id_container.add(item[0])

        self.train = train
        self.train_id_container = train_id_container

        ## test data
        img_paths_test = [osp.join(self.dataset_dir,i[0],i[1]) for i in self.data_test['img_paths']]
        pair_img_paths_test = [osp.join(self.dataset_dir,i[0],i[1]) for i in self.data_test['pair_img_paths']]

        self.test = {
            "image_pids": self.data_test['image_pids'],
            "img_paths": img_paths_test,
            "pair_img_paths": pair_img_paths_test,
            "caption_pids": self.data_test['caption_pids'],
            "captions": self.data_test['captions'],
        }
        self.test_id_container = set(self.data_test['image_pids'])

        ## val data
        img_paths_val = [osp.join(self.dataset_dir,i[0],i[1]) for i in self.data_val['img_paths']]
        pair_img_paths_val = [osp.join(self.dataset_dir,i[0],i[1]) for i in self.data_val['pair_img_paths']]

        self.val = {
            "image_pids": self.data_val['image_pids'],
            "img_paths": img_paths_val,
            "pair_img_paths": pair_img_paths_val,
            "caption_pids": self.data_val['caption_pids'],
            "captions": self.data_val['captions'],
        }
        self.val_id_container = set(self.data_val['image_pids'])


        if verbose:
            self.logger.info("=> G2APS Images and Captions are loaded")
            self.show_dataset_info()

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = \
            len(self.train_id_container), len(self.train), len(self.text_captions)
        num_test_pids, num_test_imgs, num_test_captions = \
            len(self.test_id_container), len(self.test['img_paths']), len(self.test['img_paths'])
        num_val_pids, num_val_imgs, num_val_captions = \
            len(self.val_id_container), len(self.val['img_paths']), len(self.val['img_paths'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        print(table)
        self.logger.info("\n"+str(table))


if __name__ == '__main__':
    save_path = r"E:\Share\jupyterDir\UAV-GA-TBPR"
    data = G2APS(root=save_path,name="G2APS-g")
    data.show_dataset_info()