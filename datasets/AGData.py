# encoding: utf-8
"""
@author:  huynguyen792
@contact: nguyet91@qut.edu.au
"""

import logging

from .agtbpr import AG_ReID
from .g2aps import G2APS 
import json

import os.path as osp

from prettytable import PrettyTable


class AGData(object):
    logger = logging.getLogger("IRRA.dataset")

    def __init__(self, root='./data',
                 verbose=True, type="dual", **kwargs):
        super(AGData, self).__init__()
        # agtbpr_name = 'AG-ReID.v1'
        #g2aps_name = 'UAV-GA-TBPR'

        agtbpr_data = AG_ReID(root,verbose=False)
        g2aps_data = G2APS(root,verbose=False)

        self.type=type

        train = agtbpr_data.train
        id_len_agtbpr = len(agtbpr_data.train_id_container)
        for item in g2aps_data.train:
            train.append([item[0]+id_len_agtbpr,item[1]+len(agtbpr_data.train)]+[i for i in item[2:]])
        
        self.train = train
        self.train_id_container = set([i[0] for i in train])

        test = agtbpr_data.test
        id_len_agtbpr = len(agtbpr_data.test_id_container)
        keys = list(agtbpr_data.test.keys())
        for key in keys:
            if "pids" in key:
                test[key] = list(test[key]) + [i + id_len_agtbpr for i in g2aps_data.test[key]]
            else:
                test[key] = list(test[key]) + list(g2aps_data.test[key])
        
        self.test = test
        self.test_id_container = set(list(agtbpr_data.test_id_container)+[i+id_len_agtbpr for i in list(g2aps_data.test_id_container)])

        val = agtbpr_data.val
        id_len_agtbpr = len(agtbpr_data.val_id_container)
        keys = list(agtbpr_data.val.keys())
        for key in keys:
            if "pids" in key:
                val[key] = list(val[key]) + [i + id_len_agtbpr for i in g2aps_data.val[key]]
            else:
                val[key] = list(val[key]) + list(g2aps_data.val[key])
        
        self.val = val
        self.val_id_container = set(list(agtbpr_data.val_id_container)+[i+id_len_agtbpr for i in list(g2aps_data.val_id_container)])

        if verbose:
            self.logger.info("=> AGTBPR Images and Captions are loaded")
            self.show_dataset_info()

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = \
            len(self.train_id_container), len(self.train), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = \
            len(self.test_id_container), len(self.test['captions']), len(self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = \
            len(self.val_id_container), len(self.val['captions']), len(self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        print(table)
        self.logger.info("\n"+str(table))
    


class AGSGData(object):
    logger = logging.getLogger("IRRA.dataset")

    def __init__(self, root='./data',
                 verbose=True, type="single",img="a", **kwargs):
        super(AGSGData, self).__init__()

        agtbpr_data = AG_ReID(root,verbose=False)
        g2aps_data = G2APS(root,verbose=False)
        # agtbpr_name = 'AG-ReID.v1'
        #g2aps_name = 'UAV-GA-TBPR'

        self.type=type
        self.img = img
        # a is aerial only; g is ground only; ag is all data

        train = agtbpr_data.train
        id_len_agtbpr = len(agtbpr_data.train_id_container)
        for item in g2aps_data.train:
            train.append([item[0]+id_len_agtbpr,item[1]+len(agtbpr_data.train)]+[i for i in item[2:]])
        
        dataset_tmp = []
        for item in train:
            if "a" in self.img:
                dataset_tmp.append((item[0], item[1], item[2], None, item[4]))
            if "g" in self.img:
                dataset_tmp.append((item[0], item[1], item[3], None, item[4]))
        
        # dataset = dataset_tmp
        self.train = dataset_tmp
        self.train_id_container = set([i[0] for i in dataset_tmp])

        test = agtbpr_data.test
        id_len_agtbpr = len(agtbpr_data.test_id_container)
        keys = list(agtbpr_data.test.keys())
        for key in keys:
            if "pids" in key:
                test[key] = list(test[key]) + [i + id_len_agtbpr for i in g2aps_data.test[key]]
            else:
                test[key] = list(test[key]) + list(g2aps_data.test[key])
        
        test["pair_img_paths"] = [None] * len(test["pair_img_paths"])

        self.test = test
        self.test_id_container = set(list(agtbpr_data.test_id_container)+[i+id_len_agtbpr for i in list(g2aps_data.test_id_container)])

        val = agtbpr_data.val
        id_len_agtbpr = len(agtbpr_data.val_id_container)
        keys = list(agtbpr_data.val.keys())
        for key in keys:
            if "pids" in key:
                val[key] = list(val[key]) + [i + id_len_agtbpr for i in g2aps_data.val[key]]
            else:
                val[key] = list(val[key]) + list(g2aps_data.val[key])
        
        val["pair_img_paths"] = [None] * len(val["pair_img_paths"])
        self.val = val
        self.val_id_container = set(list(agtbpr_data.val_id_container)+[i+id_len_agtbpr for i in list(g2aps_data.val_id_container)])

        if verbose:
            self.logger.info("=> AGTBPR Images and Captions are loaded")
            self.show_dataset_info()

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = \
            len(self.train_id_container), len(self.train), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = \
            len(self.test_id_container), len(self.test['captions']), len(self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = \
            len(self.val_id_container), len(self.val['captions']), len(self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        print(table)
        self.logger.info("\n"+str(table))


class AGDataAttr(object):
    logger = logging.getLogger("IRRA.dataset")

    def __init__(self, root='./data',
                 verbose=True, type="dualAttr", **kwargs):
        super(AGDataAttr, self).__init__()
        # agtbpr_name = 'AG-ReID.v1'
        #g2aps_name = 'UAV-GA-TBPR'
        # with open(osp.join(root,"atr_dict_final.json"))  as f:
        #     self.atr_dict_all = json.load(f)

        with open(osp.join(root,"attrs.json"),"r") as f:
            attr_data = json.load(f)
        
        data_key = {list(item.keys())[0]:list(item.values())[0] for item in attr_data}

        agtbpr_data = AG_ReID(root,verbose=False)
        g2aps_data = G2APS(root,verbose=False)

        self.type=type

        train = agtbpr_data.train
        id_len_agtbpr = len(agtbpr_data.train_id_container)
        for item in g2aps_data.train:
            train.append([item[0]+id_len_agtbpr,item[1]+len(agtbpr_data.train)]+[i for i in item[2:]])
        
        # pre process, cluster person attrs
        count_dict_a = {i:[] for i in set([item[0] for item in train])}
        count_dict_g = {i:[] for i in set([item[0] for item in train])}
        name_id_dict = {}

        # first read attr data, map key to id
        for item in train:
            key1 = item[2]
            key2 = item[3]
            if "AG-ReID" in key1:
                path_tmp, name = osp.split(key1)
                name = name[:-4]
                path = osp.split(path_tmp)[-1]
                key = "_".join([path,name])
                name_id_dict[key] = item[0]
                count_dict_a[item[0]].append(data_key[key])

                path_tmp, name = osp.split(key2)
                name = name[:-4]
                path = osp.split(path_tmp)[-1]
                key = "_".join([path,name])
                name_id_dict[key] = item[0]
                count_dict_g[item[0]].append(data_key[key])
                pass
            elif "G2APS-AG" in key1:
                key = key1[-22:]
                count_dict_a[item[0]].append(data_key[key])
                name_id_dict[key] = item[0]
                key = key2[-22:]
                count_dict_g[item[0]].append(data_key[key])
                name_id_dict[key] = item[0]
                pass
            else:
                raise NotImplementedError("Unknown Dataset setting")
        
        # cluster by person id
        atr_dict_all = {}
        for idx in set([item[0] for item in train]):
            atr_list_tmp = []
            for i_a in range(5):
                atr_list = [i[i_a] for i in count_dict_a[idx]] + [i[i_a] for i in count_dict_g[idx]]
                atr = max(atr_list,key=atr_list.count)
                atr_list_tmp.append(atr)
            atr_dict_all[idx] = atr_list_tmp

        # process attribute data
        train_tmp = []
        for item in train:
            key1 = item[2]
            key2 = item[3]
            if "AG-ReID" in key1:
                path_tmp, name = osp.split(key1)
                name = name[:-4]
                path = osp.split(path_tmp)[-1]
                key = "_".join([path,name])
                idx_a = name_id_dict[key]

                data_item = data_key[key]
                info_a = atr_dict_all[idx_a]+data_item[-2:]

                #info_a = self.atr_dict_all[key]

                path_tmp, name = osp.split(key2)
                name = name[:-4]
                path = osp.split(path_tmp)[-1]
                key = "_".join([path,name])
                idx_a = name_id_dict[key]

                data_item = data_key[key]
                info_g = atr_dict_all[idx_a]+data_item[-2:]

                #info_g = self.atr_dict_all[key]
                pass
            elif "G2APS-AG" in key1:
                key = key1[-22:]
                idx_a = name_id_dict[key]
                data_item = data_key[key]
                info_a = atr_dict_all[idx_a]+data_item[-2:]

                #info_a = self.atr_dict_all[key]
                key = key2[-22:]
                idx_a = name_id_dict[key]
                data_item = data_key[key]
                info_g = atr_dict_all[idx_a]+data_item[-2:]

                #info_g = self.atr_dict_all[key]
                pass
            
            info = list(item)+info_a+info_g
            train_tmp.append(info)
        
        train = train_tmp

        self.train = train
        self.train_id_container = set([i[0] for i in train])

        test = agtbpr_data.test
        id_len_agtbpr = len(agtbpr_data.test_id_container)
        keys = list(agtbpr_data.test.keys())
        for key in keys:
            if "pids" in key:
                test[key] = list(test[key]) + [i + id_len_agtbpr for i in g2aps_data.test[key]]
            else:
                test[key] = list(test[key]) + list(g2aps_data.test[key])
        
        self.test = test
        self.test_id_container = set(list(agtbpr_data.test_id_container)+[i+id_len_agtbpr for i in list(g2aps_data.test_id_container)])

        val = agtbpr_data.val
        id_len_agtbpr = len(agtbpr_data.val_id_container)
        keys = list(agtbpr_data.val.keys())
        for key in keys:
            if "pids" in key:
                val[key] = list(val[key]) + [i + id_len_agtbpr for i in g2aps_data.val[key]]
            else:
                val[key] = list(val[key]) + list(g2aps_data.val[key])
        
        self.val = val
        self.val_id_container = set(list(agtbpr_data.val_id_container)+[i+id_len_agtbpr for i in list(g2aps_data.val_id_container)])

        if verbose:
            self.logger.info("=> AGTBPR Images and Captions are loaded")
            self.show_dataset_info()

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = \
            len(self.train_id_container), len(self.train), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = \
            len(self.test_id_container), len(self.test['captions']), len(self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = \
            len(self.val_id_container), len(self.val['captions']), len(self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        print(table)
        self.logger.info("\n"+str(table))
    