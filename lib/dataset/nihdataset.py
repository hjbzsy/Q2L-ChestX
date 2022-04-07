import torch
import sys, os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm
import pickle


cate = ['Cardiomegaly', 'Pneumothorax', 'Consolidation', 'Mass', 'Pleural_Thickening', 'Infiltration', 'Edema',
        'Hernia', 'Fibrosis', 'No Finding', 'Emphysema', 'Pneumonia', 'Nodule', 'Atelectasis', 'Effusion']
# cate = ['Cardiomegaly', 'Pneumothorax', 'Consolidation', 'Mass', 'Pleural_Thickening', 'Infiltration', 'Edema',
#         'Hernia', 'Fibrosis',  'Emphysema', 'Pneumonia', 'Nodule', 'Atelectasis', 'Effusion']

category_map = {cate[i]:i+1 for i in range(15)}
# category_map = {cate[i]:i+1 for i in range(14)}

class NIHDataset(data.Dataset):
    def __init__(self, data_path,input_transform=None,
                 used_category=-1,train=True):
        # self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        # with open('./data/coco/category.json','r') as load_category:
        #     self.category_map = json.load(load_category)
        self.data_path = data_path
        if train == True:
            # self.data = np.load(data_path+"/unhealthyX_sample_train.npy",allow_pickle=True)
            # 包含全部数据的数据集/home/sda1data/zc/nih/nih/traindata384.pickle
            #包含10000条数据  /home/home/dh/xfan/paper/test/query2labels2_11/preprocessimg/traindata384.pickle
            #包含200条数据 /home/home/dh/xfan/paper/test/query2labels2_11/smalldataset/traindata384.pickle
            filepath = '/home/sda1data/zc/nih/nih/traindata416.pickle'
            # filepath = '/home/sda1data/zc/nih/nih/train512.pickle'
            # filepath = "/home/home/dh/xfan/paper/test/q2l/query2labels2_11/smalldataset/traindata384.pickle"
            self.data = pickle.load(open(filepath,'rb'))
        else:
            # self.data = np.load(data_path + "/unhealthyX_sample_test.npy", allow_pickle=True)
            filepath = '/home/sda1data/zc/nih/nih/testdata416.pickle'
            # filepath = '/home/sda1data/zc/nih/nih/test512.pickle'
            # filepath = "/home/home/dh/xfan/paper/test/q2l/query2labels2_11/smalldataset/traindata384.pickle"
            
            self.data = pickle.load(open(filepath,'rb'))
        data_length = len(self.data)
        # self.data = self.data[:int(data_length/50)]   #   considering part of data，SPEEDUP
        random.shuffle(self.data)
        self.category_map = category_map
        self.input_transform = input_transform
        # self.labels_path = labels_path
        self.used_category = used_category


    def __getitem__(self, index):
        img = Image.fromarray(self.data[index][1]).convert("RGB")
        label = np.array(self.data[index][2]).astype(np.float64)
        if self.input_transform:
            img = self.input_transform(img)
        return img, label

    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(15)
        # label_num = len(categories)
        for c in categories:
            index = self.category_map[str(c)] - 1
            label[index] = 1.0  # / label_num
        return label

    def __len__(self):
        return len(self.data)

    def save_datalabels(self, outpath):
        """
            Save datalabels to disk.
            For faster loading next time.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        # labels = np.array(self.labels)
        # np.save(outpath, labels)




