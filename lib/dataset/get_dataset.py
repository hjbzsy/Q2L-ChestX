import os
import sys
sys.path.append("/home/home/dh/xfan/paper/test/q2l/query2labels2_4/lib/")

import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from utils.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp
from dataset.nihdataset import NIHDataset

def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            RandAugment(),
                                               transforms.ToTensor(),
                                               normalize]
    if args.cutout:
        print("Using Cutout!!!")
        train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    train_data_transform = transforms.Compose(train_data_transform_list)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )
    elif args.dataname == 'nih':
        dataset_dir = args.dataset_dir
        nih_transform = transforms.Compose([
        # transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        train_dataset = NIHDataset(
        data_path=dataset_dir,
        input_transform=nih_transform,
        train=True
        )
        val_dataset = NIHDataset(
        data_path=dataset_dir,
        input_transform=nih_transform,
        train=False
        )
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset


import argparse
def parser_args():
    parser = argparse.ArgumentParser(description='Query2Label MSCOCO Training')
    #传入数据集的名字
    parser.add_argument('--dataname', help='dataname', default='nih')
    parser.add_argument('--dataset_dir', help='dir of dataset', default='./')
    parser.add_argument('--img_size', default=448, type=int,
                        help='size of input images')

    parser.add_argument('--output', metavar='DIR',
                        help='path to output folder')
    parser.add_argument('--num_class', default=80, type=int,
                        help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')


    # data aug
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ')

    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')
    
    args = parser.parse_args()

    return args

def get_args():
    args = parser_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train,test = get_datasets(args)
    from torch.utils.data.dataloader import DataLoader
    loader = DataLoader(train,batch_size=8)
    for (img,label) in loader:
        print(img.size())
        print(label.size())
        
        break
    loader = DataLoader(test,batch_size=8)
    for (img,label) in loader:
        print(img.size())
        print(label.size())
        
        break