import os
import torch
from os import path as osp
import numpy as np
import pandas as pd
import collections
from PIL import Image
from tqdm import tqdm
from config import *
from pycocotools import mask as mutils
from rle import kaggle_rle_decode


# slightly modifications on https://www.kaggle.com/abhishek/mask-rcnn-using-torchvision-0-17
class FashionDataset(torch.utils.data.Dataset):
    def __init__(self,config,transforms = None):

        self.image_dir = config.img_dir
        # self.test_dir = config.test_dir
        self.annotation = pd.read_csv(config.ann_path)
        # resize the img of width and height
        self.width = config.width
        self.height = config.height
        self.transforms = transforms
        self.train_dict = collections.defaultdict(dict)
        #create a temp_dataframe for extraction of useful data
        self.annotation['CategoryId'] = self.annotation.ClassId.apply(lambda x: str(x).split('_')[0])
        df = self.annotation.groupby('ImageId')['EncodedPixels','CategoryId'].agg(lambda x: list(x)).reset_index()
        size = self.annotation.groupby('ImageId')['Height','Width'].mean().reset_index()
        df = df.merge(size, on = 'ImageId', how ='left')
         
        for idx, row in tqdm(df.iterrows(), total = len(df)):
            self.train_dict[idx]['image_id'] = row['ImageId']
            self.train_dict[idx]['image_path'] = osp.join(self.image_dir, row['ImageId'])
            self.train_dict[idx]['labels'] = row['CategoryId']
            self.train_dict[idx]['height'] = self.height
            self.train_dict[idx]['width'] = self.width
            self.train_dict[idx]['orig_height'] = row['Height']
            self.train_dict[idx]['orig_width'] = row['Width']
            self.train_dict[idx]['annotations'] = row['EncodedPixels']

    def __getitem__(self, idx):
        # load images as masks
        img_path = self.train_dict[idx]['image_path']
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.width,self.height),resample = Image.BILINEAR)

        train_data = self.train_dict[idx]
        # for gpu, it is better for np.uint8
        mask = np.zeros((len(train_data['annotations']),self.width,self.height), dtype = np.uint8)

        labels = []

        for ind, (ann, label) in enumerate(zip(train_data['annotations'], train_data['labels'])):

            sub_mask = kaggle_rle_decode(ann, train_data['orig_height'], train_data['orig_width'])
            # to convert array to image
            sub_mask = Image.fromarray(sub_mask)
            # resize the image to (255,255)
            sub_mask = sub_mask.resize((self.width, self.height),resample=Image.BILINEAR)
            mask[ind,:,:] = sub_mask
            # 0 is for background
            labels.append(int(label)+1)

        # get bounding box coordinates for each mask
        num_objs = len(labels)
        boxes = []
        #### make a reference to https://www.kaggle.com/abhishek/mask-rcnn-using-torchvision-0-17 #####
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])
        ######################################################################################################
        # get the final masks

        final_masks = np.zeros((len(new_masks),self.width, self.height), dtype = np.uint8)

        for ind, _m in enumerate(new_masks):
            final_masks[ind,:,:] = _m

        # convert everything into a tensor.Tensor
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch. as_tensor(new_labels, dtype = torch.int64)
        masks = torch.as_tensor(final_masks, dtype = torch.uint8)

        image_id = torch.tensor([idx])

        # calculate bounding box areas
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2]- boxes[:,0])
        # to check the instance is crowded or not (assume it is single instance)
        iscrowd = torch.zeros((num_objs, ) ,dtype = torch.int64)


        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return img, target

    def __len__(self):
        return len(self.train_dict)