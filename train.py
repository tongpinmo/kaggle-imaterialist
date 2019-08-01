import torch
from torch.utils.data import DataLoader
from os import path as osp
import os
from engine import train_one_epoch
from model import get_model_instance_segmentation
from FashionDataset import FashionDataset
from config import get_config,print_usage
import transform as T
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

def train(config):
    # create our own dataset and its data_loader
    tr_dt = FashionDataset(config, get_transform(train = True))
    tr_data_loader = DataLoader(
        tr_dt, config.batch_size , shuffle = True,
        num_workers = 8, collate_fn = lambda x: tuple(zip(*x))
    )

    # save the weights
    # weight_file = osp.join(config.save_dir, 'weights')
    # check them whether exists or not
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # there are 46 classes in total
    num_classes = 46 + 1
    # create model instance
    model = get_model_instance_segmentation(num_classes)

    #set model to device
    model.to(device)

    # for optim
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr = 0.001, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                   step_size=5,
                                                   gamma=0.1)

    for epoch in range(config.num_epochs):
        train_one_epoch(model, optim, tr_data_loader, device, epoch, print_freq=config.rep_intv)
        # updt the learning rate
        lr_scheduler.step()
        w1 = osp.join(config.save_dir , 'weights')
        wfile = osp.join(w1, '{}_model.bin'.format(str(epoch)))


        torch.save(model.state_dict(), wfile)


if __name__ == '__main__':
    # parse configuration
    config, unparsed = get_config()
    if len(unparsed)>0:
        print_usage()
        exit(1)

    train(config)