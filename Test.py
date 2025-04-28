import os
import torch
from net import FMambaIR
import dataset
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
#from kornia.filters import laplacian
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def test(args):

    model = FMambaIR(in_chans=3)
    load_checkpoint(model, args.weights)
    model = model.cuda()
    model.eval()

    
    content_folder1 = './UIEBD/test/input1'
    information_folder = './UIEBD/test/input1'
    
    train_loader, My_data = dataset.style_loader(content_folder1, information_folder, args.size, 1)
    
    num_batch = len(train_loader)
    print(My_data.content1_list)
    # for epoch in range(args.epoch):
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(train_loader), total=num_batch):
                
            content = batch[0].float().cuda()
            information = batch[1].float().cuda()
            
            output = model(content)
                

            name = My_data.content1_list[idx].split('/')[4]
            print(name)
            save_image(output, args.save_dir+'/UIEBD/{}'.format(name))




def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint)
    except:
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', default=3, type=int)
    parser.add_argument('--epoch', default=5000, type=int)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--save_dir', default='result', type=str)
    parser.add_argument('--weights',
                    default='./checkpoint/UIEBD/checkpoint_UIEBD.pth', type=str,
                    help='Path to weights')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    
    test(args)
