import os
import torch
# from unet_model import UNet 
# from under_mb1 import VSSM
from under_mb import VSSM
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

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VSSM(in_chans=3)
    # model = nn.DataParallel(model, device_ids=[0, 1])
 
    load_checkpoint(model, args.weights)
 
    model = model.cuda()
    model.eval()
    
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    
    # mse = nn.L1Loss().cuda()
    
    content_folder1 = './UIEBD/test/input1'
    information_folder = './UIEBD/test/input1'
    # content_folder1 = './UIEBD/color7'
    # information_folder = './UIEBD/color7'
    # content_folder1 = './LOL/test/low'
    # information_folder = './LOL/test/high'
    # content_folder1 = './deraining/rain100L/test/rain'
    # information_folder = './deraining/rain100L/test/norain'
    # content_folder1 = './RESIDE-6K/test/hazy'
    # information_folder = './RESIDE-6K/test/GT'
    
    train_loader, My_data = dataset.style_loader(content_folder1, information_folder, args.size, 1)
    
    num_batch = len(train_loader)
    print(My_data.content1_list)
    # for epoch in range(args.epoch):
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(train_loader), total=num_batch):
            # total_iter = epoch*num_batch  + idx
                
            content = batch[0].float().cuda()
            information = batch[1].float().cuda()
            
            # optimizer.zero_grad()

            #content = torch.exp(content)
            
            output = model(content)
                
                        
            # total_loss =  mse(output, information) 

            # total_loss.backward()
            

            # optimizer.step()


            
            # if np.mod(total_iter+1, 1) == 0:
            #     print('{}, Epoch:{} Iter:{} total loss: {}'.format(args.save_dir, epoch, total_iter, total_loss.item()))
                
                
                
                
            # if not os.path.exists(args.save_dir+'/image'):
            #     os.mkdir(args.save_dir+'/image')
            name = My_data.content1_list[idx].split('/')[4]
            print(name)

            # save_image(information, args.save_dir+'/LSUI/GT/{}'.format(name))
            save_image(output, args.save_dir+'/color7/{}'.format(name))
            # save_image(information, args.save_dir+'/6k/GT/{}'.format(name))
            # torch.save(model.state_dict(), 'model' +'/our_dehaze{}.pth'.format(epoch))

        #   if epoch % 1 ==0:
        #     #content = torch.log(content)
        #     #output = torch.log(output)
        #     out_image = torch.cat([content[0:3], output[0:3], information[0:3]], dim=0)
        #     save_image(out_image, args.save_dir+'/image/iter{}_1.jpg'.format(total_iter+1))
        #     torch.save(model.state_dict(), 'model' +'/our_dehaze{}.pth'.format(epoch))



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


# def load_checkpoint(model, weights):
#     checkpoint = torch.load(weights)
#     try:
#         model.load_state_dict(checkpoint)
#     except:
#         state_dict = checkpoint
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = 'module.' + k  # remove `module.`
#             new_state_dict[name] = v
#         model.load_state_dict(new_state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', default=3, type=int)
    parser.add_argument('--epoch', default=5000, type=int)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--save_dir', default='result', type=str)
    parser.add_argument('--weights',
                    default='./model/cupdownF_UIEBD/our_lol500.pth', type=str,
                    help='Path to weights')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    
    test(args)
