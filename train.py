import models
import datas
import argparse
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from math import log10
import numpy as np
import datetime

from models import myutils
from utils.config import Config
import sys
import cv2
from utils.vis_flow import flow_to_color
import json
from loss import Loss
from torch.optim import Adamax
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# from torch.cuda.amp import autocast, GradScaler


# loading configures
parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
config = Config.from_file(args.config)
device = torch.device('cuda' if config.cuda else 'cpu')


# preparing datasets & normalization
normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
normalize2 = TF.Normalize([0, 0, 0], config.std)
trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

revmean = [-x for x in config.mean]
revstd = [1.0 / x for x in config.std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])

revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

if not os.path.exists(config.store_path):
    os.mkdir(config.store_path)

testset = datas.AniTriplet(config.testset_root, trans, config.test_size,
                                          config.test_crop_size, train=False)
trainset = datas.AniTriplet(config.trainset_root, trans, config.train_size,
                                          config.test_crop_size, train=False)
sampler = torch.utils.data.SequentialSampler(testset)
trainsampler = torch.utils.data.SequentialSampler(trainset)

validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)
trainloader = torch.utils.data.DataLoader(trainset, sampler=trainsampler, batch_size=6, shuffle=False, num_workers=1)

to_img = TF.ToPILImage()

print(testset)
print(trainset)
sys.stdout.flush()

# prepare model
model = getattr(models, config.model)(config.pwc_path)
model = torch.nn.DataParallel(model).to(device)
# scaler = GradScaler()


# load weights
# dict1 = torch.load(config.checkpoint)
# model.load_state_dict(dict1['model_state_dict'], strict=False)

# prepare others
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
store_path = config.store_path
torch.cuda.manual_seed(config.random_seed)

# loss function
criterion = Loss(config)
optimizer = Adamax(model.parameters(), lr=config.lr, betas=(0.9, 0.999))

def save_flow_to_img(flow, des):
        f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
        fcopy = f.copy()
        fcopy[:, :, 0] = f[:, :, 1]
        fcopy[:, :, 1] = f[:, :, 0]
        cf = flow_to_color(-fcopy)
        cv2.imwrite(des + '.jpg', cf)

def train(config):

    ## values for whole image
    psnr_whole = 0
    # psnrs = np.zeros([len(testset), config.inter_frames])
    ssim_whole = 0
    # ssims = np.zeros([len(testset), config.inter_frames])
    losses, psnrs, ssims = myutils.init_meters(config.loss)
    # folders = []

    print('Everything prepared. Ready to train...')
    sys.stdout.flush()

    model.train()
    criterion.train()
    torch.cuda.empty_cache()

    for validationIndex, validationData in enumerate(trainloader, 0):
        if (validationIndex % 200 == 0):
            print('Training {}/{}-th group...'.format(validationIndex, len(testset)))

        # sample, flow = validationData
        sample = validationData

        frame0 = None
        frame1 = sample[0]
        frame3 = None
        frame2 = sample[-1]

        # folders.append(folder[0][0])

        # initial SGM flow
        # F12i, F21i = flow

        # F12i = F12i.float().cuda()
        # F21i = F21i.float().cuda()

        ITs = sample[1]
        I1 = frame1.cuda()
        I2 = frame2.cuda()

        # if not os.path.exists(config.store_path + '/' + folder[0][0]):
        #     os.mkdir(config.store_path + '/' + folder[0][0])

        # revtrans(I1.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + index[0][0] + '.jpg')
        # revtrans(I2.cpu()[0]).save(store_path + '/' + folder[-1][0] + '/' + index[-1][0] + '.jpg')

        t = 1.0 / 2.0
        optimizer.zero_grad()
        outputs = model(I1, I2, None, None, t)

        It_warp = outputs[0]

        # to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0)).save(
        #     store_path + '/' + folder[1][0] + '/' + index[1][0] + '.png')

        # estimated = revNormalize(It_warp[0].cpu()).clamp(0.0, 1.0).detach().numpy().transpose(1, 2, 0)
        # gt = revNormalize(ITs[tt][0]).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)

        # whole image value
        # this_psnr = psnr(est, gt)
        # this_ssim = ssim(est, gt, multichannel=True, gaussian=True)

        myutils.eval_metrics(It_warp.cpu(), ITs, psnrs, ssims)

        loss, _ = criterion(It_warp.cpu(), ITs)
        # losses['total'].update(loss.item())
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        loss.backward()
        optimizer.step()

        # psnrs[validationIndex][tt] = this_psnr
        # ssims[validationIndex][tt] = this_ssim

        # psnr_whole += this_psnr
        # ssim_whole += this_ssim
        if (validationIndex % 200 == 0):
            print('Train Epoch: {}\tPSNR: {:.4f} \tSSIM: {:.4f}\t Lr:{:.6f}'.format(
                epoch, psnrs.avg, ssims.avg, optimizer.param_groups[0]['lr'], flush=True))

        losses, psnrs, ssims = myutils.init_meters(config.loss)

    # psnr_whole /= (len(testset) * config.inter_frames)
    # ssim_whole /= (len(testset) * config.inter_frames)

    return None


def validate(config):

    ## values for whole image
    # psnr_whole = 0
    # psnrs = np.zeros([len(testset), config.inter_frames])
    # ssim_whole = 0
    # ssims = np.zeros([len(testset), config.inter_frames])
    losses, psnrs, ssims = myutils.init_meters(config.loss)

    # folders = []

    print('Everything prepared. Ready to test...')  
    sys.stdout.flush()

    #  start testing...

    model.eval()
    criterion.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        ii = 0
        for validationIndex, validationData in enumerate(validationloader, 0):
            if (validationIndex % 200 == 0):
                print('Testing {}/{}-th group...'.format(validationIndex, len(testset)))
            sys.stdout.flush()
            # sample, flow,  index, folder = validationData
            sample = validationData

            frame0 = None
            frame1 = sample[0]
            frame3 = None
            frame2 = sample[-1]

            # folders.append(folder[0][0])
            
            # initial SGM flow
            # F12i, F21i  = flow

            # F12i = F12i.float().cuda()
            # F21i = F21i.float().cuda()

            ITs = sample[1]
            I1 = frame1.cuda()
            I2 = frame2.cuda()

            # if not os.path.exists(config.store_path + '/' + folder[0][0]):
            #     os.mkdir(config.store_path + '/' + folder[0][0])


            # revtrans(I1.cpu()[0]).save(store_path + '/' + folder[0][0] + '/'  + index[0][0] + '.jpg')
            # revtrans(I2.cpu()[0]).save(store_path + '/' + folder[-1][0] + '/' +  index[-1][0] + '.jpg')
            # for tt in range(config.inter_frames):
                # x = config.inter_frames
            t = 1.0/2.0

            # outputs = model(I1, I2, F12i, F21i, t)
            outputs = model(I1, I2, None, None, t)

            It_warp = outputs[0]

            # to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0)).save(store_path + '/' + folder[1][0] + '/' + index[1][0] + '.png')
            # if (ii % 40 == 0):
            #     to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0)).save(store_path + '/' + str(ii) + '.jpg')
            #     to_img(ITs[0]).save(store_path + '/' + str(ii) + '-.jpg')
            # ii += 1
            estimated = revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0)
            gt = ITs[0]
            # estimated = revNormalize(It_warp[0].cpu()).clamp(0.0, 1.0).detach().numpy().transpose(1, 2, 0)
            # gt = revNormalize(ITs[0]).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)

            # whole image value
            # this_psnr = psnr(estimated, gt)
            # this_ssim = ssim(estimated, gt, multichannel=True, gaussian=True)
            #
            # psnrs[validationIndex][tt] = this_psnr
            # ssims[validationIndex][tt] = this_ssim
            #
            # psnr_whole += this_psnr
            # ssim_whole += this_ssim
            # losses['total'].update(loss.item())
            myutils.eval_metrics(estimated, gt, psnrs, ssims)

        # psnr_whole /= (len(testset) * config.inter_frames)
        # ssim_whole /= (len(testset) * config.inter_frames)

    return psnrs.avg, ssims.avg


if __name__ == "__main__":

    for epoch in range(0, config.max_epoch):

        # train(config)
        psnr, ssim = validate(config)
        #torch.save()
        # print('PSNR is {}'.format(psnr))
        # print('SSIM is {}'.format(ssim))
        print('Epoch [{0}/{1}], Val_PSNR:{2:.2f}, Val_SSIM:{3:.4f}'
              .format(epoch, config.max_epoch, psnr, ssim))

    # for ii in range(config.inter_frames):
    #     print('PSNR of validation frame' + str(ii+1) + ' is {}'.format(np.mean(psnrs[:, ii])))
    #
    # for ii in range(config.inter_frames):
    #     print('PSNR of validation frame' + str(ii+1) + ' is {}'.format(np.mean(ssims[:, ii])))



    # with open(config.store_path + '/psnr.txt', 'w') as f:
    #     for index in sorted(range(len(psnrs[:, 0])), key=lambda k: psnrs[k, 0]):
    #         f.write("{}\t{}\n".format(folder[index], psnrs[index, 0]))
    #
    # with open(config.store_path + '/ssim.txt', 'w') as f:
    #     for index in sorted(range(len(ssims[:, 0])), key=lambda k: ssims[k, 0]):
    #         f.write("{}\t{}\n".format(folder[index], ssims[index, 0]))
