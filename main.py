import argparse
import os

import imageio
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image
import time

from Model import SingleGenerator, SingleDiscriminator, MultipleGenerator, MultipleDiscriminator, AngRes
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from test_multiple import test_multiple, test_angres
from test_single import test_single
from train_multiple import train_multiple, train_angres, bilinear_upsampling
from train_single import train_single
from utils import get_matlab_lf

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='data', help='path to dataset')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--cuda', action='store_true', default='true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='checkpoints/generator_final.pth', help="path to generator weights ")
parser.add_argument('--angresWeights', type=str, default='checkpoints/AngRes_final.pth', help="path to Angular resolution model weights ")
parser.add_argument('--discriminatorWeights', type=str, default='checkpoints/discriminator_final.pth', help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)

writer = SummaryWriter()

try:
    os.makedirs('output/train/bilinear_real')
    os.makedirs('output/train/bilinear_fake')
    os.makedirs('output/test/reals')
    os.makedirs('output/test/reals')
    # os.makedirs('output/test/high_res_fake')
    # os.makedirs('output/test/high_res_real')
    # os.makedirs('output/test/low_res')
    os.makedirs('output/test/ang_res_fake')
    os.makedirs('output/test/reals')
    # os.makedirs('output/train/high_res_fake')
    # os.makedirs('output/train/high_res_real')
    # os.makedirs('output/train/low_res')
    os.makedirs('output/train/ang_res_fake')
    os.makedirs('output/train/reals')
except OSError:
    print("error while creating directory")
    pass


try:
    os.makedirs(opt.out)
except OSError:
    pass

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


transform = transforms.Compose([#transforms.CenterCrop((opt.imageSize*opt.upSampling,opt.imageSize*opt.upSampling)),
                                transforms.Resize((opt.imageSize*opt.upSampling,opt.imageSize*opt.upSampling)),
                                transforms.ToTensor()])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                 std = [0.229, 0.224, 0.225])
                            ])


dataset = {x: datasets.ImageFolder(os.path.join(opt.dataroot, x), transform=transform) for x in ['train','test']}

dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=opt.batchSize,drop_last=True,
                                         shuffle=False, num_workers=int(opt.workers)) for x in ['train','test']}

# # print(len(dataset))
# generator = SingleGenerator(5,opt.upSampling)
# print(generator)
# print(len(dataset))
# discriminator = SingleDiscriminator()
# print(discriminator)

# train_single(generator, discriminator, opt, dataloader, writer, scale)

# test_single(generator, discriminator, opt, dataloader,  scale)

generator = MultipleGenerator(16,opt.upSampling)
print(generator)
discriminator = MultipleDiscriminator()
print(discriminator)
#bilinear_upsampling(opt, dataloader, scale)
# train_multiple(generator, discriminator, opt, dataloader, writer, scale)
# #
test_multiple(generator, discriminator, opt, dataloader, scale)

# ang_model = AngRes()
# print(ang_model)
#
# get_matlab_lf()
# train_angres(ang_model, dataloader, opt, writer)
#
# test_angres(ang_model, dataloader, opt)
# images = []
# import re
# numbers = re.compile(r'(\d+)')
# def numericalSort(value):
#     parts = numbers.split(value)
#     parts[1::2] = map(int, parts[1::2])
#     return parts
#
# for filename in sorted(os.listdir('output/test/high_res_fake'),key=numericalSort):
#     images.append(imageio.imread('output/test/high_res_fake/'+filename))
#
# imageio.mimsave('output/low_res.gif', images)