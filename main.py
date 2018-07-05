import argparse
import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image
import time

from Model import SingleGenerator, SingleDiscriminator, MultipleGenerator, MultipleDiscriminator
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from train_multiple import train_multiple
from train_single import train_single

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='data', help='path to dataset')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--cuda', action='store_true', default='true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='', help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)

writer = SummaryWriter()

try:
    os.makedirs('output/test/high_res_fake')
    os.makedirs('output/test/high_res_real')
    os.makedirs('output/test/low_res')
    os.makedirs('output/train/high_res_fake')
    os.makedirs('output/train/high_res_real')
    os.makedirs('output/train/low_res')
except OSError:
    pass


try:
    os.makedirs(opt.out)
except OSError:
    pass

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


transform = transforms.Compose([#transforms.RandomCrop((opt.imageSize*opt.upSampling,opt.imageSize*opt.upSampling)),
                                 transforms.Resize((opt.imageSize*opt.upSampling,opt.imageSize*opt.upSampling)),
                                transforms.ToTensor()])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

# dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
#
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                          shuffle=True, num_workers=int(opt.workers))
dataset = {x: datasets.ImageFolder(os.path.join(opt.dataroot, x), transform=transform) for x in ['train','test']}

dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=opt.batchSize,drop_last=True,
                                         shuffle=True, num_workers=int(opt.workers)) for x in ['train','test']}

# # print(len(dataset))
# generator = SingleGenerator(5,opt.upSampling)
# print(generator)
# # print(len(dataset))
# discriminator = SingleDiscriminator()
# print(discriminator)
#
# train_single(generator, discriminator, opt, dataloader, writer, scale)

generator = MultipleGenerator(16,opt.upSampling)
print(generator)
discriminator = MultipleDiscriminator()
print(discriminator)
train_multiple(generator, discriminator, opt, dataloader, writer, scale)