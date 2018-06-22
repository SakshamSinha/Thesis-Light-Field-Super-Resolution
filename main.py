import argparse
import os
import torch
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import transforms
from Model import Generator
import torch.optim as optim

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='data', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
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

try:
    os.makedirs(opt.out)
except OSError:
    pass

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([transforms.RandomCrop(opt.imageSize * opt.upSampling),
                                transforms.ToTensor()])
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# print(len(dataset))
generator = Generator()
print(generator)

if opt.cuda:
    generator.cuda()

optimizer = optim.Adam(generator.parameters(), lr=opt.generatorLR)

for batch_no, data in enumerate(dataloader):
    high_img, _ = data

    # for j in range(opt.batchSize):
    #     low_res[j] = scale(high_res_real[j])
    #     high_res_real[j] = normalize(high_res_real[j])

    input1 = high_img[0, :, :, :]
    input2 = high_img[1, :, :, :]
    input3 = high_img[2, :, :, :]
    input4 = high_img[3, :, :, :]

    for j in range(opt.batchSize):
        high_img[j] = normalize(high_img[j])

    input_comb = torch.cat([scale(input1), scale(input2), scale(input3), scale(input4)],0)
    if opt.cuda:
        optimizer.zero_grad()
        high_res_real = Variable(high_img.cuda())
        high_res_fake = generator(Variable(high_img).cuda())

