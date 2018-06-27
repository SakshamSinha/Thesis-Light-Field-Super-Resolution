import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image
import time
from utils import normalize, imshow
from Model import SingleGenerator, SingleDiscriminator
import torch.optim as optim
import torch.nn as nn

def train_single(generator, discriminator, opt, dataloader, writer, scale):
    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    ones_const = Variable(torch.ones(1, 1))

    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        ones_const = ones_const.cuda()

    optimizer = optim.Adam(generator.parameters(), lr=opt.generatorLR)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)
    scheduler_gen = ReduceLROnPlateau(optimizer, 'min', factor = 0.5,patience = 3, verbose = True)
    scheduler_dis = ReduceLROnPlateau(optim_discriminator, 'min', factor = 0.5,patience = 3, verbose = True)
    curr_time = time.time()

    for epoch in range(opt.nEpochs):
        mean_generator_content_loss = 0.0
        mean_generator_adversarial_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0
        high_res_fake = 0
        for batch_no, data in enumerate(dataloader):
            high_img, _ = data

            # for j in range(opt.batchSize):
            #     # low_res[j] = scale(high_res_real[j])
            #     high_res_real[j] = normalize(high_res_real[j])
            print("batch no. {} shape of input {}".format(batch_no,high_img.shape))
            input1 = high_img[0, :, :, :]
            input2 = high_img[1, :, :, :]
            input3 = high_img[2, :, :, :]
            input4 = high_img[3, :, :, :]
            # imshow(input3)
            for j in range(opt.batchSize):
                high_img[j] = normalize(high_img[j])
            high_comb=torch.cat([high_img[0], high_img[1], high_img[2], high_img[3]],0)

            high_comb = Variable(high_comb[np.newaxis, :]).cuda()
            # imshow(high_comb.cpu().data)
            input_comb = torch.cat([scale(input1), scale(input2), scale(input3), scale(input4)],0)
            input_comb = input_comb[np.newaxis, :]
            if opt.cuda:
                optimizer.zero_grad()
                high_res_real = Variable(high_img.cuda())
                high_res_fake = generator(Variable(input_comb).cuda())
                target_real = Variable(torch.rand(1, 1) * 0.5 + 0.7).cuda()
                target_fake = Variable(torch.rand(1, 1) * 0.3).cuda()
                # import ipdb;
                #
                # ipdb.set_trace()

                outputs = torch.chunk(high_img.cpu().data,4,0)
                # imshow(outputs[0])
                generator_content_loss = content_criterion(high_res_fake, high_comb)
                mean_generator_content_loss += generator_content_loss.data[0]
                generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
                mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

                generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
                mean_generator_total_loss += generator_total_loss.data[0]

                generator_total_loss.backward()
                optimizer.step()

                discriminator.zero_grad()

                discriminator_loss = adversarial_criterion(discriminator(high_comb), target_real) + \
                                     adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
                mean_discriminator_loss += discriminator_loss.data[0]

                discriminator_loss.backward()
                optim_discriminator.step()

        imshow(high_res_fake.cpu().data)
        scheduler_gen.step(mean_generator_total_loss)
        scheduler_dis.step(mean_discriminator_loss)
        # import ipdb;
        # ipdb.set_trace()
        writer.add_scalar("per epoch/generator lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("per epoch/discriminator lr", optim_discriminator.param_groups[0]['lr'], epoch)
        writer.add_scalar("per epoch/discriminator training loss", mean_discriminator_loss, epoch)
        writer.add_scalar("per epoch/generator training loss", mean_generator_total_loss, epoch)
        writer.add_scalar("per epoch/time taken", time.time()-curr_time, epoch)

