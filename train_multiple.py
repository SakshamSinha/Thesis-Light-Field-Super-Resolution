import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from utils import normalize, imsave, avg_msssim, psnr
import torch.optim as optim
import torch.nn as nn
import numpy as np

def train_multiple(generator, discriminator, opt, dataloader, writer, scale):
    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    ones_const = Variable(torch.ones(opt.batchSize, 1))

    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        ones_const = ones_const.cuda()

    optimizer = optim.Adam(generator.parameters(), lr=opt.generatorLR)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)
    scheduler_gen = ReduceLROnPlateau(optimizer, 'min', factor = 0.5,patience = 5, verbose = True)
    scheduler_dis = ReduceLROnPlateau(optim_discriminator, 'min', factor = 0.5,patience = 5, verbose = True)
    curr_time = time.time()
    inputs = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    for epoch in range(opt.nEpochs):
        for phase in ['train', 'test']:
            if phase == 'test':
                generator.train(False)
                discriminator.train(False)
            else:
                generator.train(True)
                discriminator.train(True)

            mean_generator_content_loss = 0.0
            mean_generator_adversarial_loss = 0.0
            mean_generator_total_loss = 0.0
            mean_discriminator_loss = 0.0
            mean_psnr = 0.0
            mean_msssim = 0.0
            high_img = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
            inputs= torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
            high_res_fake = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
            high_res_fake2 = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
            for batch_no, data in enumerate(dataloader[phase]):
                high_img, _ = data

                # for j in range(opt.batchSize):
                #     # low_res[j] = scale(high_res_real[j])
                #     high_res_real[j] = normalize(high_res_real[j])
                # print("batch no. {} shape of input {}".format(batch_no,high_img.shape))

                for j in range(opt.batchSize):
                    inputs[j] = scale(high_img[j])
                    high_img[j] = normalize(high_img[j])

                # for i in range(opt.batchSize):
                #     inputs[i] = inputs[i][np.newaxis,:]

                # imshow(input3)
                # for j in range(opt.batchSize):
                #     high_img[j] = normalize(high_img[j])
                # high_comb=torch.cat([high_img[0], high_img[1], high_img[2], high_img[3]],0)

                # high_comb = Variable(high_comb[np.newaxis, :]).cuda()
                # imshow(high_comb.cpu().data)
                # input_comb = torch.cat([scale(input1), scale(input2), scale(input3), scale(input4)],0)
                # input_comb = input_comb[np.newaxis, :]
                if opt.cuda:
                    optimizer.zero_grad()
                    high_res_real = Variable(high_img.cuda())
                    high_res_fake = generator(Variable(inputs[0][np.newaxis,:]).cuda(),Variable(inputs[1][np.newaxis,:]).cuda(),Variable(inputs[2][np.newaxis,:]).cuda(),Variable(inputs[3][np.newaxis,:]).cuda())
                    target_real = Variable(torch.rand(opt.batchSize, 1) * 0.5 + 0.7).cuda()
                    target_fake = Variable(torch.rand(opt.batchSize, 1) * 0.3).cuda()

                    discriminator.zero_grad()
                    # import ipdb;
                    # ipdb.set_trace()

                    discriminator_loss = adversarial_criterion(
                        discriminator(Variable(inputs[0][np.newaxis, :]).cuda(), Variable(inputs[1][np.newaxis, :]).cuda(),
                                      Variable(inputs[2][np.newaxis, :]).cuda(), Variable(inputs[3][np.newaxis, :]).cuda()),
                        target_real) + \
                                         adversarial_criterion(
                                             discriminator(high_res_fake[0], high_res_fake[1], high_res_fake[2],
                                                           high_res_fake[3]), target_fake)
                    mean_discriminator_loss += discriminator_loss.data[0]

                    if phase == 'train':
                        discriminator_loss.backward(retain_graph=True)
                        optim_discriminator.step()


                    high_res_fake_cat = torch.cat([ image for image in high_res_fake ], 0)
                    # outputs = torch.chunk(high_img.cpu().data,4,0)
                    # imshow(high_res_fake[0].cpu().data)
                    generator_content_loss = content_criterion(high_res_fake_cat, high_res_real)
                    mean_generator_content_loss += generator_content_loss.data[0]
                    generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake[0],high_res_fake[1],high_res_fake[2],high_res_fake[3]), ones_const)
                    mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

                    generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
                    mean_generator_total_loss += generator_total_loss.data[0]

                    if phase == 'train':
                        generator_total_loss.backward()
                        optimizer.step()

                    if(batch_no%10==0):
                        print("phase {} batch no. {} generator_content_loss {} discriminator_loss {}".format(phase, batch_no, generator_content_loss, discriminator_loss))

                    # discriminator.zero_grad()
                    # import ipdb;
                    # ipdb.set_trace()
                    #
                    # discriminator_loss = adversarial_criterion(discriminator(Variable(inputs[0][np.newaxis,:]).cuda(),Variable(inputs[1][np.newaxis,:]).cuda(),Variable(inputs[2][np.newaxis,:]).cuda(),Variable(inputs[3][np.newaxis,:]).cuda()), target_real) + \
                    #                      adversarial_criterion(discriminator(high_res_fake[0],high_res_fake[1],high_res_fake[2],high_res_fake[3]), target_fake)
                    # mean_discriminator_loss += discriminator_loss.data[0]
                    #
                    # discriminator_loss.backward()
                    # optim_discriminator.step()
            if phase == 'train':
                imsave(high_res_fake_cat.cpu().data,train=True,epoch=epoch,image_type='fake')
                imsave(high_img, train=True, epoch=epoch, image_type='real')
                imsave(inputs, train=True, epoch=epoch, image_type='low')
                writer.add_scalar(phase + " per epoch/generator lr", optimizer.param_groups[0]['lr'], epoch + 1)
                writer.add_scalar(phase + " per epoch/discriminator lr", optim_discriminator.param_groups[0]['lr'],
                                  epoch + 1)
            else:
                imsave(high_res_fake_cat.cpu().data, train=False, epoch=epoch, image_type='fake')
                imsave(high_img, train=False, epoch=epoch, image_type='real')
                imsave(inputs, train=False, epoch=epoch, image_type='low')
            # import ipdb;
            # ipdb.set_trace()
            mssim = avg_msssim(high_res_real, high_res_fake_cat)
            psnr_val = psnr(high_res_real, high_res_fake_cat)
            scheduler_gen.step(generator_content_loss)
            scheduler_dis.step(mean_discriminator_loss)

            writer.add_scalar(phase + " per epoch/PSNR", psnr_val,
                              epoch + 1)
            writer.add_scalar(phase+" per epoch/discriminator loss", mean_discriminator_loss/len(dataloader), epoch+1)
            writer.add_scalar(phase+" per epoch/generator loss", mean_generator_total_loss/len(dataloader), epoch+1)
            writer.add_scalar("per epoch/total time taken", time.time()-curr_time, epoch+1)
            writer.add_scalar(phase+" per epoch/avg_mssim", mssim, epoch+1)


