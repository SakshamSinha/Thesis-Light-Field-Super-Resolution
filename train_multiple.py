import torch
import torchvision
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

from torchvision.transforms import transforms

from utils import normalize, imsave, avg_msssim, psnr, psnr_batch, un_normalize
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Model import FeatureExtractor
import sys
import torch.nn.functional as F


def bilinear_upsampling(opt, dataloader, scale):
    for batch_no, data in enumerate(dataloader['test']):
        high_img, _ = data
        inputs = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
        for j in range(opt.batchSize):
            inputs[j] = scale(high_img[j])
            high_img[j] = normalize(high_img[j])
        outputs = F.upsample(inputs,scale_factor=opt.upSampling,mode='bilinear',align_corners=True)
        transform = transforms.Compose([transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444]),
                                        transforms.ToPILImage()])
        transform(outputs[0]).save('output/train/bilinear_fake/' + str(batch_no)  + '.png')
        transform(high_img[0]).save('output/train/bilinear_real/' + str(batch_no) + '.png')

        # for output, himg in zip (outputs, high_img):
        #     psnr_val = psnr(output,himg)
            #mssim = avg_msssim(himg, output)
        print(psnr(un_normalize(outputs),un_normalize(high_img)))

# def train_multiple(generator, discriminator, opt, dataloader, writer, scale):
#     feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
#
#     content_criterion = nn.MSELoss()
#     adversarial_criterion = nn.BCELoss()
#
#     ones_const = Variable(torch.ones(opt.batchSize, 1))
#
#     if opt.cuda:
#         generator.cuda()
#         discriminator.cuda()
#         feature_extractor.cuda()
#         content_criterion.cuda()
#         adversarial_criterion.cuda()
#         ones_const = ones_const.cuda()
#
#     optimizer = optim.RMSprop(generator.parameters(), lr=opt.generatorLR)
#     optim_discriminator = optim.RMSprop(discriminator.parameters(), lr=opt.discriminatorLR)
#     scheduler_gen = ReduceLROnPlateau(optimizer, 'min', factor = 0.7,patience = 10, verbose = True)
#     scheduler_dis = ReduceLROnPlateau(optim_discriminator, 'min', factor = 0.7,patience = 10, verbose = True)
#     curr_time = time.time()
#     inputs = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
#
#     #pretraining
#     for epoch in range(2):
#         mean_generator_content_loss = 0.0
#
#         inputs= torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
#
#         for batch_no, data in enumerate(dataloader['train']):
#             high_img, _ = data
#
#
#
#             for j in range(opt.batchSize):
#                 inputs[j] = scale(high_img[j])
#                 high_img[j] = normalize(high_img[j])
#
#
#             if opt.cuda:
#                 optimizer.zero_grad()
#                 high_res_real = Variable(high_img.cuda())
#                 high_res_fake = generator(Variable(inputs[0][np.newaxis,:]).cuda(),Variable(inputs[1][np.newaxis,:]).cuda(),Variable(inputs[2][np.newaxis,:]).cuda(),Variable(inputs[3][np.newaxis,:]).cuda())
#                 target_real = Variable(torch.rand(opt.batchSize, 1) * 0.5 + 0.7).cuda()
#                 target_fake = Variable(torch.rand(opt.batchSize, 1) * 0.3).cuda()
#                 generator_content_loss = content_criterion(high_res_fake, high_res_real)
#                 mean_generator_content_loss += generator_content_loss.data[0]
#                 generator_content_loss.backward()
#                 optimizer.step()
#
#                 sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch, 2, batch_no, len(dataloader['train']), generator_content_loss.data[0]))
#
#     #training
#     gen_iterations = 0
#     for epoch in range(opt.nEpochs):
#         for phase in ['train', 'test']:
#             if phase == 'test':
#                 generator.train(False)
#                 discriminator.train(False)
#             else:
#                 generator.train(True)
#                 discriminator.train(True)
#
#             mean_generator_content_loss = 0.0
#             mean_generator_adversarial_loss = 0.0
#             mean_generator_total_loss = 0.0
#             mean_discriminator_loss = 0.0
#             mean_psnr = 0.0
#             mean_msssim = 0.0
#             high_img = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
#             inputs= torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
#             if phase == 'train':
#                 for p in discriminator.parameters():  # reset requires_grad
#                     p.requires_grad = True
#
#                 if gen_iterations < 25 or gen_iterations % 25 == 0:
#                     Diters = 100
#                 else:
#                     Diters = 10
#                 j = 0
#                 while(j<Diters):
#                     j+=1
#                     for batch_no, data in enumerate(dataloader[phase]):
#                         high_img, _ = data
#
#                         for i in range(opt.batchSize):
#                             inputs[i] = scale(high_img[i])
#                             high_img[i] = normalize(high_img[i])
#
#
#                         if opt.cuda:
#                             high_res_real = Variable(high_img.cuda())
#                             high_res_fake = generator(Variable(inputs[0][np.newaxis,:]).cuda(),Variable(inputs[1][np.newaxis,:]).cuda(),Variable(inputs[2][np.newaxis,:]).cuda(),Variable(inputs[3][np.newaxis,:]).cuda())
#                             target_real = Variable(torch.rand(opt.batchSize, 1) * 0.5 + 0.7).cuda()
#                             target_fake = Variable(torch.rand(opt.batchSize, 1) * 0.3).cuda()
#
#                             discriminator_loss = adversarial_criterion(
#                                 discriminator(Variable(inputs[0][np.newaxis, :]).cuda(), Variable(inputs[1][np.newaxis, :]).cuda(),
#                                               Variable(inputs[2][np.newaxis, :]).cuda(), Variable(inputs[3][np.newaxis, :]).cuda()),
#                                 target_real) + \
#                                                  adversarial_criterion(
#                                                      discriminator(high_res_fake[0][np.newaxis, :], high_res_fake[1][np.newaxis, :], high_res_fake[2][np.newaxis, :],
#                                                                    high_res_fake[3][np.newaxis, :]), target_fake)
#                             mean_discriminator_loss += discriminator_loss.data[0]
#
#                             optim_discriminator.zero_grad()
#                             discriminator_loss.backward(retain_graph=True)
#                             optim_discriminator.step()
#
#                             if (batch_no % 10 == 0):
#                                 # print("phase {} batch no. {} generator_content_loss {} discriminator_loss {}".format(phase, batch_no, generator_content_loss, discriminator_loss))
#                                 sys.stdout.write(
#                                     '\rphase [%s] Diter [%d/%d] batch no. [%d/%d] discriminator_loss %.4f' % (
#                                         phase, j, Diters, batch_no, len(dataloader[phase]),discriminator_loss))
#
#             for batch_no, data in enumerate(dataloader[phase]):
#                 high_img, _ = data
#
#                 for i in range(opt.batchSize):
#                     inputs[i] = scale(high_img[i])
#                     high_img[i] = normalize(high_img[i])
#
#                 high_res_real = Variable(high_img.cuda())
#                 high_res_fake = generator(Variable(inputs[0][np.newaxis, :]).cuda(),
#                                           Variable(inputs[1][np.newaxis, :]).cuda(),
#                                           Variable(inputs[2][np.newaxis, :]).cuda(),
#                                           Variable(inputs[3][np.newaxis, :]).cuda())
#                 if phase == 'train':
#                     for p in discriminator.parameters():  # reset requires_grad
#                         p.requires_grad = False
#
#                 fake_features = feature_extractor(high_res_fake)
#                 real_features = Variable(feature_extractor(high_res_real).data)
#
#                 generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
#                 mean_generator_content_loss += generator_content_loss.data[0]
#                 generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake[0][np.newaxis, :],high_res_fake[1][np.newaxis, :],high_res_fake[2][np.newaxis, :],high_res_fake[3][np.newaxis, :]), ones_const)
#                 mean_generator_adversarial_loss += generator_adversarial_loss.data[0]
#
#                 generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
#                 mean_generator_total_loss += generator_total_loss.data[0]
#
#                 if phase == 'train':
#                     optimizer.zero_grad()
#                     generator_total_loss.backward()
#                     optimizer.step()
#
#                 if(batch_no%10==0):
#                     # print("phase {} batch no. {} generator_content_loss {} discriminator_loss {}".format(phase, batch_no, generator_content_loss, discriminator_loss))
#                     sys.stdout.write('\rphase [%s] epoch [%d/%d] batch no. [%d/%d] Generator_content_Loss: %.4f discriminator_loss %.4f' % (
#                     phase, epoch, opt.nEpochs, batch_no, len(dataloader[phase]), generator_content_loss, generator_adversarial_loss))
#
#
#             if phase == 'train':
#                 imsave(high_res_fake.cpu().data,train=True,epoch=epoch,image_type='fake')
#                 imsave(high_img, train=True, epoch=epoch, image_type='real')
#                 imsave(inputs, train=True, epoch=epoch, image_type='low')
#                 writer.add_scalar(phase + " per epoch/generator lr", optimizer.param_groups[0]['lr'], epoch + 1)
#                 writer.add_scalar(phase + " per epoch/discriminator lr", optim_discriminator.param_groups[0]['lr'],
#                                   epoch + 1)
#                 scheduler_gen.step(mean_generator_total_loss / len(dataloader[phase]))
#                 scheduler_dis.step(mean_discriminator_loss / len(dataloader[phase]))
#             else:
#                 imsave(high_res_fake.cpu().data, train=False, epoch=epoch, image_type='fake')
#                 imsave(high_img, train=False, epoch=epoch, image_type='real')
#                 imsave(inputs, train=False, epoch=epoch, image_type='low')
#             # import ipdb;
#             # ipdb.set_trace()
#             mssim = avg_msssim(high_res_real, high_res_fake)
#             psnr_val = psnr(un_normalize(high_res_real), un_normalize(high_res_fake))
#
#             writer.add_scalar(phase + " per epoch/PSNR", psnr_val,
#                               epoch + 1)
#             writer.add_scalar(phase+" per epoch/discriminator loss", mean_discriminator_loss/len(dataloader[phase]), epoch+1)
#             writer.add_scalar(phase+" per epoch/generator loss", mean_generator_total_loss/len(dataloader[phase]), epoch+1)
#             writer.add_scalar("per epoch/total time taken", time.time()-curr_time, epoch+1)
#             writer.add_scalar(phase+" per epoch/avg_mssim", mssim, epoch+1)
#         gen_iterations += 1
#         # Do checkpointing
#         torch.save(generator.state_dict(), '%s/generator_final.pth' % opt.out)
#         torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % opt.out)


def train_multiple(generator, discriminator, opt, dataloader, writer, scale):
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))

    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    ones_const = Variable(torch.ones(opt.batchSize, 1))

    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        feature_extractor.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        ones_const = ones_const.cuda()

    optimizer = optim.Adam(generator.parameters(), lr=opt.generatorLR)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)
    scheduler_gen = ReduceLROnPlateau(optimizer, 'min', factor = 0.7,patience = 10, verbose = True)
    scheduler_dis = ReduceLROnPlateau(optim_discriminator, 'min', factor = 0.7,patience = 10, verbose = True)
    curr_time = time.time()
    inputs = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

    #pretraining
    for epoch in range(2):
        mean_generator_content_loss = 0.0

        inputs= torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

        for batch_no, data in enumerate(dataloader['train']):
            high_img, _ = data



            for j in range(opt.batchSize):
                inputs[j] = scale(high_img[j])
                high_img[j] = normalize(high_img[j])


            if opt.cuda:
                optimizer.zero_grad()
                high_res_real = Variable(high_img.cuda())
                high_res_fake = generator(Variable(inputs[0][np.newaxis,:]).cuda(),Variable(inputs[1][np.newaxis,:]).cuda(),Variable(inputs[2][np.newaxis,:]).cuda(),Variable(inputs[3][np.newaxis,:]).cuda())
                target_real = Variable(torch.rand(opt.batchSize, 1) * 0.5 + 0.7).cuda()
                target_fake = Variable(torch.rand(opt.batchSize, 1) * 0.3).cuda()
                generator_content_loss = content_criterion(high_res_fake, high_res_real)
                mean_generator_content_loss += generator_content_loss.data[0]
                generator_content_loss.backward()
                optimizer.step()

                sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch, 2, batch_no, len(dataloader['train']), generator_content_loss.data[0]))

    #training
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

            for batch_no, data in enumerate(dataloader[phase]):
                high_img, _ = data

                for j in range(opt.batchSize):
                    inputs[j] = scale(high_img[j])
                    high_img[j] = normalize(high_img[j])


                if opt.cuda:
                    optimizer.zero_grad()
                    high_res_real = Variable(high_img.cuda())
                    high_res_fake = generator(Variable(inputs[0][np.newaxis,:]).cuda(),Variable(inputs[1][np.newaxis,:]).cuda(),Variable(inputs[2][np.newaxis,:]).cuda(),Variable(inputs[3][np.newaxis,:]).cuda())
                    target_real = Variable(torch.rand(opt.batchSize, 1) * 0.5 + 0.7).cuda()
                    target_fake = Variable(torch.rand(opt.batchSize, 1) * 0.3).cuda()

                    discriminator.zero_grad()

                    discriminator_loss = adversarial_criterion(
                        discriminator(Variable(inputs[0][np.newaxis, :]).cuda(), Variable(inputs[1][np.newaxis, :]).cuda(),
                                      Variable(inputs[2][np.newaxis, :]).cuda(), Variable(inputs[3][np.newaxis, :]).cuda()),
                        target_real) + \
                                         adversarial_criterion(
                                             discriminator(high_res_fake[0][np.newaxis, :], high_res_fake[1][np.newaxis, :], high_res_fake[2][np.newaxis, :],
                                                           high_res_fake[3][np.newaxis, :]), target_fake)
                    mean_discriminator_loss += discriminator_loss.data[0]

                    if phase == 'train':
                        discriminator_loss.backward(retain_graph=True)
                        optim_discriminator.step()


                    #high_res_fake_cat = torch.cat([ image for image in high_res_fake ], 0)
                    fake_features = feature_extractor(high_res_fake)
                    real_features = Variable(feature_extractor(high_res_real).data)
                    # outputs = torch.chunk(high_img.cpu().data,4,0)
                    # imshow(high_res_fake[0].cpu().data)
                    generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
                    mean_generator_content_loss += generator_content_loss.data[0]
                    generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake[0][np.newaxis, :],high_res_fake[1][np.newaxis, :],high_res_fake[2][np.newaxis, :],high_res_fake[3][np.newaxis, :]), ones_const)
                    mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

                    generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
                    mean_generator_total_loss += generator_total_loss.data[0]

                    if phase == 'train':
                        generator_total_loss.backward()
                        optimizer.step()

                    if(batch_no%10==0):
                        # print("phase {} batch no. {} generator_content_loss {} discriminator_loss {}".format(phase, batch_no, generator_content_loss, discriminator_loss))
                        sys.stdout.write('\rphase [%s] epoch [%d/%d] batch no. [%d/%d] Generator_content_Loss: %.4f discriminator_loss %.4f' % (
                        phase, epoch, opt.nEpochs, batch_no, len(dataloader[phase]), generator_content_loss, discriminator_loss))


            if phase == 'train':
                imsave(high_res_fake.cpu().data,train=True,epoch=epoch,image_type='fake')
                imsave(high_img, train=True, epoch=epoch, image_type='real')
                imsave(inputs, train=True, epoch=epoch, image_type='low')
                writer.add_scalar(phase + " per epoch/generator lr", optimizer.param_groups[0]['lr'], epoch + 1)
                writer.add_scalar(phase + " per epoch/discriminator lr", optim_discriminator.param_groups[0]['lr'],
                                  epoch + 1)
                scheduler_gen.step(mean_generator_total_loss / len(dataloader[phase]))
                scheduler_dis.step(mean_discriminator_loss / len(dataloader[phase]))
            else:
                imsave(high_res_fake.cpu().data, train=False, epoch=epoch, image_type='fake')
                imsave(high_img, train=False, epoch=epoch, image_type='real')
                imsave(inputs, train=False, epoch=epoch, image_type='low')
            # import ipdb;
            # ipdb.set_trace()
            mssim = avg_msssim(high_res_real, high_res_fake)
            psnr_val = psnr(un_normalize(high_res_real), un_normalize(high_res_fake))

            writer.add_scalar(phase + " per epoch/PSNR", psnr_val,
                              epoch + 1)
            writer.add_scalar(phase+" per epoch/discriminator loss", mean_discriminator_loss/len(dataloader[phase]), epoch+1)
            writer.add_scalar(phase+" per epoch/generator loss", mean_generator_total_loss/len(dataloader[phase]), epoch+1)
            writer.add_scalar("per epoch/total time taken", time.time()-curr_time, epoch+1)
            writer.add_scalar(phase+" per epoch/avg_mssim", mssim, epoch+1)
        # Do checkpointing
        torch.save(generator.state_dict(), '%s/generator_final.pth' % opt.out)
        torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % opt.out)

# def train_angres(AngRes, lflists, opt, writer):
#     content_criterion = nn.MSELoss()
#     fake_ang_res = torch.FloatTensor(opt.batchSize, 3, opt.upSampling*opt.imageSize, opt.upSampling*opt.imageSize)
#     optimizer = optim.Adam(AngRes.parameters(), lr=0.0001)
#     scheduler_angres = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
#     curr_time = time.time()
#     if opt.cuda:
#         AngRes.cuda()
#         content_criterion.cuda()
#     for epoch in range(opt.nEpochs):
#         for phase in ['train', 'test']:
#             total_loss = 0.0
#             if phase == 'test':
#                 continue
#                 AngRes.train(False)
#             else:
#                 AngRes.train(True)
#
#
#             new_img = torch.FloatTensor(opt.batchSize, 3, opt.upSampling*opt.imageSize, opt.upSampling*opt.imageSize)
#             inputs = torch.FloatTensor(opt.batchSize, 3, opt.upSampling*opt.imageSize, opt.upSampling*opt.imageSize)
#
#             for batch_no, data in enumerate(dataloader[phase]):
#                 new_img, _ = data
#
#                 for j in range(opt.batchSize):
#                     # inputs[j] = normalize(new_img[j])
#                     inputs[j] = new_img[j]
#
#                 if opt.cuda:
#                     fake_img = AngRes(Variable(inputs[0][np.newaxis, :]).cuda(),
#                                               Variable(inputs[2][np.newaxis, :]).cuda())
#                     loss = content_criterion(fake_img[0], inputs[1].cuda())
#                     total_loss += loss
#                     if phase == 'train':
#                         AngRes.zero_grad()
#                         loss.backward()
#                         optimizer.step()
#
#                     if (batch_no % 10 == 0):
#                         sys.stdout.write(
#                             '\rphase [%s] epoch [%d/%d] batch no. [%d/%d] content_Loss: %.4f' % (
#                                 phase, epoch, opt.nEpochs, batch_no, len(dataloader[phase]), loss))
#
#             if phase == 'train':
#                 imsave(fake_img.cpu().data, train=True, epoch=epoch, image_type='new', ang_res=True)
#                 imsave(inputs, train=True, epoch=epoch, image_type='real', ang_res=True)
#                 writer.add_scalar(phase + " per epoch/angres lr", optimizer.param_groups[0]['lr'], epoch + 1)
#
#             else:
#                 imsave(fake_img.cpu().data, train=False, epoch=epoch, image_type='new', ang_res=True)
#                 imsave(inputs, train=False, epoch=epoch, image_type='real', ang_res=True)
#             scheduler_angres.step(total_loss / len(dataloader[phase]))
#             writer.add_scalar(phase + " per epoch/angres loss", total_loss / len(dataloader[phase]),
#                               epoch + 1)
#             writer.add_scalar("per epoch/total time taken", time.time() - curr_time, epoch + 1)
#
#         # Do checkpointing
#         torch.save(AngRes.state_dict(), '%s/AngRes_final.pth' % opt.out)

def train_angres(AngRes, lflists, opt, writer):
    content_criterion = nn.MSELoss()
    fake_ang_res = torch.FloatTensor(4, 3, opt.upSampling*opt.imageSize, opt.upSampling*opt.imageSize)
    optimizer = optim.Adam(AngRes.parameters(), lr=0.0001)
    scheduler_angres = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
    curr_time = time.time()
    if opt.cuda:
        AngRes.cuda()
        content_criterion.cuda()
    AngRes.train(True)
    for epoch in range(opt.nEpochs):
        total_loss = 0.0
        count = 0
        for lf_image in lflists:
            i = j = -1
            gt_img = torch.FloatTensor(1, 3, opt.upSampling * opt.imageSize, opt.upSampling * opt.imageSize)
            new_img = torch.FloatTensor(4, 3, opt.upSampling*opt.imageSize, opt.upSampling*opt.imageSize)

            while i<14:
                i += 1
                j = -1
                while j<14:
                    j += 1
                    img1=torch.tensor(lf_image[i][j])
                    img2=torch.tensor(lf_image[i][j+2])
                    img3=torch.tensor(lf_image[i+2][j])
                    img4=torch.tensor(lf_image[i+2][j+2])
                    gt=torch.tensor(lf_image[i+1][j+1])

                    img1 = torch.transpose(img1, 0, 2)
                    img2 = torch.transpose(img2, 0, 2)
                    img3 = torch.transpose(img3, 0, 2)
                    img4 = torch.transpose(img4, 0, 2)
                    gt = torch.transpose(gt, 0, 2)

                    new_img[0] = torch.transpose(img1, 1, 2)
                    new_img[1] = torch.transpose(img2, 1, 2)
                    new_img[2] = torch.transpose(img3, 1, 2)
                    new_img[3] = torch.transpose(img4, 1, 2)
                    gt_img[0] = torch.transpose(gt, 1, 2)

                    # transform = transforms.Compose(
                    #     [transforms.ToPILImage()])
                    # img = transform(gt_img[0])
                    # img.save('output/test.png')

                    if opt.cuda:
                        fake_img = AngRes(Variable(new_img[0][np.newaxis, :]).cuda(),
                                                  Variable(new_img[1][np.newaxis, :]).cuda(),Variable(new_img[2][np.newaxis, :]).cuda(),Variable(new_img[3][np.newaxis, :]).cuda())
                        loss = content_criterion(fake_img[0], gt_img[0].cuda())
                        total_loss += loss
                        AngRes.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if (j == 14):
                            sys.stdout.write(
                                '\repoch [%d/%d] content_Loss: %.4f' % (
                                   epoch, opt.nEpochs, loss))

            imsave(fake_img.cpu().data, train=True, epoch=count, image_type='new', ang_res=True)
            imsave(gt_img, train=True, epoch=count, image_type='real', ang_res=True)
            writer.add_scalar(" per epoch/angres lr", optimizer.param_groups[0]['lr'], epoch + 1)

                # else:
                #     imsave(fake_img.cpu().data, train=False, epoch=epoch, image_type='new', ang_res=True)
                #     imsave(inputs, train=False, epoch=epoch, image_type='real', ang_res=True)
        scheduler_angres.step(total_loss / len(lflists))
        writer.add_scalar(" per epoch/angres loss", total_loss / len(lflists),
                          epoch + 1)
        writer.add_scalar("per epoch/total time taken", time.time() - curr_time, epoch + 1)

        # Do checkpointing
        torch.save(AngRes.state_dict(), '%s/AngRes_final.pth' % opt.out)
