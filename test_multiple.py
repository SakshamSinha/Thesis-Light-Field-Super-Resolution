import torch
import torchvision
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from utils import normalize, imsave, avg_msssim, psnr, psnr_batch, un_normalize
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Model import FeatureExtractor
import sys

def test_multiple(generator, discriminator, opt, dataloader, scale):
    generator.load_state_dict(torch.load(opt.generatorWeights))
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

    curr_time = time.time()
    inputs = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0
    mean_psnr = 0.0
    mean_msssim = 0.0
    high_img = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    inputs = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    high_res_fake = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    max_psnr=0.0
    for batch_no, data in enumerate(dataloader['test']):
        high_img, _ = data
        generator.train(False)
        discriminator.train(False)
        for j in range(opt.batchSize):
            inputs[j] = scale(high_img[j])
            high_img[j] = normalize(high_img[j])

        if opt.cuda:
            high_res_real = Variable(high_img.cuda())
            high_res_fake = generator(Variable(inputs[0][np.newaxis, :]).cuda(),
                                      Variable(inputs[1][np.newaxis, :]).cuda(),
                                      Variable(inputs[2][np.newaxis, :]).cuda(),
                                      Variable(inputs[3][np.newaxis, :]).cuda())
            target_real = Variable(torch.rand(opt.batchSize, 1) * 0.5 + 0.7).cuda()
            target_fake = Variable(torch.rand(opt.batchSize, 1) * 0.3).cuda()

            discriminator_loss = adversarial_criterion(
                discriminator(Variable(inputs[0][np.newaxis, :]).cuda(), Variable(inputs[1][np.newaxis, :]).cuda(),
                              Variable(inputs[2][np.newaxis, :]).cuda(), Variable(inputs[3][np.newaxis, :]).cuda()),
                target_real) + \
                                 adversarial_criterion(
                                     discriminator(high_res_fake[0][np.newaxis, :], high_res_fake[1][np.newaxis, :], high_res_fake[2][np.newaxis, :],
                                                   high_res_fake[3][np.newaxis, :]), target_fake)
            mean_discriminator_loss += discriminator_loss.data[0]

            #high_res_fake_cat = torch.cat([image for image in high_res_fake], 0)
            fake_features = feature_extractor(high_res_fake)
            real_features = Variable(feature_extractor(high_res_real).data)

            generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006 * content_criterion(
                fake_features, real_features)
            mean_generator_content_loss += generator_content_loss.data[0]
            generator_adversarial_loss = adversarial_criterion(
                discriminator(high_res_fake[0][np.newaxis, :], high_res_fake[1][np.newaxis, :], high_res_fake[2][np.newaxis, :], high_res_fake[3][np.newaxis, :]), ones_const)
            mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

            generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
            mean_generator_total_loss += generator_total_loss.data[0]


        imsave(high_res_fake.cpu().data, train=False, epoch=batch_no, image_type='fake')
        imsave(high_img, train=False, epoch=batch_no, image_type='real')
        imsave(inputs, train=False, epoch=batch_no, image_type='low')

        mssim = avg_msssim(high_res_real, high_res_fake)
        psnr_val = psnr(un_normalize(high_res_fake), un_normalize(high_res_real))
        max_psnr = psnr_val if psnr_val>max_psnr else max_psnr
        sys.stdout.write(
            '\rTesting batch no. [%d/%d] Generator_content_Loss: %.4f discriminator_loss %.4f psnr %.4f mssim %.4f' % (
                batch_no, len(dataloader['test']), generator_content_loss,
                discriminator_loss, psnr_val, mssim))
    print("Max psnr is: ",max_psnr)

def test_angres(angres, lflists, opt):
    angres.load_state_dict(torch.load(opt.angresWeights))
    content_criterion = nn.MSELoss()
    if opt.cuda:
        angres.cuda()
        content_criterion.cuda()
    angres.eval()
    if opt.cuda:
        angres.cuda()
        content_criterion.cuda()
    max_psnr = 0.0

    total_loss = 0.0
    count = 0
    for lf_image in lflists:
        i = j = -1
        new_img = torch.FloatTensor(4, 3, opt.upSampling*opt.imageSize, opt.upSampling*opt.imageSize)
        gt_img = torch.FloatTensor(1, 3, opt.upSampling * opt.imageSize, opt.upSampling * opt.imageSize)
        while i<14:
            i += 1
            j = -1
            while j<14:
                count += 1
                j += 1
                img1=torch.tensor(lf_image[i][j])
                img2=torch.tensor(lf_image[i][j+1])
                img3=torch.tensor(lf_image[i+1][j])
                img4=torch.tensor(lf_image[i+1][j+1])
                gt = torch.tensor(lf_image[i + 1][j + 1])

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

                with torch.no_grad():

                    if opt.cuda:
                        fake_img = angres(Variable(new_img[0][np.newaxis, :]).cuda(),
                                                  Variable(new_img[1][np.newaxis, :]).cuda(),Variable(new_img[2][np.newaxis, :]).cuda(),Variable(new_img[3][np.newaxis, :]).cuda())
                        loss = content_criterion(fake_img[0], gt_img[0].cuda())
                        total_loss += loss
                        imsave(fake_img.cpu().data, train=False, epoch=count, image_type='new', ang_res=True)
                        imsave(gt_img, train=False, epoch=count, image_type='real', ang_res=True)
                        mssim = avg_msssim(gt_img, fake_img)
                        psnr_val = psnr(gt_img.cpu().data ,fake_img.cpu().data)
                        max_psnr = psnr_val if psnr_val > max_psnr else max_psnr
                        sys.stdout.write('\rcontent_Loss: %.4f psnr %.4f max_psnr %.4f mssim %.4f' % (loss, psnr_val, max_psnr, mssim))