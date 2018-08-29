import torch
import torchvision
from torch.autograd import Variable
import time

import pytorch_ssim
from utils import normalize, imsave, avg_msssim, psnr, psnr_batch, un_normalize, angres_psnr
import torch.nn as nn
import numpy as np
from Model import FeatureExtractor
import sys

def test_multiple(generator, discriminator, opt, dataloader, scale):
    generator.load_state_dict(torch.load(opt.generatorWeights))
    discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
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
    mean_psnr = 0.0
    min_psnr = 999.0
    mean_ssim = 0.0
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
        ssim = pytorch_ssim.ssim(high_res_fake, high_res_real).data[0]
        mean_ssim += ssim
        psnr_val = psnr(un_normalize(high_res_fake), un_normalize(high_res_real))
        mean_psnr += psnr_val
        max_psnr = psnr_val if psnr_val > max_psnr else max_psnr
        min_psnr = psnr_val if psnr_val < min_psnr else min_psnr
        sys.stdout.write(
            '\rTesting batch no. [%d/%d] Generator_content_Loss: %.4f discriminator_loss %.4f psnr %.4f ssim %.4f' % (
                batch_no, len(dataloader['test']), generator_content_loss,
                discriminator_loss, psnr_val, ssim))
    print("Min psnr is: ",min_psnr)
    print("Mean psnr is: ", mean_psnr/72)
    print("Max psnr is: ", max_psnr)
    print("Mean ssim is: ", mean_ssim / 72)

# def test_angres(angres, lflists, opt):
#     angres.load_state_dict(torch.load(opt.angresWeights))
#     content_criterion = nn.MSELoss()
#     if opt.cuda:
#         angres.cuda()
#         content_criterion.cuda()
#     angres.eval()
#     if opt.cuda:
#         angres.cuda()
#         content_criterion.cuda()
#     max_psnr = 0.0
#     mean_psnr = 0.0
#     min_psnr = 999.0
#     mean_ssim = 0.0
#
#
#     total_loss = 0.0
#     count = 0
#     for lf_image in lflists:
#         i = j = -1
#         new_img = torch.FloatTensor(4, 3, opt.upSampling*opt.imageSize, opt.upSampling*opt.imageSize)
#         gt_img = torch.FloatTensor(1, 3, opt.upSampling * opt.imageSize, opt.upSampling * opt.imageSize)
#         fake_img = torch.FloatTensor(1, 3, opt.upSampling * opt.imageSize, opt.upSampling * opt.imageSize)
#         while i<14:
#             i += 1
#             j = -1
#             while j<14:
#                 count += 1
#                 j += 1
#                 img1=torch.tensor(lf_image[i][j])
#                 img2=torch.tensor(lf_image[i][j+1])
#                 img3=torch.tensor(lf_image[i+1][j])
#                 img4=torch.tensor(lf_image[i+1][j+1])
#                 gt = torch.tensor(lf_image[i + 1][j + 1])
#
#                 img1 = torch.transpose(img1, 0, 2)
#                 img2 = torch.transpose(img2, 0, 2)
#                 img3 = torch.transpose(img3, 0, 2)
#                 img4 = torch.transpose(img4, 0, 2)
#                 gt = torch.transpose(gt, 0, 2)
#
#                 new_img[0] = torch.transpose(img1, 1, 2)
#                 new_img[1] = torch.transpose(img2, 1, 2)
#                 new_img[2] = torch.transpose(img3, 1, 2)
#                 new_img[3] = torch.transpose(img4, 1, 2)
#                 gt_img[0] = torch.transpose(gt, 1, 2)
#
#                 with torch.no_grad():
#
#                     if opt.cuda:
#                         fake_img = angres(Variable(new_img[0][np.newaxis, :]).cuda(),
#                                                   Variable(new_img[1][np.newaxis, :]).cuda(),Variable(new_img[2][np.newaxis, :]).cuda(),Variable(new_img[3][np.newaxis, :]).cuda())
#                         loss = content_criterion(fake_img[0], gt_img[0].cuda())
#                         total_loss += loss
#                         imsave(fake_img.cpu().data, train=False, epoch=count, image_type='new', ang_res=True)
#                         imsave(gt_img, train=False, epoch=count, image_type='real_center', ang_res=True)
#                         # mssim = avg_msssim(gt_img.cpu().data, fake_img.cpu().data)
#                         mean_ssim += pytorch_ssim.ssim(fake_img.cpu().data, gt_img).data[0]
#                         psnr_val = angres_psnr(gt_img.cpu().data ,fake_img.cpu().data)
#                         mean_psnr += psnr_val
#                         max_psnr = psnr_val if psnr_val > max_psnr else max_psnr
#                         min_psnr = psnr_val if psnr_val < min_psnr else min_psnr
#                         sys.stdout.write('\rcontent_Loss: %.4f psnr %.4f max_psnr %.4f ' % (loss, psnr_val, max_psnr ))
#
#     print("Min psnr is: ", min_psnr)
#     print("Mean psnr is: ", mean_psnr / count)
#     print("Max psnr is: ", max_psnr)
#     print("Mean ssim is: ", mean_ssim / count)

#test for new angres model
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
    max_center_psnr = 0.0
    max_horizontaltop_psnr = 0.0
    max_horizontalbottom_psnr = 0.0
    max_verticalleft_psnr = 0.0
    max_verticalright_psnr = 0.0

    mean_center_psnr = 0.0
    mean_horizontaltop_psnr = 0.0
    mean_horizontalbottom_psnr = 0.0
    mean_verticalleft_psnr = 0.0
    mean_verticalright_psnr = 0.0

    min_center_psnr = 999.0
    min_horizontaltop_psnr = 999.0
    min_horizontalbottom_psnr = 999.0
    min_verticalleft_psnr = 999.0
    min_verticalright_psnr = 999.0

    mean_center_ssim = 0.0
    mean_horizontaltop_ssim = 0.0
    mean_horizontalbottom_ssim = 0.0
    mean_verticalleft_ssim = 0.0
    mean_verticalright_ssim = 0.0

    total_loss = 0.0
    count = 0
    for lf_image in lflists:
        i = 0
        new_img = torch.FloatTensor(4, 3, opt.upSampling*opt.imageSize, opt.upSampling*opt.imageSize)
        gt_center_img = torch.FloatTensor(1, 3, opt.upSampling * opt.imageSize, opt.upSampling * opt.imageSize)
        gt_horizontaltop_img = torch.FloatTensor(1, 3, opt.upSampling * opt.imageSize, opt.upSampling * opt.imageSize)
        gt_verticalleft_img = torch.FloatTensor(1, 3, opt.upSampling * opt.imageSize, opt.upSampling * opt.imageSize)
        gt_horizontalbottom_img = torch.FloatTensor(1, 3, opt.upSampling * opt.imageSize,
                                                    opt.upSampling * opt.imageSize)
        gt_verticalright_img = torch.FloatTensor(1, 3, opt.upSampling * opt.imageSize,
                                                 opt.upSampling * opt.imageSize)
        while i<14:
            j = 0
            while j<14:
                count += 1
                img1 = torch.tensor(lf_image[i][j])
                img2 = torch.tensor(lf_image[i][j + 1])
                img3 = torch.tensor(lf_image[i + 1][j])
                img4 = torch.tensor(lf_image[i + 1][j + 1])
                gt_center = torch.tensor(lf_image[i + 1][j + 1])
                gt_horizontaltop = torch.tensor(lf_image[i][j + 1])
                gt_verticalleft = torch.tensor(lf_image[i + 1][j])
                gt_horizontalbottom = torch.tensor(lf_image[i + 1][j + 1])
                gt_verticalright = torch.tensor(lf_image[i + 1][j + 1])

                img1 = torch.transpose(img1, 0, 2)
                img2 = torch.transpose(img2, 0, 2)
                img3 = torch.transpose(img3, 0, 2)
                img4 = torch.transpose(img4, 0, 2)
                gt_center = torch.transpose(gt_center, 0, 2)
                gt_horizontaltop = torch.transpose(gt_horizontaltop, 0, 2)
                gt_verticalleft = torch.transpose(gt_verticalleft, 0, 2)
                gt_horizontalbottom = torch.transpose(gt_horizontalbottom, 0, 2)
                gt_verticalright = torch.transpose(gt_verticalright, 0, 2)

                new_img[0] = torch.transpose(img1, 1, 2)
                new_img[1] = torch.transpose(img2, 1, 2)
                new_img[2] = torch.transpose(img3, 1, 2)
                new_img[3] = torch.transpose(img4, 1, 2)
                gt_center_img[0] = torch.transpose(gt_center, 1, 2)
                gt_horizontaltop_img[0] = torch.transpose(gt_horizontaltop, 1, 2)
                gt_verticalleft_img[0] = torch.transpose(gt_verticalleft, 1, 2)
                gt_horizontalbottom_img[0] = torch.transpose(gt_horizontalbottom, 1, 2)
                gt_verticalright_img[0] = torch.transpose(gt_verticalright, 1, 2)

                with torch.no_grad():

                    if opt.cuda:
                        fake_img = angres(Variable(new_img[0][np.newaxis, :]).cuda(),
                                                  Variable(new_img[1][np.newaxis, :]).cuda(),Variable(new_img[2][np.newaxis, :]).cuda(),Variable(new_img[3][np.newaxis, :]).cuda())
                        center_loss = content_criterion(fake_img[0], gt_center_img[0].cuda())

                        horizontaltop_loss = content_criterion(fake_img[1], gt_horizontaltop_img[0].cuda())
                        horizontalbottom_loss = content_criterion(fake_img[2], gt_horizontalbottom_img[0].cuda())
                        verticalleft_loss = content_criterion(fake_img[3], gt_verticalleft_img[0].cuda())
                        verticalright_loss = content_criterion(fake_img[4], gt_verticalright_img[0].cuda())
                        total_loss = center_loss + horizontaltop_loss + horizontalbottom_loss + verticalleft_loss + verticalright_loss

                        imsave(fake_img.cpu().data, train=False, epoch=count, image_type='new', ang_res=True)
                        imsave(gt_center_img, train=False, epoch=count, image_type='real_center', ang_res=True)
                        imsave(gt_horizontaltop_img, train=False, epoch=count, image_type='real_horizontal', ang_res=True)
                        imsave(gt_horizontalbottom_img, train=False, epoch=count, image_type='real_horizontal',
                               ang_res=True)
                        imsave(gt_verticalleft_img, train=False, epoch=count, image_type='real_vertical', ang_res=True)
                        imsave(gt_verticalright_img, train=False, epoch=count, image_type='real_vertical', ang_res=True)
                        imsave(new_img, train=False, epoch=count, image_type='actual', ang_res=True)

                        ssim_center = pytorch_ssim.ssim(fake_img[0][np.newaxis, :].cpu().data, gt_center_img)
                        psnr_center_val = angres_psnr(gt_center_img.cpu().data ,fake_img[0].cpu().data)

                        ssim_horizontaltop = pytorch_ssim.ssim(fake_img[1][np.newaxis, :].cpu().data, gt_horizontaltop_img)
                        psnr_horizontaltop_val = angres_psnr(gt_horizontaltop_img.cpu().data, fake_img[1].cpu().data)

                        ssim_horizontalbottom = pytorch_ssim.ssim(fake_img[2][np.newaxis, :].cpu().data, gt_horizontalbottom_img)
                        psnr_horizontalbottom_val = angres_psnr(gt_horizontalbottom_img.cpu().data, fake_img[2].cpu().data)

                        ssim_verticalleft = pytorch_ssim.ssim(fake_img[3][np.newaxis, :].cpu().data, gt_verticalleft_img)
                        psnr_verticalleft_val = angres_psnr(gt_verticalleft_img.cpu().data, fake_img[3].cpu().data)

                        ssim_verticalright = pytorch_ssim.ssim(fake_img[4][np.newaxis, :].cpu().data, gt_verticalright_img)
                        psnr_verticalright_val = angres_psnr(gt_verticalright_img.cpu().data, fake_img[4].cpu().data)

                        max_center_psnr = psnr_center_val if psnr_center_val > max_center_psnr else max_center_psnr

                        max_verticalleft_psnr = psnr_verticalleft_val if psnr_verticalleft_val > max_verticalleft_psnr else max_verticalleft_psnr
                        max_verticalright_psnr = psnr_verticalright_val if psnr_verticalright_val > max_verticalright_psnr else max_verticalright_psnr

                        max_horizontaltop_psnr = psnr_horizontaltop_val if psnr_horizontaltop_val > max_horizontaltop_psnr else max_horizontaltop_psnr
                        max_horizontalbottom_psnr = psnr_horizontalbottom_val if psnr_horizontalbottom_val > max_horizontalbottom_psnr else max_horizontalbottom_psnr

                        min_center_psnr = psnr_center_val if psnr_center_val < min_center_psnr else min_center_psnr

                        min_verticalleft_psnr = psnr_verticalleft_val if psnr_verticalleft_val < min_verticalleft_psnr else min_verticalleft_psnr
                        min_verticalright_psnr = psnr_verticalright_val if psnr_verticalright_val < min_verticalright_psnr else min_verticalright_psnr

                        min_horizontaltop_psnr = psnr_horizontaltop_val if psnr_horizontaltop_val < min_horizontaltop_psnr else min_horizontaltop_psnr
                        min_horizontalbottom_psnr = psnr_horizontalbottom_val if psnr_horizontalbottom_val < min_horizontalbottom_psnr else min_horizontalbottom_psnr

                        mean_center_psnr += psnr_center_val

                        mean_verticalleft_psnr += psnr_verticalleft_val
                        mean_verticalright_psnr += psnr_verticalright_val

                        mean_horizontaltop_psnr += psnr_horizontaltop_val
                        mean_horizontalbottom_psnr += psnr_horizontalbottom_val

                        mean_center_ssim += ssim_center

                        mean_verticalleft_ssim += ssim_verticalleft
                        mean_verticalright_ssim += ssim_verticalright

                        mean_horizontaltop_ssim += ssim_horizontaltop
                        mean_horizontalbottom_ssim += ssim_horizontalbottom



                        sys.stdout.write('\rcontent_Loss: %.4f psnr_center %.4f '
                                         'psnr_verticalleft %.4f psnr_horizontaltop %.4f '
                                         'psnr_verticalright %.4f psnr_horizontalbottom %.4f '
                                         'max_center_psnr %.4f max_horizontaltop_psnr %.4f '
                                         'max_verticalleft_psnr %.4f max_horizontalbottom_psnr %.4f '
                                         'max_verticalright_psnr %.4f'
                                         % (total_loss, psnr_center_val, psnr_verticalleft_val, psnr_horizontaltop_val, psnr_verticalright_val,
                                            psnr_horizontalbottom_val, max_center_psnr, max_horizontaltop_psnr, max_verticalleft_psnr,
                                            max_horizontalbottom_psnr, max_verticalright_psnr))
                j += 2
            i += 2
    print("Min center psnr is: ", min_center_psnr)
    print("Mean center psnr is: ", mean_center_psnr / count)
    print("Max center psnr is: ", max_center_psnr)
    print("Mean center ssim is: ", mean_center_ssim / count)

    print("Min hb psnr is: ", min_horizontalbottom_psnr)
    print("Mean hb psnr is: ", mean_horizontalbottom_psnr / count)
    print("Max hb psnr is: ", max_horizontalbottom_psnr)
    print("Mean hb ssim is: ", mean_horizontalbottom_ssim / count)

    print("Min ht psnr is: ", min_horizontaltop_psnr)
    print("Mean ht psnr is: ", mean_horizontaltop_psnr / count)
    print("Max ht psnr is: ", max_horizontaltop_psnr)
    print("Mean ht ssim is: ", mean_horizontaltop_ssim / count)

    print("Min vl psnr is: ", min_verticalleft_psnr)
    print("Mean vl psnr is: ", mean_verticalleft_psnr / count)
    print("Max vl psnr is: ", max_verticalleft_psnr)
    print("Mean vl ssim is: ", mean_verticalleft_ssim / count)

    print("Min vr psnr is: ", min_verticalright_psnr)
    print("Mean vr psnr is: ", mean_verticalright_psnr / count)
    print("Max vr psnr is: ", max_verticalright_psnr)
    print("Mean vr ssim is: ", mean_verticalright_ssim / count)