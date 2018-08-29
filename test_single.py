import sys
import torch
import numpy as np
from torch.autograd import Variable
import time
from utils import normalize, imsave, avg_msssim, psnr, un_normalize
import torch.nn as nn

#modelv1 testing
def test_single(generator, discriminator, opt, dataloader, scale):
    generator.load_state_dict(torch.load(opt.generatorWeights))
    discriminator.load_state_dict(torch.load(opt.discriminatorWeights))

    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    ones_const = Variable(torch.ones(1, 1))

    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        ones_const = ones_const.cuda()


    curr_time = time.time()

    # for epoch in range(opt.nEpochs):
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0
    high_res_fake = 0
    for batch_no, data in enumerate(dataloader['test']):
        high_img, _ = data
        generator.train(False)
        discriminator.train(False)

        print("batch no. {} shape of input {}".format(batch_no, high_img.shape))
        input1 = high_img[0, :, :, :]
        input2 = high_img[1, :, :, :]
        input3 = high_img[2, :, :, :]
        input4 = high_img[3, :, :, :]
        inputs = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
        # imshow(input3)
        for j in range(opt.batchSize):
            inputs[j] = scale(high_img[j])
            high_img[j] = normalize(high_img[j])
        high_comb = torch.cat([high_img[0], high_img[1], high_img[2], high_img[3]], 0)

        high_comb = Variable(high_comb[np.newaxis, :]).cuda()
        # imshow(high_comb.cpu().data)
        input_comb = torch.cat([scale(input1), scale(input2), scale(input3), scale(input4)], 0)
        # inputs = [scale(input1), scale(input2), scale(input3), scale(input4)]
        input_comb = input_comb[np.newaxis, :]
        if opt.cuda:
            # optimizer.zero_grad()
            high_res_real = Variable(high_img.cuda())
            high_res_fake = generator(Variable(input_comb).cuda())
            target_real = Variable(torch.rand(1, 1) * 0.5 + 0.7).cuda()
            target_fake = Variable(torch.rand(1, 1) * 0.3).cuda()

            outputs = torch.chunk(high_res_fake, 4, 1)
            outputs = torch.cat([outputs[0], outputs[1], outputs[2], outputs[3]], 0)
            # imshow(outputs[0])
            generator_content_loss = content_criterion(high_res_fake, high_comb)
            mean_generator_content_loss += generator_content_loss.data[0]
            generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
            mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

            generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
            mean_generator_total_loss += generator_total_loss.data[0]

            discriminator_loss = adversarial_criterion(discriminator(high_comb), target_real) + \
                                 adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
            mean_discriminator_loss += discriminator_loss.data[0]
            psnr_val = psnr(un_normalize(outputs), un_normalize(high_res_real))
            print(psnr_val)

            imsave(outputs.cpu().data, train=False, epoch=batch_no, image_type='fake')
            imsave(high_img, train=False, epoch=batch_no, image_type='real')
            imsave(inputs, train=False, epoch=batch_no, image_type='low')


#testing of simple CNN model
def test_firstmodel(generator, opt, dataloader, writer, scale):
    content_criterion = nn.MSELoss()

    ones_const = Variable(torch.ones(1, 1))

    if opt.cuda:
        generator.cuda()
        content_criterion.cuda()

    curr_time = time.time()

    for epoch in range(opt.nEpochs):
        mean_generator_content_loss = 0.0
        mean_generator_total_loss = 0.0

        high_res_fake = 0

        for batch_no, data in enumerate(dataloader['test']):
            high_img, _ = data
            generator.train(False)

            input1 = high_img[0, :, :, :]
            input2 = high_img[1, :, :, :]
            input3 = high_img[2, :, :, :]
            input4 = high_img[3, :, :, :]
            # imshow(input3)

            for j in range(opt.batchSize):
                high_img[j] = normalize(high_img[j])
            high_comb = torch.cat([high_img[0], high_img[1], high_img[2], high_img[3]], 0)

            high_comb = Variable(high_comb[np.newaxis, :]).cuda()
            # imshow(high_comb.cpu().data)
            input_comb = torch.cat([scale(input1), scale(input2), scale(input3), scale(input4)], 0)
            input_comb = input_comb[np.newaxis, :]

            if opt.cuda:
                high_res_real = Variable(high_img.cuda())
                high_res_fake = generator(Variable(input_comb).cuda())

                outputs = torch.chunk(high_res_fake, 4, 1)
                outputs = torch.cat([outputs[0], outputs[1], outputs[2], outputs[3]], 0)
                # imshow(outputs[0])
                generator_content_loss = content_criterion(high_res_fake, high_comb)
                mean_generator_content_loss += generator_content_loss.data[0]

                generator_total_loss = generator_content_loss
                mean_generator_total_loss += generator_total_loss.data[0]

                if (batch_no % 10 == 0):
                    #                         # print("phase {} batch no. {} generator_content_loss {} discriminator_loss {}".format(phase, batch_no, generator_content_loss, discriminator_loss))
                    sys.stdout.write(
                        '\r epoch [%d/%d] batch no. [%d/%d] Generator_content_Loss: %.4f ' % (
                            epoch, opt.nEpochs, batch_no, len(dataloader['test']), generator_content_loss))


        mssim = avg_msssim(high_res_real, outputs)
        psnr_val = psnr(un_normalize(high_res_real), un_normalize(outputs))

        writer.add_scalar("test per epoch/PSNR", psnr_val,
                          epoch + 1)
        # writer.add_scalar(phase+" per epoch/discriminator loss", mean_discriminator_loss/len(dataloader[phase]), epoch+1)
        writer.add_scalar("test per epoch/generator loss", mean_generator_total_loss / len(dataloader[phase]),
                          epoch + 1)
        writer.add_scalar("per epoch/total time taken", time.time() - curr_time, epoch + 1)
        writer.add_scalar("test per epoch/avg_mssim", mssim, epoch + 1)

        torch.save(generator.state_dict(), '%s/generator_firstfinal.pth' % opt.out)
