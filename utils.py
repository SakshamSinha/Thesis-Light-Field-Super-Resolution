import os
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import pytorch_msssim
from torch.autograd import Variable
import scipy.io as sio

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

def resize_highgt(img_tensor,opt):
    transform = transforms.Resize((opt.imageSize*opt.upSampling,opt.imageSize*opt.upSampling))
    for i in range(len(img_tensor)):
        img_tensor[i] = transform(img_tensor[i])
    return img_tensor

def un_normalize(img_tensor):
    transform = transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])
    for i in range(len(img_tensor)):
        img_tensor[i] = transform(img_tensor[i])
    return img_tensor

def imsave(img_tensor, train, epoch, image_type, ang_res=False, title=None, row_ind=0, column_ind=0):
    """Imshow for Tensor."""
    for i, img in  enumerate(img_tensor):
        if(ang_res==False):
            transform = transforms.Compose([transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444]),
                                        transforms.ToPILImage()])
        else:
            img = img.type(torch.ByteTensor)
            transform = transforms.Compose(
                [transforms.ToPILImage()])
        img = transform(img)

        # save_image(inp, 'output/high_res_real/try.png')
        # plt.imshow(img)
        if not ang_res:
            if train:
                if(image_type=='fake'):
                    img.save('output/train/high_res_fake/'+str(epoch)+'_'+str(i)+'.png')
                elif(image_type=='low'):
                    # plt.imshow(img)
                    img.save('output/train/low_res/'+str(epoch)+'_'+str(i)+'.png')
                elif(image_type == 'real'):
                    img.save('output/train/high_res_real/'+str(epoch)+'_'+str(i)+'.png')
            else:
                if(image_type=='fake'):
                    img.save('output/test/high_res_fake/'+str(epoch)+'_'+str(i)+'.png')
                elif(image_type=='low'):
                    img.save('output/test/low_res/'+str(epoch)+'_'+str(i)+'.png')
                elif(image_type == 'real'):
                    img.save('output/test/high_res_real/'+str(epoch)+'_'+str(i)+'.png')
        else:
            if train:
                if(image_type=='new' and i==0):
                    img.save('output/train/ang_res_fake_center/'+str(epoch)+'_'+str(i)+'.png')
                elif (image_type == 'new' and i == 1):
                    img.save('output/train/ang_res_fake_horizontaltop/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'new' and i == 2):
                    img.save('output/train/ang_res_fake_horizontalbottom/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'new' and i == 3):
                    img.save('output/train/ang_res_fake_verticalleft/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'new' and i == 4):
                    img.save('output/train/ang_res_fake_verticalright/' + str(epoch) + '_' + str(i) + '.png')

                elif (image_type == 'real_center'):
                    img.save('output/train/reals_center/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'real_horizontaltop'):
                    img.save('output/train/reals_horizontaltop/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'real_horizontalbottom'):
                    img.save('output/train/reals_horizontalbottom/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'real_verticalleft'):
                    img.save('output/train/reals_verticalleft/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'real_verticalright'):
                    img.save('output/train/reals_verticalright/' + str(epoch) + '_' + str(i) + '.png')
            else:
                if (image_type == 'new' and i == 0):
                    img.save('output/test/ang_res_fake_center/' + str(epoch) +'_'+ str(row_ind+1) + '_' + str(column_ind+1) + '.png')
                elif (image_type == 'new' and i == 1):
                    img.save('output/test/ang_res_fake_horizontaltop/' + str(epoch) +'_'+ str(row_ind) + '_' + str(column_ind+1) + '.png')
                elif (image_type == 'new' and i == 2):
                    img.save('output/test/ang_res_fake_horizontalbottom/' + str(epoch) +'_'+str(row_ind+2) + '_' + str(column_ind+1) + '.png')
                elif (image_type == 'new' and i == 3):
                    img.save('output/test/ang_res_fake_verticalleft/' + str(epoch) +'_'+str(row_ind+1) + '_' + str(column_ind) + '.png')
                elif (image_type == 'new' and i == 4):
                    img.save('output/test/ang_res_fake_verticalright/' + str(epoch) +'_'+str(row_ind+1) + '_' + str(column_ind+2) + '.png')

                if (image_type == 'actual' and i == 0):
                    img.save('output/test/corner1/' + str(epoch) +'_'+str(row_ind) + '_' + str(column_ind) + '.png')
                elif (image_type == 'actual' and i == 1):
                    img.save('output/test/corner2/' + str(epoch) +'_'+str(row_ind) + '_' + str(column_ind+1) + '.png')
                elif (image_type == 'actual' and i == 2):
                    img.save('output/test/corner3/' + str(epoch) +'_'+str(row_ind+1) + '_' + str(column_ind) + '.png')
                elif (image_type == 'actual' and i == 3):
                    img.save('output/test/corner4/' + str(epoch) +'_'+str(row_ind+1) + '_' + str(column_ind+1) + '.png')
                # elif (image_type == 'new' and i == 4):
                #     img.save('output/test/ang_res_fake_verticalright/' + str(row_ind) + '_' + str(column_ind) + '.png')

                elif (image_type == 'real_center'):
                    img.save('output/test/reals_center/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'real_horizontaltop'):
                    img.save('output/test/reals_horizontaltop/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'real_horizontalbottom'):
                    img.save('output/test/reals_horizontalbottom/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'real_verticalleft'):
                    img.save('output/test/reals_verticalleft/' + str(epoch) + '_' + str(i) + '.png')
                elif (image_type == 'real_verticalright'):
                    img.save('output/test/reals_verticalright/' + str(epoch) + '_' + str(i) + '.png')
        # plt.pause(1)

def avg_msssim(real_images, fake_images):
    avg=0.0
    # for real_image,fake_image in zip(real_images,fake_images):
    avg = pytorch_msssim.msssim(Variable(real_images).cuda(), Variable(fake_images).cuda())

    return avg

def mse(img1,img2):
    # import ipdb
    # ipdb.set_trace()
    mse_val = ((img2-img1)**2).data.mean()
    if mse_val == 0:
        return 100
    PIXEL_MAX = 255.0
    return mse_val

def psnr(img1,img2):
    mse = torch.mean((img1-img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(1.0/math.sqrt(mse))
    # return 10 * math.log10(1.0 / mse)

def angres_psnr(img1,img2):
    mse = torch.mean((img1-img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    # return 10 * math.log10(255.0/math.sqrt(mse))
    return 10 * math.log10(255 **2 / mse)

def psnr_batch(ten1, ten2):
    avg_mse=0.0
    for fake_img,real_img in zip(ten1,ten2):
        avg_mse += mse(fake_img,real_img)
    return 10 * math.log10(255.0/(avg_mse/4.0))

# def psnr(img1,img2):
#     # import ipdb
#     # ipdb.set_trace()
#     mse = torch.mean((img1-img2)**2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# def psnr(pred, gt, shave_border=0):
#     height, width = pred.shape[:2]
#     pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
#     gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
#     imdff = pred - gt
#     rmse = math.sqrt(np.mean(imdff ** 2))
#     if rmse == 0:
#         return 100
#     return 20 * math.log10(255.0 / rmse)

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def save_matlab_files(epoch):
    pass

def get_matlab_lf(phase = 'train'):
    lfimages=[]
    if phase=='train':
        for filename in os.listdir('data/lfimages_train'):
            print(filename)
            traindata = sio.loadmat('data/lfimages_train/'+filename)
            lfimages.append(traindata['LF_lfname'])
        return lfimages
    else:
        for filename in os.listdir('data/lfimages_test'):
            print(filename)
            traindata = sio.loadmat('data/lfimages_test/'+filename)
            lfimages.append(traindata['LF_lfname'])
        return lfimages