from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import pytorch_msssim
from torch.autograd import Variable

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

def imsave(img_tensor, train, epoch, image_type, title=None):
    """Imshow for Tensor."""
    for img in  img_tensor:
        transform = transforms.Compose([transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444]),
                                        transforms.ToPILImage()])
        img = transform(img)

        # save_image(inp, 'output/high_res_real/try.png')
        # plt.imshow(inp)
        if train:
            if(image_type=='fake'):
                img.save('output/train/high_res_fake/'+str(epoch)+'.png')
            elif(image_type=='low'):
                img.save('output/train/low_res/'+str(epoch)+'.png')
            elif(image_type == 'real'):
                img.save('output/train/high_res_real/'+str(epoch)+'.png')
        else:
            if(image_type=='fake'):
                img.save('output/test/high_res_fake/'+str(epoch)+'.png')
            elif(image_type=='low'):
                img.save('output/test/low_res/'+str(epoch)+'.png')
            elif(image_type == 'real'):
                img.save('output/test/high_res_real/'+str(epoch)+'.png')
        # plt.pause(1)

def avg_msssim(real_images, fake_images):
    avg=0.0
    # for real_image,fake_image in zip(real_images,fake_images):
    avg = pytorch_msssim.msssim(Variable(real_images).cuda(), Variable(fake_images).cuda())

    return avg

def psnr(img1,img2):
	mse = torch.mean((img1-img2)**2)
	if mse ==0:
		return 100
	PIXEL_MAX = 255.0
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# def psnr(pred, gt, shave_border=0):
#     height, width = pred.shape[:2]
#     pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
#     gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
#     imdff = pred - gt
#     rmse = math.sqrt(np.mean(imdff ** 2))
#     if rmse == 0:
#         return 100
#     return 20 * math.log10(255.0 / rmse)