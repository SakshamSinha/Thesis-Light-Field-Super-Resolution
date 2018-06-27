from torchvision.transforms import transforms
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

def imshow(inp, title=None):
    """Imshow for Tensor."""
    transform = transforms.Compose([transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444]),
                                    transforms.ToPILImage()])
    inp = transform(inp)

    # save_image(inp, 'output/high_res_real/try.png')
    # plt.imshow(inp)
    inp.save('output/high_res_fake/try_multiple.png')
    # plt.pause(1)