# These are utility functions / classes that you probably dont need to alter.

import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def back_to_img_format(tensor):
    tensor = tensor.cpu()
    img = transforms.functional.to_pil_image(tensor / 2 + 0.5)
    return np.array(img)

def tensorshow(tensor,cmap=None):
    img = transforms.functional.to_pil_image(tensor/2+0.5)
    kek = np.array(img)
    if cmap is not None:
        plt.imshow(img,cmap=cmap)
    else:
        plt.imshow(img)

class HW3ImageFolder(torchvision.datasets.ImageFolder):
    """A version of the ImageFolder dataset class, customized for the super-resolution task"""

    def __init__(self, root, device):
        super(HW3ImageFolder, self).__init__(root, transform=None)
        self.device = device

    def prepimg(self,img):
        return (transforms.functional.to_tensor(img)-0.5)*2 # normalize tensorized image from [0,1] to [-1,+1]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (grayscale_image, color_image) where grayscale_image is the decolorized version of the color_image.
        """
        color_image,_ = super(HW3ImageFolder, self).__getitem__(index) # Image object (PIL)
        grayscale_image = torchvision.transforms.functional.to_grayscale(color_image)
        return self.prepimg(grayscale_image).to(self.device), self.prepimg(color_image).to(self.device),self.imgs[index]

def visualize_batch(inputs,preds,targets,save_path=''):
    inputs = inputs.cpu()
    preds = preds.cpu()
    targets = targets.cpu()
    plt.cla()

    bs = inputs.shape[0]
    for j in range(bs):

        sp0 = plt.subplot(3,bs,j+1)
        sp0.axis('off')
        assert(inputs[j].shape[0]==1)
        tensorshow(inputs[j],cmap='gray')
        sp1 = plt.subplot(3,bs,bs+j+1)
        sp1.axis('off')
        tensorshow(preds[j])
        sp2 = plt.subplot(3,bs,2*bs+j+1)
        sp2.axis('off')
        tensorshow(targets[j])
    if save_path is not '':
        plt.savefig(save_path)
    else:
        plt.show(block=True)


