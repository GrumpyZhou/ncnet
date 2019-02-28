import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

def plot_image(im,batch_idx=0, mean_std=True, return_im=False):
    if im.dim()==4:
        im=im[batch_idx,:,:,:]
    if mean_std:
        mean=Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3,1,1))
        std=Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3,1,1))
        if im.is_cuda:
            mean=mean.cuda()
            std=std.cuda()
        im=im.mul(std).add(mean)*255.0
    im=im.permute(1,2,0).data.cpu().numpy().astype(np.uint8)
    if return_im:
        return im
    plt.imshow(im)
    plt.show()

def save_plot(filename):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename, bbox_inches = 'tight',
        pad_inches = 0)
    
    
def plot_matches(src, tgt, matches, inliers=None, Npts=None, lines=False):
    from PIL import Image

    # Read images and resize
    I1 = Image.open(src)
    I2 = Image.open(tgt)
    w1, h1 = I1.size
    w2, h2 = I2.size

    if h1 <= h2:
        scale1 = 1;
        scale2 = h1/h2
        w2 = int(scale2 * w2)
        I2 = I2.resize((w2, h1))
    else:
        scale1 = h2/h1
        scale2 = 1
        w1 = int(scale1 * w1)
        I1 = I1.resize((w1, h2))
    catI = np.concatenate([np.array(I1), np.array(I2)], axis=1)

    # Load all matches
    match_num = matches.shape[0]
    if inliers is None:
        Npts = Npts if Npts < match_num else match_num
        inliers = range(Npts) # Everthing as an inlier
    else:
        if Npts is not None and Npts < inliers.shape[0]:
            inliers = inliers[:Npts]

    x1 = scale1*matches[inliers, 0]
    y1 = scale1*matches[inliers, 1]
    x2 = scale2*matches[inliers, 2] + w1
    y2 = scale2*matches[inliers, 3]
    c = np.random.rand(inliers.shape[0], 3) 

    
    # Plot images and matches
    plt.imshow(catI)
    ax = plt.gca()
    for i, inid in enumerate(inliers):
        # Plot
        ax = plt.gca()
        ax.add_artist(plt.Circle((x1[i], y1[i]), radius=3, color=c[i,:]))
        ax.add_artist(plt.Circle((x2[i], y2[i]), radius=3, color=c[i,:]))
        if lines:
            plt.plot([x1, x2], [y1, y2], c=c[i,:], linestyle='-', linewidth=0.2)
    plt.gcf().set_dpi(300)
    plt.show()