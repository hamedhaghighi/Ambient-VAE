import os
import errno
import numpy as np
from skimage import io
import scipy
import scipy.misc
from skimage.transform import resize
from skimage.measure import compare_psnr , compare_ssim
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib
import utils_mnist as utils
import pdb

def normalize(images):
    M = np.max(images,(1,2,3))
    m = np.min(images, (1,2,3))
    Mm = np.array([val*np.ones_like(images[0]) for val  in M ])
    mm = np.array([val*np.ones_like(images[0]) for val  in m ])
    return ((images - mm)/(Mm - mm)) * 2.0 - 1.0


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_celebA(data , n):
    dt = data.test_data_list[:n]
    images = np.array([get_image(ls,108, is_crop=True, resize_w=64,is_grayscale=False) for ls in dt])
    return images

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def transform(image, npx=64, is_crop=False, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
        cropped_image = resize(cropped_image,
                                            [resize_w, resize_w])
    return np.array(cropped_image) / 127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return resize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])
def compute_pnsr_ssim(x , y):
    return compare_psnr(x, y), compare_ssim(x, y, multichannel=True)

def save_images(images, size, image_path, measure_dict, titles, x_range):
    images = [merge(img , size, x_range) for img in images]
    PSNR, SSIM = compute_pnsr_ssim(images[0], images[2])
    measure_dict['psnr'].append(PSNR)
    measure_dict['ssim'].append(SSIM)
    return imsave(images, size, image_path,measure_dict, titles)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return io.imread(path.decode('utf-8'), flatten=True).astype(np.float)
    else:
        return io.imread(path.decode('utf-8')).astype(np.float)


def imsave(images, size, path, measure_dict, titles):
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(2, 4, figure = fig)
    for i in range(len(images)):
        f_ax = fig.add_subplot(gs[0, i])
        f_ax.set_title(titles[i])
        f_ax.imshow(images[i])
    for ind , (k , v) in enumerate(measure_dict.items()):
        f_ax = fig.add_subplot(gs[1, ind])
        f_ax.set_title(k)
        f_ax.plot(np.arange(len(v)) + 1 , v) 
    fig.savefig(path , format='png')
    plt.close()

def merge(images, size, x_range):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((np.int64(h * size[0]), np.int64(w * size[1]), 3) , dtype = np.float32)
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = (image/(x_range[1] - x_range[0])) + np.abs(x_range[0])/2.0

    return img

def inverse_transform(image):
    return (image * 255.0).astype(np.uint8)

class CelebA(object):
    def __init__(self, images_path):

        self.dataname = "CelebA"
        self.dims = 64 * 64
        self.shape = [64, 64, 3]
        self.image_size = 64
        self.channel = 3
        self.images_path = images_path
        self.train_data_list, self.val_data_list, self.test_data_list = self.read_image_list_file(images_path)

    def load_celebA(self):

        # get the list of image path
        return self.train_data_list

    def load_test_celebA(self):

        # get the list of image path
        return self.val_data_list

    def read_image_list_file(self, category):
        lines = open(category + "list_eval_partition.txt")
        train_list = []
        val_list = []
        test_list = []

        for line in lines:
            name , tag = line.split(' ')
            name = category + 'celebA/' + name
            if tag[0] == '0':
                train_list += [name]
            elif tag[0] == '1':
                val_list += [name]
            else:
                test_list += [name]
        return train_list, val_list, test_list