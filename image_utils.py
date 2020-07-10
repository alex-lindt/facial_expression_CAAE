from __future__ import division
import numpy as np
from scipy.misc import imread, imresize, imsave

def load_image(image_path, image_size=64,  image_value_range=(-1, 1),  is_gray=False):
    """
    Load image from file

    @param image_path: path of the image
    @param image_size: expected size of the image
    @param image_value_range: expected pixel value range of the image
    @param is_gray: gray scale or color image
    """
    if is_gray:
        image = imread(image_path, flatten=True).astype(np.float32)
    else:
        image = imread(image_path).astype(np.float32)
    image = imresize(image, [image_size, image_size])
    image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    return image

def save_batch_images(batch_images, save_path, image_value_range=(-1,1),  size_frame=None):
    """
    Save batch of images to file

    @param batch_images: batch of images (tensor)
    @param save_path: path to save the images to
    @param image_value_range: value range of the images tensor
    @param size_frame: size of the image matrix, number of images in each row and column
    """
    # transform the pixel value to 0~1
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
    for ind, image in enumerate(images):
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
    imsave(save_path, frame)

def save_output(input_image, output, path, image_value_range = (-1,1), size_frame=[7, 10]):
    """
    Save input image and output images to file

    @param input_image: input image tensor
    @param output: network output (i.e. 49 images) for input image and 49 different valence/arousal labels
    @param path: path to save the image grid to
    """
    # Tile black background
    black_image = np.zeros((1, 96, 96, 3))
    black_image3 = np.tile(black_image, (3, 1, 1, 1))

    # Build final image from components
    input_frame = np.concatenate([black_image3,output[:4],black_image, input_image, black_image,output[4:8],black_image3,output[8:12],black_image3,output[12:]])

    # Transform into savable format
    final_image = images_to_grid(input_frame, image_value_range=image_value_range, size_frame=size_frame)

    # Save image
    imsave(path, final_image)

def images_to_grid(batch_images, image_value_range=(-1, 1), size_frame=None):
    """
    Transform images tensor to a grid of images

    @param batch_images: images tensor
    @param image_value_range: value range of the images tensor
    @param size_frame:  size of the image matrix, number of images in each row and column

    @return: images grid
    """
    # transform the pixel value to 0~1
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
    for ind, image in enumerate(images):
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
    return frame