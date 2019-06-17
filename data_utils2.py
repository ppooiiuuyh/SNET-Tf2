import os
import io
import PIL
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import numpy as  np
import sys
import random
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import copy
from functools import partial
import multiprocessing


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def normalize(images):
    return (images.astype(np.float32)/255.0)
def denormalize(images):
    return np.clip(images*255.0, a_min=0.001, a_max=254.99).astype(np.uint8)




class Trainset_Dispenser():
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.jpeg_quality = config.jpeg_quality
        self.patch_size = config.patch_size
        self.batch_size = config.batch_size
        self.images_input, self.images_label = self.load_images()

    def load_images(self):
        #pre-load images. For example, using the DIV2K dataset, 800 images will be preloaded on the memory
        file_list = glob(os.path.join(self.data_path,"*.*"))
        input_list = []
        label_list = []
        for f in tqdm(file_list):
            # read image
            label = PIL.Image.open(f).convert('RGB')

            # compress
            buffer = io.BytesIO()
            label.save(buffer, format='jpeg', quality=self.jpeg_quality)
            input = PIL.Image.open(buffer)

            # normalization and appending
            input_list.append(normalize(np.array(input)))
            label_list.append(normalize(np.array(label)))

        return input_list, label_list

    def __iter__(self):
        return self

    def __next__(self):
        patches_input = []
        patches_label = []
        for i in range(self.batch_size):
            rand_idx = random.randint(0,len(self.images_label)-1)
            crop_y = random.randint(0, self.images_label[rand_idx].shape[0] - self.patch_size-1)
            crop_x = random.randint(0, self.images_label[rand_idx].shape[1] - self.patch_size-1)
            input_patch = self.images_input[rand_idx][crop_y:crop_y+self.patch_size, crop_x:crop_x+self.patch_size]
            label_patch = self.images_label[rand_idx][crop_y:crop_y+self.patch_size, crop_x:crop_x+self.patch_size]
            patches_input.append(input_patch)
            patches_label.append(label_patch)

        patches_input = np.array(patches_input)
        patches_label = np.array(patches_label)
        return patches_input, patches_label



class Testset_Dispenser():
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.jpeg_quality = config.jpeg_quality
        self.images_input, self.images_label = self.load_images()

    def load_images(self):
        #pre-load images. For example, using the DIV2K dataset, 800 images will be preloaded on the memory
        file_list = glob(os.path.join(self.data_path,"*.*"))
        input_list = []
        label_list = []
        for f in tqdm(file_list):
            # read image
            label = PIL.Image.open(f).convert('RGB')

            # compress
            buffer = io.BytesIO()
            label.save(buffer, format='jpeg', quality=self.jpeg_quality)
            input = PIL.Image.open(buffer)

            # normalization and appending
            input_list.append(normalize(np.expand_dims(np.array(input),axis=0)))
            label_list.append(normalize(np.expand_dims(np.array(label),axis=0)))

        return input_list, label_list

    def __iter__(self):
        return zip(self.images_input,self.images_label)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_size", type=list, default=[512, 512], nargs="+", help='Image size after crop.')
    parser.add_argument("--batch_size", type=int, default=128, help='Minibatch size(global)')
    parser.add_argument("--patch_size", type=int, default=48, help='Minibatch size(global)')
    parser.add_argument("--jpeg_quality", type=int, default=20, help='Minibatch size(global)')
    parser.add_argument("--data_root_train", type=str, default='./dataset/test/Set5', help='Data root dir')
    parser.add_argument("--data_root_test", type=str, default='./dataset/test/Set5', help='Data root dir')
    parser.add_argument("--channels", type=int, default=3, help='Channel size')
    config = parser.parse_args()


    data_path = "/projects/datasets/restoration/LIVE1/all"
    data_dispenser = Trainset_Dispenser(data_path = data_path, config = config)
    for train_inputs, train_labels in data_dispenser:
        plt.imshow(train_inputs[0])
        plt.show()
        print(train_inputs.shape, train_labels.shape)
    """
    data_path = "/projects/datasets/restoration/LIVE1"
    data_dispenser = Testset_Dispenser(data_path=data_path, config=config)
    for train_input, train_label in data_dispenser:
        plt.imshow(train_input)
        plt.show()
        print(train_input.shape, train_label.shape)
    """



'''
def make_iterator_ontime(config):
    """ mapping functions """
    def mapping_function_for_paired_iterator(inputs, crop = True):
        file_names = [file_name.decode("utf-8") for file_name in inputs.numpy()]
        inputs = []
        labels = []
        for file_name in file_names :
            label = PIL.Image.open(file_name).convert('RGB')

            if crop :
                # randomly crop patch from training set
                crop_x = random.randint(0, label.width - config.patch_size)
                crop_y = random.randint(0, label.height - config.patch_size)
                label = label.crop((crop_x, crop_y, crop_x + config.patch_size, crop_y + config.patch_size))

            # additive jpeg noise
            buffer = io.BytesIO()
            label.save(buffer, format='jpeg', quality=config.jpeg_quality)
            input = PIL.Image.open(buffer)

            # normalization and appending
            inputs.append(normalize(np.array(input)))
            labels.append(normalize(np.array(label)))

        inputs = np.array(inputs)
        labels = np.array(labels)
        return inputs, labels

    """ prepare train iterator """
    # prepare paired iterator
    paired_file_names = tf.data.Dataset.list_files(os.path.normcase(os.path.join(config.data_root_train,"*.*")))
    paired_dataset = paired_file_names.batch(config.batch_size,drop_remainder=True).shuffle(config.buffer_size).repeat().prefetch(buffer_size=100)
    paired_iterator = Tensor_Iterator_Wraper(paired_dataset.__iter__(), map_func= mapping_function_for_paired_iterator)
    train_iterator = paired_iterator


    """ prepare test dataset """
    paired_file_names = tf.data.Dataset.list_files(os.path.normcase(os.path.join(config.data_root_test,"*.*")))
    paired_file_names = paired_file_names.batch(1,drop_remainder=True).shuffle(config.buffer_size)
    paired_iterator = Tensor_Dataset_Wraper(paired_file_names, map_func= partial(mapping_function_for_paired_iterator,crop=False))
    test_dataset= paired_iterator

    return train_iterator, test_dataset  #, reference_iterator
'''