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



def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def normalize(images):
    return (images.astype(np.float32)/255.0)
def denormalize(images):
    return (images*255.0).astype(np.uint8)



class Tensor_Iterator_Wraper():
    def __init__(self,iterator, map_func = (lambda x : x) ):
        self.iterator = iterator
        self.map_func = map_func

    def __iter__(self):
        return self

    def __next__(self):
        return self.map_func(self.iterator.__next__())


    def get_next(self):
        return self.__next__()


class Tensor_Dataset_Wraper():
    def __init__(self,dataset, map_func = (lambda x : x) ):
        self.dataset = dataset
        self.map_func = map_func

    def __iter__(self):
        return Tensor_Iterator_Wraper(self.dataset.__iter__(),self.map_func)


def make_iterator_offtime(config):
    def make_image_patches (inputs, is_train = True):
        file_names = [file_name.numpy().decode("utf-8") for file_name in inputs]
        inputs = []
        labels = []
        for file_name in tqdm(file_names):
            label = PIL.Image.open(file_name).convert('RGB')
            buffer = io.BytesIO()
            label.save(buffer, format='jpeg', quality=config.jpeg_quality)
            input = PIL.Image.open(buffer)

            if is_train :
                """ crop patch """
                crop_y = 0
                while crop_y + config.patch_size < label.height:
                    crop_x = 0
                    while crop_x + config.patch_size < label.width:
                        input_patch = input.crop((crop_x, crop_y, crop_x + config.patch_size, crop_y + config.patch_size))
                        label_patch = label.crop((crop_x, crop_y, crop_x + config.patch_size, crop_y + config.patch_size))
                        inputs.append(normalize(np.array(input_patch)))
                        labels.append(normalize(np.array(label_patch)))
                        crop_x += int(random.randint(37*5,62*5))
                    crop_y += int(random.randint(37*5,62*5))

            else :
                inputs.append(normalize(np.array(input).reshape[1,input.height,input.width, config.channels]))
                labels.append(normalize(np.array(label).reshape[1,label.height, label.width, config.channels]))

        print("total patches : ", len(inputs))
        return np.array(inputs), np.array(labels)


    """ prepare train iterator """
    paired_file_names = tf.data.Dataset.list_files(os.path.normcase(os.path.join(config.data_root_train, "*.*")))
    paired_input_patches, paired_label_patches = make_image_patches(paired_file_names)
    paired_train_dataset = tf.data.Dataset.from_tensor_slices((paired_input_patches,paired_label_patches)).batch(config.batch_size,drop_remainder=True).shuffle(config.buffer_size).repeat()
    paired_train_iterator = paired_train_dataset.__iter__()

    """ prepare test dataset """
    paired_file_names = tf.data.Dataset.list_files(os.path.normcase(os.path.join(config.data_root_test, "*.*")))
    paired_input_patches, paired_label_patches = make_image_patches(paired_file_names,is_train = False)
    paired_test_dataset = zip(paired_input_patches, paired_label_patches)

    return paired_train_iterator, paired_test_dataset


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
    test_iterator = paired_iterator

    return train_iterator, test_iterator  #, reference_iterator





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=4)  # -1 for CPU
    parser.add_argument("--crop_size", type=list, default=[512, 512], nargs="+", help='Image size after crop.')
    parser.add_argument("--buffer_size", type=int, default=20000, help='Data buffer size.')
    parser.add_argument("--batch_size", type=int, default=2, help='Minibatch size(global)')
    parser.add_argument("--patch_size", type=int, default=48, help='Minibatch size(global)')
    parser.add_argument("--jpeg_quality", type=int, default=20, help='Minibatch size(global)')
    parser.add_argument("--data_root_train", type=str, default='./dataset/test/Set5', help='Data root dir')
    parser.add_argument("--data_root_test", type=str, default='./dataset/test/Set5', help='Data root dir')
    parser.add_argument("--channels", type=int, default=3, help='Channel size')
    parser.add_argument("--model_tag", type=str, default="default", help='Exp name to save logs/checkpoints.')
    parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help='Dir for checkpoints.')
    parser.add_argument("--summary_dir", type=str, default='../__outputs/summaries/', help='Dir for tensorboard logs.')
    parser.add_argument("--restore_file", type=str, default=None, help='file for resotration')
    parser.add_argument("--graph_mode", type=bool, default=False, help='use graph mode for training')
    config = parser.parse_args()
    tf.executing_eagerly()

    train_iterator, test_dataset = make_iterator_offtime(config)
    for test_input, test_label in test_dataset:
        print(test_input.shape,test_label.shape)

    for train_input, train_label in train_iterator :
        print(train_input.shape, train_label.shape)
"""
    train_iterator, test_dataset = make_iterator_ontime(config)
    print(train_iterator)



    for test_input, test_label in test_dataset:
        print(test_input.shape, test_label.shape)
        # plt.imshow(input_test)
        # plt.show()

    for test_input, test_label in test_dataset:
        print(test_input.shape, test_label.shape)
        # plt.imshow(input_test)
        # plt.show()

    for inputs,labels in train_iterator:
        print(inputs.shape, labels.shape)
        plt.imshow(np.concatenate([inputs[0],labels[0]],axis=1))
        plt.show()
"""
