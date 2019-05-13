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

def make_iterator_ontime(config):
    """ mapping functions """
    def mapping_function_for_paired_iterator(inputs):
        file_names = [file_name.decode("utf-8") for file_name in inputs.numpy()]
        inputs = []
        labels = []
        for file_name in file_names :
            label = PIL.Image.open(file_name).convert('RGB')
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
    paired_file_names = paired_file_names.batch(config.batch_size,drop_remainder=True).shuffle(config.buffer_size).repeat()
    paired_iterator = Tensor_Iterator_Wraper(paired_file_names.__iter__(), map_func= mapping_function_for_paired_iterator)
    train_iterator = paired_iterator


    """ prepare test dataset """
    paired_file_names = tf.data.Dataset.list_files(os.path.normcase(os.path.join(config.data_root_test,"*.*")))
    paired_file_names = paired_file_names.batch(1,drop_remainder=True).shuffle(config.buffer_size)
    paired_iterator = Tensor_Dataset_Wraper(paired_file_names, map_func= mapping_function_for_paired_iterator)
    test_iterator = paired_iterator

    """
    list_files_tests = sorted(glob(os.path.join(os.path.normcase(config.data_root_test),"*.*")))
    test_dataset = []
    for l in list_files_tests:
        flag_channels = cv2.IMREAD_GRAYSCALE if config.channels == 1 else cv2.IMREAD_COLOR
        img = cv2.imread(l, flags=flag_channels)[...,::-1]
        img = np.expand_dims(img,axis=0)
        test_dataset.append((img.astype(np.float32)  ))
    """
    return train_iterator, test_iterator  #, reference_iterator





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=4)  # -1 for CPU
    parser.add_argument("--crop_size", type=list, default=[512, 512], nargs="+", help='Image size after crop.')
    parser.add_argument("--buffer_size", type=int, default=20000, help='Data buffer size.')
    parser.add_argument("--batch_size", type=int, default=16, help='Minibatch size(global)')
    parser.add_argument("--patch_size", type=int, default=48, help='Minibatch size(global)')
    parser.add_argument("--jpeg_quality", type=int, default=20, help='Minibatch size(global)')
    parser.add_argument("--data_root_train", type=str, default='./dataset/train/BSD400', help='Data root dir')
    parser.add_argument("--data_root_test", type=str, default='./dataset/test/Set5', help='Data root dir')
    parser.add_argument("--channels", type=int, default=3, help='Channel size')
    parser.add_argument("--model_tag", type=str, default="default", help='Exp name to save logs/checkpoints.')
    parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help='Dir for checkpoints.')
    parser.add_argument("--summary_dir", type=str, default='../__outputs/summaries/', help='Dir for tensorboard logs.')
    parser.add_argument("--restore_file", type=str, default=None, help='file for resotration')
    parser.add_argument("--graph_mode", type=bool, default=False, help='use graph mode for training')
    config = parser.parse_args()
    tf.executing_eagerly()

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
