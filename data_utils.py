import os
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import numpy as  np
import sys
from random import random
from tqdm import tqdm




def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def normalize(images):
    return (images.astype(np.float32)/255.0)
def denormalize(images):
    return (images*255.0).astype(np.uint8)


def resize_and_random_crop(image,crop_size): #assumed numpy single image which has shape [1,H,W,ch]
    """ resize """
    small_side_len = np.minimum(image.shape[1],image.shape[2])
    if small_side_len < np.maximum(crop_size[0],crop_size[1]):
        scale = np.maximum(crop_size[0],crop_size[1])/small_side_len
        new_h = int(image.shape[1] * scale) +1
        new_w = int(image.shape[2] * scale) +1

        image = cv2.resize(image.reshape([image.shape[1], image.shape[2]]), (new_w, new_h)).reshape([image.shape[0],new_h,new_w,image.shape[-1]])

    """ random crop """
    _,h,w,_ = image.shape
    crop_h_start = np.random.randint(0,h-crop_size[0]) if h-crop_size[0] > 0 else 0
    crop_w_start = np.random.randint(0,w-crop_size[1]) if w-crop_size[1] > 0 else 0
    image = image[:,crop_h_start:crop_h_start+crop_size[0],crop_w_start:crop_w_start+crop_size[1],:]
    return image

def prepare_image_as_numpy_4Dshape_from_paths(list_files,channels, crop_size):
    image_list = []
    #for l in tqdm(list_files,total= len(list_files)):
    for l in list_files:
        """ load image """
        flag_channels = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        img = cv2.imread(l, flags=flag_channels)

        """ resize image """
        bigger_size = max(img.shape[0], img.shape[1])
        if bigger_size > 1024:
            mult = 1024 / float(bigger_size)
            new_s = (int(img.shape[1] * mult), int(img.shape[0] * mult))
            img = cv2.resize(img, new_s)

        new_w = int(img.shape[1])  - int(img.shape[1])%8
        new_h = int(img.shape[0])  - int(img.shape[0])%8
        img = cv2.resize(img, (new_w,new_h))

        """ shaping """
        if img.shape.__len__() <= 2: img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        """ processing """
        img = resize_and_random_crop(img, crop_size)

        """ appending """
        image_list.append(img)


    images = np.concatenate(image_list, axis=0)
    images = normalize(images)
    return images

def normalize_line_from_numpy_images(images, channels, line_normalizer): #assuming images shape is [bs,h,w,ch]
    results_list = []
    for i in range(images.shape[0]):
        result = line_normalizer.inference(images[i:i+1])
        result = line_normalizer.inference(result).numpy()
        results_list.append(result)
    results = np.concatenate(results_list,axis=0)
    return results






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







def make_iterator(config,line_normalizer):
    """ prepare file names """
    list_files_ori = sorted(glob(os.path.join(os.path.normcase(config.data_root), "train/*.*")))

    """ prepare dataset as numpy"""
    images_ori = prepare_image_as_numpy_4Dshape_from_paths(list_files_ori, config.channels, config.crop_size)
    images_normed = normalize_line_from_numpy_images(images_ori, config.channels, line_normalizer= line_normalizer)
    print(images_normed.shape, images_ori.shape)

    """ prepare unpaired iterator """
    train_dataset_normed = tf.data.Dataset.from_tensor_slices((images_normed)).batch(config.batch_size, drop_remainder=True).shuffle(config.buffer_size).repeat().__iter__()
    train_dataset_ori = tf.data.Dataset.from_tensor_slices((images_ori)).batch(config.batch_size, drop_remainder=True).shuffle(config.buffer_size).repeat().__iter__()

    """ prepare paired iterator """
    train_dataset_paired = tf.data.Dataset.from_tensor_slices((images_normed, images_ori)).batch(config.batch_size,drop_remainder=True).shuffle(config.buffer_size).repeat().__iter__()
    train_iterator = zip(train_dataset_paired, train_dataset_normed, train_dataset_ori)

    """ prepare test dataset"""
    list_files_tests = sorted(glob(os.path.join(os.path.normcase(config.data_root), "test/*.*")))
    images_normed_tests = prepare_image_as_numpy_4Dshape_from_paths(list_files_tests, config.channels, config.crop_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((images_normed_tests)).batch(1)

    reference_iterator = tf.data.Dataset.from_tensor_slices((images_ori)).batch(1).repeat().__iter__()
    return train_iterator, test_dataset, reference_iterator




def make_iterator_ontime(config, line_normalizer ):

    """ mapping functions """
    def mapping_function_for_paired_iterator(inputs):
        inputs = [file_name.decode("utf-8") for file_name in inputs.numpy()]  #byte-strings
        images = prepare_image_as_numpy_4Dshape_from_paths(inputs, config.channels, config.crop_size)
        images_normed = normalize_line_from_numpy_images(images, config.channels, line_normalizer= line_normalizer)
        return images_normed, images

    def mapping_function_for_unpaired_iterator(inputs):
        inputs = [file_name.decode("utf-8") for file_name in inputs.numpy()]  #byte-strings
        images = prepare_image_as_numpy_4Dshape_from_paths(inputs, config.channels, config.crop_size)
        return images

    def mapping_function_for_unpaired_iterator_normalize(inputs):
        inputs = [file_name.decode("utf-8") for file_name in inputs.numpy()]  #byte-strings
        images = prepare_image_as_numpy_4Dshape_from_paths(inputs, config.channels, config.crop_size)
        images_normed = normalize_line_from_numpy_images(images, config.channels, line_normalizer= line_normalizer)
        return images_normed


    """ prepare train iterator """
    # prepare paired iterator
    paired_dataset = tf.data.Dataset.list_files(os.path.join(config.data_root, 'train/*.*'))
    paired_dataset = paired_dataset.batch(config.batch_size,drop_remainder=True).shuffle(config.buffer_size).repeat()
    paired_iterator = Tensor_Iterator_Wraper(paired_dataset.__iter__(), map_func= mapping_function_for_paired_iterator)

    # prepare unpaired iterator
    unpaired_dataset = tf.data.Dataset.list_files(os.path.join(config.data_root, 'train/*.*'))
    unpaired_dataset = unpaired_dataset.batch(config.batch_size,drop_remainder=True).shuffle(config.buffer_size).repeat()
    unpaired_iterator = Tensor_Iterator_Wraper(unpaired_dataset.__iter__(), map_func= mapping_function_for_unpaired_iterator)

    unpaired_dataset_normed = tf.data.Dataset.list_files(os.path.join(config.data_root, 'train/*.*'))
    unpaired_dataset_normed = unpaired_dataset_normed.batch(config.batch_size,drop_remainder=True).shuffle(config.buffer_size).repeat()
    unpaired_iterator_normed = Tensor_Iterator_Wraper(unpaired_dataset_normed.__iter__(), map_func= mapping_function_for_unpaired_iterator_normalize)
    #unpaired_dataset_normed = tf.data.Dataset.list_files(os.path.join('/datasets/legacy/legacy_sketchDB/eurographics_data/synthetics/lines_scale1', '*.*'))
    #unpaired_dataset_normed = unpaired_dataset_normed.batch(config.batch_size,drop_remainder=True).shuffle(config.buffer_size).repeat()
    #unpaired_iterator_normed = Tensor_Iterator_Wraper(unpaired_dataset_normed.__iter__(), map_func= mapping_function_for_unpaired_iterator)



    train_iterator = Tensor_Iterator_Wraper(zip(paired_iterator, unpaired_iterator_normed, unpaired_iterator))


    """ prepare test dataset """
    list_files_tests = sorted(glob(os.path.join(os.path.normcase(config.data_root), "test/*.*")))
    images_normed_tests = prepare_image_as_numpy_4Dshape_from_paths(list_files_tests, config.channels, config.crop_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((images_normed_tests)).batch(1)

    reference_dataset = tf.data.Dataset.list_files(os.path.join(config.data_root, 'train/*.*'))
    reference_dataset = reference_dataset.batch(config.batch_size,drop_remainder=True).shuffle(config.buffer_size).repeat()
    reference_iterator = Tensor_Iterator_Wraper(reference_dataset.__iter__(), map_func= mapping_function_for_unpaired_iterator)


    return train_iterator, test_dataset, reference_iterator





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=4)  # -1 for CPU
    parser.add_argument("--crop_size", type=list, default=[512, 512], nargs="+", help='Image size after crop.')
    parser.add_argument("--buffer_size", type=int, default=20000, help='Data buffer size.')
    parser.add_argument("--batch_size", type=int, default=1, help='Minibatch size(global)')
    parser.add_argument("--data_root", type=str, default='/datasets/line_stylizer_data/', help='Data root dir')
    parser.add_argument("--channels", type=int, default=1, help='Channel size')
    parser.add_argument("--model_tag", type=str, default="default", help='Exp name to save logs/checkpoints.')
    parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help='Dir for checkpoints.')
    parser.add_argument("--summary_dir", type=str, default='../__outputs/summaries/', help='Dir for tensorboard logs.')
    parser.add_argument("--restore_file", type=str, default=None, help='file for resotration')
    parser.add_argument("--graph_mode", type=bool, default=False, help='use graph mode for training')
    config = parser.parse_args()
    tf.executing_eagerly()

    train_iterator, test_dataset, reference_iterator = make_iterator_ontime(config)
    for i in train_iterator:
        print(i)