import os
import random
import math
import cv2
import numpy as np
import tensorflow as tf
from copy import deepcopy


class Imagefolder(object):
    def __init__(self,
                 data_root,
                 data_split,
                 input_size=224,
                 data_augment=False,
                 hyp_params=None):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if hyp_params is None:
            hyp_params = {'hsv': [0.015, 0.7, 0.4],
                          'flip': 0.5,
                          'crop': 0.5,
                          'mean': [0.485, 0.456, 0.406],
                          'std': [0.229, 0.224, 0.225]}
        self.data_root = data_root
        self.data_split = data_split
        self.input_size = input_size
        self.data_augment = data_augment
        self.hyp_params = hyp_params
        data_path = os.path.join(data_root, data_split)
        extensions = ('.jpg', '.jpeg', '.png', '.ppm',
                      '.bmp', '.pgm', '.tif', '.tiff')
        classes, class_map = self.find_classes(data_path)
        samples = self.make_dataset(data_path,
                                    class_map,
                                    extensions)
        if len(samples) == 0:
            raise RuntimeError('Found 0 files in ' + data_path)
        self.classes = classes
        self.class_map = class_map
        self.samples = samples
        self.sample_index = 0

    @staticmethod
    def find_classes(data_root):
        classes = [d.name for d in os.scandir(data_root) if d.is_dir()]
        classes.sort()
        class_map = {cls_name: k for k, cls_name in enumerate(classes)}
        return classes, class_map

    @staticmethod
    def make_dataset(directory,
                     class_map,
                     extensions):
        instances = []
        directory = os.path.expanduser(directory)
        for target_class in sorted(class_map.keys()):
            class_index = class_map[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, file_names in sorted(os.walk(target_dir,
                                                      followlinks=True)):
                for name in sorted(file_names):
                    path = os.path.join(root, name)
                    if path.lower().endswith(extensions):
                        instances.append((path, class_index))
        return instances

    @staticmethod
    def load_image(image_path):
        if not os.path.exists(image_path):
            raise KeyError('%s does not exist ...' % image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.uint8)
        return image

    @staticmethod
    def random_size_rect(image,
                         scale=(0.08, 1.0),
                         ratio=(3 / 4.0, 4 / 3.0),
                         max_times=10):
        height, width = image.shape[:2]
        area = height * width
        for _ in range(max_times):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]),
                         math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)
                return top, left, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        top = (height - h) // 2
        left = (width - w) // 2
        return top, left, h, w

    def fixed_size_rect(self, image):
        height, width = image.shape[:2]
        th, tw = self.input_size
        if height == th and width == tw:
            return 0, 0, height, width

        top = random.randint(0, height - th)
        left = random.randint(0, width - tw)
        return top, left, th, tw

    def random_crop(self,
                    image,
                    inplace=False):
        if not inplace:
            image = deepcopy(image)
        height, width = image.shape[:2]
        input_h, input_w = self.input_size
        h_thresh, w_thresh = (round(1.25 * input_h),
                              round(1.25 * input_w))
        padding = None
        if height < h_thresh or width < w_thresh:
            padding_h, padding_w = (0, 0), (0, 0)
            if height < h_thresh:
                padding_size = h_thresh - height
                padding_h = (padding_size // 2,
                             padding_size // 2 + padding_size % 2)
            if width < w_thresh:
                padding_size = w_thresh - width
                padding_w = (padding_size // 2,
                             padding_size // 2 + padding_size % 2)
            padding = (padding_h, padding_w, (0, 0))
        if padding is not None:
            image = np.pad(image,
                           padding,
                           mode='constant',
                           constant_values=0)
            top, left, h, w = self.fixed_size_rect(image)
        else:
            top, left, h, w = self.random_size_rect(image)
        return image[top:(top + h), left:(left + w), :]

    def random_hsv(self, image):
        ratio = 1.0 + np.random.uniform(-1, 1, 3) * self.hyp_params['hsv']
        hue, sat, val = cv2.split(cv2.cvtColor(image,
                                               cv2.COLOR_BGR2HSV))
        dtype = image.dtype
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue, lut_sat, lut_val = \
            (((x * ratio[0]) % 180).astype(dtype),
             np.clip(x * ratio[1], 0, 255).astype(dtype),
             np.clip(x * ratio[2], 0, 255).astype(dtype))

        image_hsv = cv2.merge((cv2.LUT(hue, lut_hue),
                               cv2.LUT(sat, lut_sat),
                               cv2.LUT(val, lut_val)))
        image_hsv = image_hsv.astype(dtype)
        return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    def normalize(self,
                  image,
                  inplace=False):
        if not inplace:
            image = deepcopy(image)
        image = image.astype(np.float32)
        mean = np.array(self.hyp_params['mean'], dtype=np.float32)
        std = np.array(self.hyp_params['std'], dtype=np.float32)
        if (std == 0).any():
            raise ValueError('Normalization division by zero')
        image /= 255.0
        image -= mean
        image /= std
        return image

    def load_sample(self):
        image_path, label = self.samples[self.sample_index]
        image = self.load_image(image_path)
        if self.data_augment:
            image = self.random_hsv(image)
            if random.random() < self.hyp_params['flip']:
                image = np.fliplr(image)
            if random.random() < self.hyp_params['crop']:
                image = self.random_crop(image)
        if image.shape[:2] != self.input_size:
            interpolation = cv2.INTER_LINEAR
            if self.data_augment:
                interpolation = cv2.INTER_AREA
            image = cv2.resize(image,
                               (self.input_size[1], self.input_size[0]),
                               interpolation=interpolation)
        image = self.normalize(image, inplace=True)
        image = np.ascontiguousarray(image)
        self.sample_index += 1
        yield image, label

    def make_dataset(self,
                     batch_size,
                     shuffle=False,
                     iter_count=None,
                     drop_remainder=False,
                     buffer_size=0):
        output_shapes = ((None, None, 3), (1,))
        ds = tf.data.Dataset.from_generator(
            self.load_sample,
            output_types=(tf.float32, tf.float32),
            output_shapes=output_shapes,
            args=None)
        if shuffle:
            ds = ds.shuffle(len(self.samples))
        ds = ds.repeat(count=iter_count)
        ds_batch = ds.padded_batch(
            batch_size,
            padded_shapes=output_shapes,
            padding_values=(0, 0),
            drop_remainder=drop_remainder)
        ds_batch = ds_batch.apply(tf.data.experimental.ignore_errors())
        if buffer_size == 0:
            buffer_size = tf.data.experimental.AUTOTUNE
        ds_batch = ds_batch.prefetch(buffer_size=buffer_size)
        return ds_batch
