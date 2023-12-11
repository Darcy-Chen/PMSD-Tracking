import os
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import opencv_loader
import torch
import random
import torchvision.datasets as datasets
from collections import OrderedDict
from lib.train.admin import env_settings
import csv
import pandas
import numpy as np


class CustomDataset(BaseVideoDataset):
    """ The ImageNet1k dataset. ImageNet1k is an image dataset. Thus, we treat each image as a sequence of length 1.
    """

    def __init__(self, root=None, image_loader=opencv_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the custom dataset.
            image_loader (default_image_loader) -  The function to read the images.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
            split - 'train' or 'val'.
        """
        root = env_settings().custom_dir if root is None else root
        if split == 'train':
            root = os.path.join(root, 'train')
        elif split == 'val':
            root = os.path.join(root, 'val')
                    
        super().__init__('custom', root, image_loader)

        self.sequence_list = self._get_sequence_list()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        seq_ids = list(range(0, len(self.sequence_list)))


    def get_name(self):
        return 'custom'

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, label_path):
        bb_anno_file = os.path.join(label_path, "label.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        label_path = os.path.join(seq_path, 'label')
        bbox = self._read_bb_anno(label_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'mask': None, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        img_path = os.path.join(seq_path, 'img')
        return os.path.join(img_path, '{:04}.png'.format(frame_id))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        obj_meta = None
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key != 'mask':
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return anno_frames
