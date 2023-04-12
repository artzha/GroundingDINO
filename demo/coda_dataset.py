import copy
import torch
import pickle
import numpy as np

from skimage import io
from torch.utils.data import Dataset

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("SCRIPT ", SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
from coda_utils import object3d_kitti

class CODataset(Dataset):
    """
    Assumes that CODa is in KITTI format
    """
    def __init__(self, class_names, root_path, split, training=True):
        super().__init__()
        self.split = split
        self.mode = training
        self.root_path = root_path
        self.class_names = class_names
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.coda_infos = []
        self.include_coda_data(self.mode)

        # self.coda_infos = self.balanced_infos_resampling(self.coda_infos)

    def include_coda_data(self, mode):
        coda_infos = []

        info_path = "coda_infos_val.pkl"
        if mode=="training":
            info_path = "coda_infos_train.pkl"

        info_path = self.root_path / info_path
        if not info_path.exists():
            print("info path dne %s " % info_path)
            return
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            coda_infos.extend(infos)

        self.coda_infos.extend(coda_infos)

        print('Total samples for CODa dataset: %d' % (len(self.coda_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of CODa dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}

        for info in infos:
            for name in set(info['annos']['name']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        print('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['annos']['name']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_image(self, idx):
        img_file = self.root_split_path / 'image_0' / ('%s.jpg' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file), dtype=np.int32), img_file

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        
        return np.fromfile(str(lidar_file)).reshape(-1, 4)

    def get_bbox_label(self, idx):
        """
        In camera coordinates
        """
        label_file = self.root_split_path / 'label_0' / ('%s.txt' % idx)
        assert label_file.exists(), "Label file %s does not exist" % label_file
        return object3d_kitti.get_objects_from_label(label_file)

    def __len__(self):
        return len(self.coda_infos)

    def collate_fn(self, batch):
        sample_batch = [bi[0] for bi in batch]
        img_batch = [bi[1] for bi in batch]
        img_file_batch = [bi[2] for bi in batch]
        gt_label_batch = [bi[3] for bi in batch]
        gt_bbox_batch = [bi[4] for bi in batch]
        return sample_batch, img_batch, img_file_batch, gt_label_batch, gt_bbox_batch

    def __getitem__(self, index):
        info = copy.deepcopy(self.coda_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img, img_file   = self.get_image(sample_idx)
        labels = info['annos']['name']
        bbox   = info['annos']['bbox']

        valid_labels_idx = np.logical_not(np.all(bbox==0, axis=-1))

        valid_labels = labels[valid_labels_idx]
        valid_bbox = bbox[valid_labels_idx]

        return sample_idx, img, img_file, valid_labels, valid_bbox
