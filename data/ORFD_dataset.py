import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import glob
from data.base_dataset import BaseDataset
from models.sne_model import SNE


class orfdCalibInfo():
    """
    Read calibration files in the ORFD dataset,
    we need to use the intrinsic parameter
    """
    def __init__(self, filepath):
        """
        Args:
            filepath ([str]): calibration file path (AAA.txt)
        """
        self.data = self._load_calib(filepath)

    def get_cam_param(self):
        """
        Returns:
            [numpy.array]: intrinsic parameter 
        """
        return self.data['K']

    def _load_calib(self, filepath):
        rawdata = self._read_calib_file(filepath)
        data = {}
        K = np.reshape(rawdata['cam_K'], (3,3))
        data['K'] = K
        return data

    def _read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data


class orfddataset(BaseDataset):
    """dataloader for ORFD dataset"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.root = opt.dataroot # path for the dataset
        self.use_sne = opt.use_sne
        self.num_labels = 2
        self.use_size = (opt.useWidth, opt.useHeight)
        if self.use_sne:
            self.sne_model = SNE()

        if opt.phase == "train":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'training', '*','image_data', '*.png'))) # shape: 1280*720
        elif opt.phase == "val":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'validation', '*','image_data', '*.png')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'testing', '*','image_data', '*.png')))

    def __getitem__(self, index):
        useDir = "/".join(self.image_list[index].split('/')[:-2])
        name = self.image_list[index].split('/')[-1]

        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'image_data', name)), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(os.path.join(useDir, 'dense_depth', name), cv2.IMREAD_ANYDEPTH)
        oriHeight, oriWidth, _ = rgb_image.shape
        if self.opt.phase == 'test' and self.opt.no_label:
            # Since we have no gt label (e.g., kitti submission), we generate pseudo gt labels to
            # avoid destroying the code architecture
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        else:
            #label_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'gt_image_2', name[:-10]+'road_'+name[-10:])), cv2.COLOR_BGR2RGB)
            label_img_name = name.split('.')[0]+"_fillcolor.png"
            label_dir = os.path.join(useDir, 'gt_image', label_img_name)
            label_image = cv2.cvtColor(cv2.imread(label_dir), cv2.COLOR_BGR2RGB)
            
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
            label[label_image[:,:,2] > 200] = 1


        # resize image to enable sizes divide 32
        rgb_image = cv2.resize(rgb_image, self.use_size)
        label = cv2.resize(label, (int(self.use_size[0]/4), int(self.use_size[1]/4)), interpolation=cv2.INTER_NEAREST)

        # another_image will be normal when using SNE, otherwise will be depth
        if self.use_sne:
            calib = orfdCalibInfo(os.path.join(useDir, 'calib', name.split('.')[0] +'.txt'))
            camParam = torch.tensor(calib.get_cam_param(), dtype=torch.float32)
            camParam[1, 2] = camParam[1, 2] - 8  # 720-16=704
            
            #normal = self.sne_model(torch.tensor(depth_image.astype(np.float32)/1000), camParam)
            normal = self.sne_model(torch.tensor(depth_image.astype(np.float32)/256), camParam)
            another_image = normal.cpu().numpy()
            another_image = np.transpose(another_image, [1, 2, 0])
            another_image = cv2.resize(another_image, self.use_size)
        else:
            another_image = depth_image.astype(np.float32)/65535
            another_image = cv2.resize(another_image, self.use_size)
            another_image = another_image[:,:,np.newaxis]

        label[label > 0] = 1
        rgb_image = rgb_image.astype(np.float32) / 255

        rgb_image = transforms.ToTensor()(rgb_image)
        another_image = transforms.ToTensor()(another_image)

        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        # return a dictionary containing useful information
        # input rgb images, another images and labels for training;
        # 'path': image name for saving predictions
        # 'oriSize': original image size for evaluating and saving predictions
        return {'rgb_image': rgb_image, 'another_image': another_image, 'label': label,
                'path': name, 'oriSize': (oriWidth, oriHeight)}

    def __len__(self):
        return len(self.image_list)

    def name(self):
        return 'orfd'
