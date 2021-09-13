import os
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2labelim, tensor2confidencemap
from models.sne_model import SNE
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import copy
import tqdm
import glob


class dataset():
    def __init__(self):
        self.num_labels = 2


def load_calib(filepath):

    rawdata = read_calib_file(filepath)
    K = np.reshape(rawdata['cam_K'], (3,3))
    K[1, 2] = K[1, 2] - 8  # 720-16=704

    return K

def read_calib_file(filepath):
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


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.isTrain = False

    example_dataset = dataset()
    model = create_model(opt, example_dataset)
    model.setup(opt)
    model.eval()

    # if you want to use your own data, please modify seq_name, image_data, dense_depth, calib and use_size correspondingly.
    use_size = (1280, 704)
    root_dir = 'examples'
    seq_name = 'y0613_1242' # Need to be changed

    save_dir = os.path.join(root_dir, seq_name, 'results')
    os.makedirs(save_dir, exist_ok=True)

    img_list = glob.glob(os.path.join(root_dir, seq_name, 'image_data', '*.png'))

    for img_list_i in img_list:
        img_name = img_list_i.split('/')[-1].split('.')[0]
        print('img_name:',img_name)
        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(root_dir, seq_name, 'image_data', img_name+'.png')), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(os.path.join(root_dir, seq_name, 'dense_depth', img_name+'.png'), cv2.IMREAD_ANYDEPTH) 
        oriHeight, oriWidth, _ = rgb_image.shape
        oriSize = (oriWidth, oriHeight)
        rgb_image_save = copy.copy(rgb_image)
        rgb_image_save_ = copy.copy(rgb_image)

        # resize image to enable sizes divide 32

        rgb_image = cv2.resize(rgb_image, use_size)
        
        rgb_image = rgb_image.astype(np.float32) / 255

        # compute normal using SNE
        sne_model = SNE()
        calib_dir = os.path.join(root_dir, seq_name, 'calib', img_name+'.txt')
        camParam = load_calib(calib_dir)
        
        normal = sne_model(torch.tensor(depth_image.astype(np.float32)/256), camParam)
        normal_image = normal.cpu().numpy()
        normal_image = np.transpose(normal_image, [1, 2, 0])
        cv2.imwrite(os.path.join(os.path.join(save_dir, img_name+'_normal.png')), cv2.cvtColor(255*(1+normal_image)/2, cv2.COLOR_RGB2BGR))
        normal_image_save = cv2.cvtColor(255*(1+normal_image)/2, cv2.COLOR_RGB2BGR)
        normal_image = cv2.resize(normal_image, use_size)

        rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(dim=0)
        normal_image = transforms.ToTensor()(normal_image).unsqueeze(dim=0)

        with torch.no_grad():
            pred = model.netRoadSeg(rgb_image, normal_image)

            palet_file = 'datasets/palette.txt'
            impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3*256))
            pred_img = tensor2labelim(pred, impalette)
            pred_img = cv2.resize(pred_img, oriSize)
            prob_map = tensor2confidencemap(pred)
            prob_map = cv2.resize(prob_map, oriSize)
            cv2.imwrite(os.path.join(os.path.join(save_dir, img_name+'_pred.png')), pred_img)
            cv2.imwrite(os.path.join(os.path.join(save_dir, img_name+'_probmap.png')), prob_map)

            rgb_image_save = rgb_image_save.transpose(2,0,1)
            pred_img = pred_img.transpose(2,0,1)

            inds = prob_map>128

            rgb_image_save[:,inds] = pred_img[:,inds]
            rgb_image_save = rgb_image_save.transpose(1,2,0)

            cv2.imwrite(os.path.join(os.path.join(save_dir, img_name+'_mask.png')), rgb_image_save)

            img_cat = np.concatenate((rgb_image_save_, normal_image_save, rgb_image_save),axis=1)
            img_cat = cv2.resize(img_cat, (int(img_cat.shape[1]*0.5),int(img_cat.shape[0]*0.5)))
            cv2.imwrite(os.path.join(os.path.join(root_dir, seq_name, img_name+'.png')), img_cat)
            #cv2.imshow('img_cat',img_cat)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    print('Done!')
