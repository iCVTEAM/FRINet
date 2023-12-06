import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
import shutil
def check_mkdir(dir_name, delete_if_exists=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        if delete_if_exists:
            print(f"{dir_name} will be re-created!!!")
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)

class TBRecorder(object):
    __slots__ = ["tb"]

    def __init__(self, tb_path):
        check_mkdir(tb_path, delete_if_exists=True)
        self.tb = SummaryWriter(tb_path)

    def record_curve(self, name, data, curr_iter):
        self.tb.add_scalars(f"data/{name}", data, curr_iter)
        # if not isinstance(data, (tuple, list)):
        #     self.tb.add_scalar(f"data/{name}", data, curr_iter)
        # else:
        #     for idx, data_item in enumerate(data):
        #         self.tb.add_scalar(f"data/{name}_{idx}", data_item, curr_iter)

    def record_image(self, name, data, curr_iter,dataformats='CHW'):
        data_grid = make_grid(data[0:4], nrow=4, padding=5)
        self.tb.add_image(name, data_grid, curr_iter,dataformats=dataformats)

    def record_images(self, data_container: dict, curr_iter,dataformats='CHW'):
        for name, data in data_container.items():
            if'p' in name:
                data_grid = make_grid(data[0:4], nrow=4, padding=5,normalize=True,value_range=(0,255))
            data_grid = make_grid(data[0:4], nrow=4, padding=5)
            self.tb.add_image(name, data_grid, curr_iter)

    def record_histogram(self, name, data, curr_iter):
        self.tb.add_histogram(name, data, curr_iter)

    def close_tb(self):
        self.tb.close()