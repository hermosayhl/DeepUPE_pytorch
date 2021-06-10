import os
import cv2
import sys
import math
import numpy
import random
import torch



def set_seed(seed):
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    numpy.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)



# 为 torch 数据随机做准备
GLOBAL_SEED = 19980212
GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


import datetime

class Timer:
    def __enter__(self):
        self.start = datetime.datetime.now()

    def __exit__(self, type, value, trace):
        _end = datetime.datetime.now()
        print('耗时  :  {}'.format(_end - self.start))




def visualize_a_batch(batch_images, save_path, total_size=16):
    # tensor -> numpy
    batch_images = batch_images.detach().cpu().permute(0, 2, 3, 1).mul(255).numpy().astype('uint8')
    # (16, 512, 512, 3) -> [4 * 512, 4 * 512, 3]
    composed_images = numpy.concatenate([numpy.concatenate([batch_images[4 * i + j] for j in range(4)], axis=1) for i in range(4)], axis=0)
    cv2.imwrite(save_path, composed_images)



