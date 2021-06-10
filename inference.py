# Python
import os
import sys
import random
import traceback
import warnings
warnings.filterwarnings('ignore')
# 3rd party
import cv2
import numpy
import dill as pickle
# torch
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# self
import utils
import model


# 忽略警告
warnings.filterwarnings('ignore')
# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_default_tensor_type(torch.FloatTensor)


# ------------------------------- 定义输入参数 --------------------------------------

# 参数
opt = lambda: None
opt.luma_bins = 8
opt.channel_multiplier = 1
opt.spatial_bin = 8
opt.batch_norm = True
opt.low_size = 256
opt.full_size = 512
opt.use_cuda = True
opt.eps_value = 1e-4
opt.images_dir = "./sample_imgs"
opt.result_dir = './sample_results'
opt.save_extra = True
opt.checkpoints_file = './checkpoints/simple_batch_32/epoch_100_psnr_23.721.pth'
for l, r in vars(opt).items(): print(l, " : ", r)
os.makedirs(opt.result_dir, exist_ok=True)
assert os.path.exists(opt.checkpoints_file), "checkpoints_file {} doesn't exists !".format(opt.checkpoints_file)
assert os.path.exists(opt.images_dir), "imaged folder {} doesn't exists !".format(opt.images_dir)


# ------------------------------- 加载网络权重 --------------------------------------
network = model.HDRNet(opt)
network.load_state_dict(torch.load(opt.checkpoints_file))
print('loaded weights from {}'.format(opt.checkpoints_file))
if(opt.use_cuda):
	network = network.cuda()
# 关闭 dropout, batch_norm
network.eval()


# ------------------------------- 处理图片 --------------------------------------

# numpy -> tensor
pre_transform = lambda x: torch.from_numpy(x.astype('float32')).div(255).permute(2, 0, 1).unsqueeze(0).cuda()
# tensor -> numpy
post_transform = lambda x: x.detach().squeeze(0).permute(1, 2, 0).cpu().mul(255).numpy().astype('uint8')


images_list = os.listdir(opt.images_dir)
images_list = [it for it in images_list if(it.lower().endswith(
	('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')))]
images_list = [os.path.join(opt.images_dir, it) for it in images_list]
print('{} images are to be processed !'.format(len(images_list)))


with utils.Timer() as time_scope:
	# 不计算梯度
	with torch.no_grad():
		for cnt, image_path in enumerate(images_list, 1):
			try:
				# 读取图像
				origin = cv2.imread(image_path)
				low_res = cv2.resize(origin, (opt.low_size, opt.low_size))
				# 转化成 tensor
				origin_tensor = pre_transform(origin)
				low_res_tensor = pre_transform(low_res)
				# 送到 GPU
				if(opt.use_cuda):
					origin_tensor, low_res_tensor = origin_tensor.cuda(), low_res_tensor.cuda()
				# 网络推理
				illmunination = network(low_res_tensor, origin_tensor)
				# 反射图 = 原图 / 光照估计
				reflectance = torch.div(origin_tensor, illmunination + opt.eps_value)
				reflectance = torch.clamp(reflectance, 0, 1)
				# 保存
				result = post_transform(reflectance)
				image_name = os.path.split(image_path)[-1]
				save_path = os.path.join(opt.result_dir, image_name)
				if(opt.save_extra == True):
					result = numpy.concatenate([origin, result], axis=1)
					result = numpy.concatenate([result, post_transform(illmunination)], axis=1)
				cv2.imwrite(save_path, result)
				print('{}/{}===> processing {} and saved to {}'.format(cnt, len(images_list), image_path, save_path))
			except Exception as e:
				print(traceback.print_exc())
				print('{}/{}===> processing {} errored !')
