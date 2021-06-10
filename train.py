# Python
import os
import sys
import random
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
import evaluate
import pipeline

# 设置随机种子
utils.set_seed(212)
# 忽略警告
warnings.filterwarnings('ignore')
# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_default_tensor_type(torch.FloatTensor)


# ------------------------------- 定义超参等 --------------------------------------

# 参数
opt = lambda: None
# hdrnet 参数
opt.luma_bins = 8
opt.channel_multiplier = 1
opt.spatial_bin = 8
opt.batch_norm = True
opt.low_size = 256
opt.full_size = 512
opt.eps_value = 1e-4
# 训练参数
opt.use_cuda = True
opt.optimizer = torch.optim.Adam
opt.lr = 1e-2
opt.total_epochs = 100
opt.train_batch_size = 16
opt.valid_batch_size = 1
opt.test_batch_size = 1
opt.valid_repeat = 4
# 实验参数
opt.exp_name = "simple_batch_16"
opt.save = True
opt.valid_interval = 1
opt.checkpoints_dir = os.path.join("./checkpoints/", opt.exp_name)

opt.dataset_name = 'fivek'
opt.dataset_ratios = [0.9, 0.1]
opt.dataset_dir = '/home/cgy/Chang/image_enhancement/datasets/fiveK'
# opt.dataset_dir = "C:/Code/HermosaWork/datasets/MIT-Adobe FiveK"
# 可视化参数
opt.visualize_size = 16
opt.visualize_batch = 100
opt.visualize_dir = os.path.join(opt.checkpoints_dir, './train_phase') 


for l, r in vars(opt).items(): print(l, " : ", r)
os.makedirs(opt.checkpoints_dir, exist_ok=True)
os.makedirs(opt.visualize_dir, exist_ok=True)
assert os.path.exists(opt.dataset_dir), "dataset for low/high quality image pairs doesn't exist !"



# ------------------------------- 定义数据读取 --------------------------------------
train_images_list, valid_images_list = pipeline.get_images(opt)
print('\ntrain  :  {}\nvalid  :  {}'.format(len(train_images_list), len(valid_images_list)))
# train
train_dataset = pipeline.HDRDataset(train_images_list, low_size=opt.low_size, full_size=opt.full_size)
train_loader = DataLoader(
	train_dataset, 
	batch_size=opt.train_batch_size, 
	shuffle=True,
	worker_init_fn=utils.worker_init_fn)
# valid
valid_dataset = pipeline.HDRDataset(valid_images_list, low_size=opt.low_size, full_size=opt.full_size)
valid_loader = DataLoader(
	valid_dataset,
	batch_size=opt.valid_batch_size,
	shuffle=False,
	# 为了泛化, 对验证数据集进行重复采样, 而且还会进行数据增强
	sampler=torch.utils.data.RandomSampler(valid_dataset, replacement=True, num_samples=opt.valid_repeat * len(valid_dataset)))

# # 将数据划分保存到 checkpoints
# with open(os.path.join(opt.checkpoints_dir, "dataset_split.pkl"), 'wb') as writer:
# 	pickle.dump({"train_images_list": train_images_list, "valid_images_list": valid_images_list}, writer)


# ------------------------------- 定义网络结构 --------------------------------------
network = model.HDRNet(opt)
if(opt.use_cuda):
	network = network.cuda()


# ------------------------------- 定义优化器和损失函数等 --------------------------------------

# 损失函数
train_evaluator = evaluate.DeepupeEvaluator(psnr_only=False)

# 优化器
optimizer = opt.optimizer(filter(lambda p: p.requires_grad, network.parameters()), lr=opt.lr, weight_decay=1e-5) 

# 学习率调整策略
def lr_decay(epoch):
	# 1e-2 * 0.00003 = 3e-5
	if(epoch > 30): return 0.003
	# 1e-2 * 0.01 = 1e-4
	elif(epoch > 18): return 0.01
	# 1e-2 * 0.05 = 5e-4
	elif(epoch > 8): return 0.05
	# 1e-2 * 0.1 = 1e-3
	elif(epoch >= 2): return 0.1
	# 1e-2 * 0.5 = 5e-2
	elif(epoch >= 1): return 0.5
	# 1e-2
	elif(epoch == 0): return 1
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 10, 20, 40], gamma=0.5)


# 保存本次的训练设置
with open(os.path.join(opt.checkpoints_dir, "options.pkl"), 'wb') as file:
	pickle.dump({
		"opt": opt, 
		"train_images_list": train_images_list, 
		"valid_images_list": valid_images_list, 
		"train_evaluator": train_evaluator, 
		"optimizer": optimizer, 
		"scheduler": scheduler}, file)


# ------------------------------- 开始训练 --------------------------------------

for ep in range(1, opt.total_epochs + 1):
	print()
	# 计时验证的时间
	with utils.Timer() as time_scope:
		network.train()
		train_evaluator.clear()
		# 迭代 batch
		for train_batch, (low, full, true_reflect) in enumerate(train_loader, 1):
			# 清空梯度
			optimizer.zero_grad()
			# 数据送到 GPU
			if(opt.use_cuda):
				low, full, true_reflect = low.cuda(), full.cuda(), true_reflect.cuda()
			# 经过网络
			pred_illmunination = network(low, full)
			# 根据光照估计图, 求出反射图
			pred_reflect = torch.div(full, pred_illmunination + opt.eps_value)
			pred_reflect = torch.clamp(pred_reflect, 0, 1)
			# 评估损失
			loss_value = train_evaluator.update(
				true_reflect, pred_reflect, 
				pred_illmunination=pred_illmunination, 
				gray_image=full)
			# 损失回传
			loss_value.backward()
			# w -= lr * gradient
			optimizer.step()
			# 输出信息
			output_infos = '\rTrain===> [epoch {}/{}] [batch {}/{}] [loss {:.3f}] [mse {:.4f}] [color {:.3f}] [tv {:.3f}] [psnr {:.3f}] [lr {:.5f}]'.format(
				ep, opt.total_epochs, train_batch, len(train_loader), *train_evaluator.get(), optimizer.state_dict()['param_groups'][0]['lr'])
			sys.stdout.write(output_infos)
			# 可视化一些图像
			if(train_batch % opt.visualize_batch == 0 and opt.train_batch_size % opt.visualize_size == 0):
				utils.visualize_a_batch(pred_reflect, save_path=os.path.join(opt.visualize_dir, "ep_{}_batch_{}_reflect.png".format(ep, train_batch)))
				utils.visualize_a_batch(full, save_path=os.path.join(opt.visualize_dir, "ep_{}_batch_{}_origin_full.png".format(ep, train_batch)))
				utils.visualize_a_batch(pred_illmunination, save_path=os.path.join(opt.visualize_dir, "ep_{}_batch_{}_illmunination.png".format(ep, train_batch)))
				
		# 更新学习率
		scheduler.step()
	# --------------------------- validation ------------------------
	# 验证
	if(ep % opt.valid_interval == 0):
		with utils.Timer() as time_scope:
			network.eval()
			valid_evaluator = evaluate.DeepupeEvaluator(psnr_only=True)
			with torch.no_grad():
				for valid_batch, (low, full, true_reflect) in enumerate(valid_loader, 1):
					# 数据送到 GPU
					if(opt.use_cuda):
						low, full, true_reflect = low.cuda(), full.cuda(), true_reflect.cuda()
					# 经过网络
					pred_illmunination = network(low, full)
					# 根据光照估计图, 求出反射图
					pred_reflect = torch.div(full, pred_illmunination + opt.eps_value)
					pred_reflect = torch.clamp(pred_reflect, 0, 1)
					# 评估损失
					valid_evaluator.update(true_reflect, pred_reflect)
					# 输出信息
					output_infos = '\rvalid===> [epoch {}/{}] [batch {}/{}] [loss {:.3f}] [psnr {:.3f}]'.format(
						ep, opt.total_epochs, valid_batch, len(valid_loader), *valid_evaluator.get())
					sys.stdout.write(output_infos)
				# 保存网络
				save_path = os.path.join(opt.checkpoints_dir, 'epoch_{}_psnr_{:.3f}.pth'.format(ep, valid_evaluator.get()[1]))
				print(' ---- saved to {}'.format(save_path), end="\t")
				torch.save(network.state_dict(), save_path)



				



