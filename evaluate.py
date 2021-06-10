import numpy
import torch





class DeepupeEvaluator():

	def __init__(self, psnr_only=True):
		self.psnr_only = psnr_only
		# mse 损失函数
		self.mse_loss_fn = torch.nn.MSELoss()
		# 统计一些值
		self.mean_psnr = 0
		self.mean_ssim = 0
		self.mean_loss = 0
		self.mean_mse_loss = 0
		self.mean_tv_loss = 0
		self.mean_color_loss = 0
		# 统计第几次
		self.count = 0
		# 根据 mse_loss 计算 psnr
		self.compute_psnr = lambda mse: 10 * torch.log10(1. / mse).item() if(mse > 1e-5) else 50



	def update(self, true_reflect, pred_reflect, pred_illmunination=None, gray_image=None):
		# 计数 + 1
		self.count += 1
		# mse loss
		mse_loss_value = self.mse_loss_fn(true_reflect, pred_reflect)
		self.mean_mse_loss += mse_loss_value.item()
		psnr_value = self.compute_psnr(mse_loss_value)
		self.mean_psnr += psnr_value
		# 计算损失
		total_loss_value = 1.0 * mse_loss_value
		# 如果不仅仅算 psnr 指标的话
		if(self.psnr_only == False):
			# color loss
			b, c, h, w = true_reflect.shape
			true_reflect_view = true_reflect.view(b, c, h * w).permute(0, 2, 1)
			pred_reflect_view = pred_reflect.view(b, c, h * w).permute(0, 2, 1) # 16 x (512x512) x 3
			true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
			pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
			cose_value = true_reflect_norm * pred_reflect_norm
			cose_value = torch.sum(cose_value, dim=-1) # 16 x (512x512)  # print(cose_value.min(), cose_value.max())
			color_loss = torch.mean(1 - cose_value)
			self.mean_color_loss += color_loss
			
			# tv_loss 光照平滑性损失
			# I = 0.144 * gray_image[:, 0] + 0.587 * gray_image[:, 1] + 0.299 * gray_image[:, 2]
			alpha = 1.2
			lamda = 1.5
			I = gray_image
			L = torch.log(I + 1e-4)
			dx = L[:, :, 1:, :-1] - L[:, :, 1:, 1:]
			dy = L[:, :, :-1, 1:] - L[:, :, 1:, 1:]
			# print('dx  :  {}  | dy  :  {}'.format(dx.shape, dy.shape))
			dx = lamda / (torch.pow(torch.abs(dx), alpha) + 1e-4)
			dy = lamda / (torch.pow(torch.abs(dy), alpha) + 1e-4)
			S = pred_illmunination
			x_loss = dx * torch.pow(S[:, :, 1:, :-1] - S[:, :, 1:, 1:], 2)
			y_loss = dy * torch.pow(S[:, :, :-1, 1:] - S[:, :, 1:, 1:], 2)
			tv_loss = torch.mean(x_loss + y_loss)
			self.mean_tv_loss += tv_loss

			total_loss_value += 0.2 * color_loss + 0.05 * tv_loss

		self.mean_loss += total_loss_value.item()
		return total_loss_value


	def get(self):
		if(self.count == 0):
			return 0
		if(self.psnr_only):
			return self.mean_loss * (255 ** 2) / self.count, self.mean_psnr / self.count 
		else:
			# * (255 ** 2)
			return self.mean_loss / self.count, \
				self.mean_mse_loss  / self.count, \
				self.mean_color_loss / self.count, \
				self.mean_tv_loss / self.count, \
				self.mean_psnr / self.count

	def clear(self):
		self.count = 0
		self.mean_psnr = self.mean_ssim = self.mean_loss = self.mean_tv_loss = self.mean_color_loss = 0



if __name__ == '__main__':
	import os

	os.system('python train.py')