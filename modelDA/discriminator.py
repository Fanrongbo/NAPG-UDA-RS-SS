import torch.nn as nn
import torch.nn.functional as F
import torch

class layoutDiscriminator_wgan(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(layoutDiscriminator_wgan, self).__init__()

		self.conv1 = nn.Conv2d(5, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)
		x = x.view(1,-1)
		return x
class FCDiscriminatorHigh(nn.Module):
	def __init__(self, num_classes, ndf=32):
		super(FCDiscriminatorHigh, self).__init__()
		self.FCDiscriminator=nn.Sequential(
			nn.Conv2d(num_classes, ndf, kernel_size=3, stride=4, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=4, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=4, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf*8, ndf*8, kernel_size=1, stride=2, padding=0),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1),
			nn.Sigmoid()
		)
	def forward(self,x):
		x=self.FCDiscriminator(x)
		return x.view(-1,1)
class FCDiscriminatorHigh2(nn.Module):
	def __init__(self, num_classes, ndf=32):
		super(FCDiscriminatorHigh2, self).__init__()
		self.FCDiscriminator=nn.Sequential(
			nn.Conv2d(num_classes, ndf, kernel_size=3, stride=4, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=4, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=4, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf*8, ndf*8, kernel_size=1, stride=2, padding=0),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1),
			# nn.Sigmoid()
		)
	def forward(self,x):
		x=self.FCDiscriminator(x)
		return x
class FCDiscriminatorLow(nn.Module):
	def __init__(self, num_classes, ndf=32):
		super(FCDiscriminatorLow, self).__init__()
		self.FCDiscriminator=nn.Sequential(
			nn.Conv2d(num_classes, ndf*8, kernel_size=3, stride=4, padding=1),#8
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 8, ndf * 4, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 4, ndf * 4, kernel_size=1, stride=2, padding=0),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 4, 1, kernel_size=1, stride=1),
			nn.Sigmoid()
		)
	def forward(self,x):
		x=self.FCDiscriminator(x)
		# x = self.FCDiscriminator(x)
		return x.view(-1, 1)
class FCDiscriminatorHighMask(nn.Module):
	def __init__(self, num_classes, ndf=32):
		super(FCDiscriminatorHighMask, self).__init__()
		self.FCDiscriminator=nn.Sequential(
			nn.Conv2d(num_classes, ndf, kernel_size=3, stride=2, padding=1),
			# nn.BatchNorm2d(ndf),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),

			nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=1),
			# nn.BatchNorm2d( ndf*2),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),

			nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=1),
			# nn.BatchNorm2d(ndf*4),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),

			nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1),
			# nn.BatchNorm2d(ndf*8),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),

			nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1),
			# nn.Sigmoid()
		)
		self.up_sample = nn.Upsample(scale_factor=16, mode='bilinear')
		self.Sigmoid = nn.Sigmoid()
	def forward(self,x):
		x=self.FCDiscriminator(x)
		x = self.up_sample(x)
		x = self.Sigmoid(x)
		return x
# 定义判别器网络
class DiscriminatorProto(nn.Module):
	def __init__(self,num_classes):
		super(DiscriminatorProto, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(num_classes * 128, 512),
			nn.LeakyReLU(),
			nn.Linear(512, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = x.view(x.size(0), -1)  # 展平
		out = self.model(x)
		return out
class FCDiscriminatorLowMask(nn.Module):
	def __init__(self, num_classes, ndf=32):
		super(FCDiscriminatorLowMask, self).__init__()
		self.FCDiscriminator=nn.Sequential(
			nn.Conv2d(num_classes, ndf*8, kernel_size=3, stride=1, padding=1),#8
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 8, ndf * 4, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 4, 1, kernel_size=1, stride=1),
		)
		self.up_sample = nn.Upsample(scale_factor=16, mode='bilinear')
		self.Sigmoid=nn.Sigmoid()
	def forward(self,x):
		x=self.FCDiscriminator(x)
		x=self.up_sample(x)
		x=self.Sigmoid(x)
		# x = self.FCDiscriminator(x)
		return x

class FCDiscriminatorLow2(nn.Module):
	def __init__(self, num_classes, ndf=32):
		super(FCDiscriminatorLow2, self).__init__()
		self.FCDiscriminator=nn.Sequential(
			nn.Conv2d(num_classes, ndf*8, kernel_size=3, stride=4, padding=1),#8
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 8, ndf * 4, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 4, ndf * 4, kernel_size=1, stride=2, padding=0),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(ndf * 4, 1, kernel_size=1, stride=1),
			# nn.Sigmoid()
		)
	def forward(self,x):
		x=self.FCDiscriminator(x)
		# x = self.FCDiscriminator(x)
		return x
class FCDiscriminator(nn.Module):
	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()
		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=4, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=4, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=4, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=2, padding=1)
		if ndf==256:
			self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=1)
		else:
			self.classifier = nn.Sequential(nn.Conv2d(ndf*8, 1, kernel_size=3, stride=2, padding=1),
											nn.LeakyReLU(negative_slope=0.2, inplace=True),
											)
			self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		# self.sigmoid = nn.Sigmoid()
		# self.flatten=nn.()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		# print(x.shape)
		x = self.classifier(x)
		#x = self.up_sample(x)
		# x = self.sigmoid(x)
		x = torch.flatten(x,start_dim=1,end_dim=3)


		return x
# 梯度逆转层
class GradientReversalFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		return -grad_output

class GradientReversalLayer(nn.Module):
	def forward(self, x):
		return GradientReversalFunction.apply(x)

# 领域鉴别器
class DomainDiscriminator(nn.Module):
	def __init__(self,num_classes, ndf = 64):
		super(DomainDiscriminator, self).__init__()
		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		return self.main(x)


