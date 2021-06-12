from torch import nn
import torch
from dataset import MSCOCO
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse

# generator's convolutional blocks 2D


class Conv_block2D(nn.Module):
	def __init__(self, n_ch_in, n_ch_out, m=0.1):
		super(Conv_block2D, self).__init__()

		self.conv1 = nn.Conv2d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
		self.bn1 = nn.BatchNorm2d(n_ch_out, momentum=m)
		self.conv2 = nn.Conv2d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
		self.bn2 = nn.BatchNorm2d(n_ch_out, momentum=m)
		self.conv3 = nn.Conv2d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
		self.bn3 = nn.BatchNorm2d(n_ch_out, momentum=m)

	def forward(self, x):
		x = torch.cat((x[:, :, -1, :].unsqueeze(2), x,
					   x[:, :, 0, :].unsqueeze(2)), 2)
		x = torch.cat((x[:, :, :, -1].unsqueeze(3), x,
					   x[:, :, :, 0].unsqueeze(3)), 3)
		x = F.leaky_relu(self.bn1(self.conv1(x)))
		x = torch.cat((x[:, :, -1, :].unsqueeze(2), x,
					   x[:, :, 0, :].unsqueeze(2)), 2)
		x = torch.cat((x[:, :, :, -1].unsqueeze(3), x,
					   x[:, :, :, 0].unsqueeze(3)), 3)
		x = F.leaky_relu(self.bn2(self.conv2(x)))
		x = F.leaky_relu(self.bn3(self.conv3(x)))
		return x

# Up-sampling + batch normalization block


class Up_Bn2D(nn.Module):
	def __init__(self, n_ch):
		super(Up_Bn2D, self).__init__()

		self.up = nn.Upsample(scale_factor=2, mode='nearest')
		self.bn = nn.BatchNorm2d(n_ch)

	def forward(self, x):
		x = self.bn(self.up(x))
		return x


class Pyramid2D(nn.Module):
	def __init__(self, ch_in=3, ch_step=8):
		super(Pyramid2D, self).__init__()

		self.cb1_1 = Conv_block2D(ch_in, ch_step)
		self.up1 = Up_Bn2D(ch_step)

		self.cb2_1 = Conv_block2D(ch_in, ch_step)
		self.cb2_2 = Conv_block2D(2 * ch_step, 2 * ch_step)
		self.up2 = Up_Bn2D(2 * ch_step)

		self.cb3_1 = Conv_block2D(ch_in, ch_step)
		self.cb3_2 = Conv_block2D(3 * ch_step, 3 * ch_step)
		self.up3 = Up_Bn2D(3 * ch_step)

		self.cb4_1 = Conv_block2D(ch_in, ch_step)
		self.cb4_2 = Conv_block2D(4 * ch_step, 4 * ch_step)
		self.up4 = Up_Bn2D(4 * ch_step)

		self.cb5_1 = Conv_block2D(ch_in, ch_step)
		self.cb5_2 = Conv_block2D(5 * ch_step, 5 * ch_step)
		self.up5 = Up_Bn2D(5 * ch_step)

		self.cb6_1 = Conv_block2D(ch_in, ch_step)
		self.cb6_2 = Conv_block2D(6 * ch_step, 6 * ch_step)
		self.last_conv = nn.Conv2d(6 * ch_step, 3, 1, padding=0, bias=True)

	def forward(self, z):

		y = self.cb1_1(z[5])
		y = self.up1(y)
		y = torch.cat((y, self.cb2_1(z[4])), 1)
		y = self.cb2_2(y)
		y = self.up2(y)
		y = torch.cat((y, self.cb3_1(z[3])), 1)
		y = self.cb3_2(y)
		y = self.up3(y)
		y = torch.cat((y, self.cb4_1(z[2])), 1)
		y = self.cb4_2(y)
		y = self.up4(y)
		y = torch.cat((y, self.cb5_1(z[1])), 1)
		y = self.cb5_2(y)
		y = self.up5(y)
		y = torch.cat((y, self.cb6_1(z[0])), 1)
		y = self.cb6_2(y)
		y = self.last_conv(y)
		return y


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir',      type=str,
						default='./data/names', help='data_directory')
	parser.add_argument("--test", help="increase output verbosity",
						action="store_true")
	args = parser.parse_args()

	if args.test:
		D = MSCOCO(args.data_dir)
		generator = Pyramid2D()
		test_loader = DataLoader(dataset=D, batch_size=2, shuffle=True)
		for inputs in test_loader:

			out = generator.forward(inputs)
			print(out)
			print(out.shape)
			break


if __name__ == '__main__':
	main()
