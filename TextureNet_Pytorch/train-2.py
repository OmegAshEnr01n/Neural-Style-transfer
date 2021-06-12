import argparse
import os
from VGG import VGG
from dataset import MSCOCO
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from torch import nn
from shutil import copyfile
import datetime

from Generator import Pyramid2D
import utils

'''
python train-2.py --data_dir /media/omegashenr01n/System1/Users/Shobhit/Documents/DL/COCO/train2014/ --cuda --texture Textures/Whitehousenight.jpg --save_every 500 --verbose
TODO: add visdom functionailty
TODO: tensor to image functionailty

'''


# gram matrix and loss
class GramMatrix(nn.Module):
	def forward(self, input):
		b, c, h, w = input.size()
		F = input.view(b, c, h * w)
		G = torch.bmm(F, F.transpose(1, 2))
		# G.div_(h*w) # Gatys
		G.div_(h * w * c)  # Ulyanov
		return G


class GramMSELoss(nn.Module):
	def forward(self, input, target):
		out = nn.MSELoss()(GramMatrix()(input), target)
		return(out)


def create_descriptor(device):
	vgg = VGG(pool='avg', pad=1)
	vgg.load_state_dict(torch.load(os.path.join('Models', 'vgg_conv.pth')))

	for param in vgg.parameters():
		param.requires_grad = False
	vgg.to(device)
	return vgg


def create_generator(device):
	gen = Pyramid2D()
	params = list(gen.parameters())
	total_parameters = 0
	for p in params:
		total_parameters = total_parameters + p.data.numpy().size
	print('Generator''s total number of parameters = ' + str(total_parameters))
	return gen.to(device)


def create_data(data_dir, text, device, batch_sz):
	D = MSCOCO(data_dir)
	test_loader = DataLoader(dataset=D, batch_size=batch_sz, shuffle=True)
	target = utils.load_image(text, batch_sz)
	return test_loader, target.to(device)


def create_noise_and_normalize(input, imgsz, nz):
	sz = [imgsz / 1, imgsz / 2, imgsz / 4, imgsz / 8, imgsz / 16, imgsz / 32]
	zk = [torch.rand(nz, 3, int(szk), int(szk))
		  for szk in sz]  # the 3 is for the number of channels
	for i in range(len(input)):
		input[i] = torch.add(input[i], zk[i])

	# print(len(input)) length is 6

	return input


def get_losses(device):
	style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
	content_layers = ['r42']
	loss_layers = style_layers + content_layers
	style_weights = [1e3 / n**2 for n in [64, 128, 256, 512, 512]]
	content_weights = [1e0]
	weights = style_weights + content_weights
	loss_fns = [GramMSELoss().to(device)] * len(style_layers) + \
		[nn.MSELoss().to(device)] * len(content_layers)
	return loss_layers, weights, loss_fns, style_layers, content_layers


def train(loader, target_texture, device, num_epochs, batch_sz, learning_rate, image_sz, max_iter, print_every, save_every, outf):
	descriptor_net = create_descriptor(device)
	generator_net = create_generator(device)
	generator_net.train()
	optimizer = Adam(generator_net.parameters(), learning_rate)

	loss_layers, weights, loss_fns, style_layers, content_layers = get_losses(
		device)

	for epoch in range(num_epochs):
		print("Epoch ", epoch)
		for i, data in enumerate(loader):
			data = create_noise_and_normalize(data, image_sz, 1)
			for index, dat in enumerate(data):
				data[index] = dat.to(device)

			y = generator_net(data)

			style_targets = [GramMatrix()(A).detach()
							 for A in descriptor_net(target_texture, style_layers)]
			content_targets = [A.detach() for A in descriptor_net(
				target_texture, content_layers)]
			targets = style_targets + content_targets

			out = descriptor_net(y, loss_layers)

			optimizer.zero_grad()

			layer_losses = [weights[a] * loss_fns[a]
							(A, targets[a]) for a, A in enumerate(out)]
			single_loss = (1 / (batch_sz)) * sum(layer_losses)

			single_loss.backward(retain_graph=False)

			# print(y.shape) [batch_sz, 3, 1024, 1024]

			if i % print_every == 0:
				style_loss = 0
				content_loss = 0
				sl = sum(layer_losses[:len(style_layers)]) * (1 / (batch_sz))
				cl = sum(layer_losses[len(style_layers):])*(1 / (batch_sz))
				if _verbose:
					print('Style Loss: ', sl.item())
				if _verbose:
					print('Content Loss: ', cl.item())

			if i % save_every == 0:
				# print(y.squeeze(0).shape, data[0].squeeze(0).shape) 3,1024,1024
				y_img = utils.tensor_to_img(y.squeeze(0))
				input_img = utils.tensor_to_img(data[0].squeeze(0))
				y_img.save(os.path.join(outf,str(i)+'y.png'))
				input_img.save(os.path.join(outf,str(i)+'x.png'))


			optimizer.step()

			del data, y, out
			if device == torch.device('cuda'):
				torch.cuda.empty_cache()

	input()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir',      type=str,
						default='./data/names', help='data_directory')
	parser.add_argument('--cuda', help="increase output verbosity",
						action="store_true")
	parser.add_argument('--num_epoch', type=int,
						default=5, help='data_directory')
	parser.add_argument('--batch_size', type=int,
						default=1, help='data_directory')
	parser.add_argument('--max_iter', type=int,
						default=5, help='data_directory')
	parser.add_argument('--texture', type=str,
						default='null', help='data_directory', required=True)
	parser.add_argument('--lr', type=float,
						default=0.1, help='data_directory')
	parser.add_argument('--print_every', type=int,
						default=100, help='data_directory')
	parser.add_argument('--save_every', type=int,
						default=2000, help='data_directory')

	parser.add_argument('--verbose',          type=str,   default='false',
						help='verbose for debugging')

	args = parser.parse_args()

	# Initializing the codebase
	global _verbose
	_verbose = True if args.verbose == 'true' else False
	device = torch.device('cuda') if args.cuda else torch.device('cpu')
	image_sz = 1024

	time_info = datetime.datetime.now()

	out_folder_name = 'Trained_models/' + time_info.strftime("%Y-%m-%d") + '_' + time_info.strftime("_%H%M") \
					+ args.texture[9:-4]
	if not os.path.exists(out_folder_name):
		if not os.path.exists('Trained_models'):
			os.mkdir('Trained_models')
		os.mkdir(out_folder_name)
	copyfile(__file__,
			out_folder_name + '/code.txt')

	loader, target = create_data(
		args.data_dir, args.texture, device,  args.batch_size)
	train(loader, target, device, args.num_epoch, args.batch_size, args.lr,
		  image_sz, args.max_iter, args.print_every, args.save_every, out_folder_name)


if __name__ == '__main__':
	main()
