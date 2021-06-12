import argparse
import os
from VGG import VGG
from dataset import MSCOCO
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable, Function
from torch.optim import Adam
from torch import nn
from shutil import copyfile
import datetime
import time
from Generator import Pyramid2D
import utils
from visualizer import VisdomImgUpload, VisdomPlot, VisdomText


'''
python train-3.py --data_dir /media/omegashenr01n/System1/Users/Shobhit/Documents/DL/COCO/train2014/ --cuda --texture Textures/Whitehousenight.jpg --save_every 500 --verbose

Time since start:3429.961896419525

TODO: save model

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
	gen = Pyramid2D(ch_in = 6)
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
	zk = [torch.randn(nz, 3, int(szk), int(szk))
		  for szk in sz]  # the 3 is for the number of channels
	zk = [z*255 for z in zk]
	dat = []
	for i in range(len(input)):
		dat.append(input[i].clone())
		input[i] = torch.cat((input[i], zk[i]),1) # 1,6,1024,1024

	# print(len(input)) length is 6

	return input, dat

class Normalize_gradients(Function):
    @staticmethod
    def forward(self, input):
        return input.clone()
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.mul(1./torch.norm(grad_input, p=1))
        return grad_input,

def get_losses(device):
	style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
	content_layers = ['r42']
	loss_layers = style_layers + content_layers
	style_weights = [5,6,2,2,2]
	content_weights = [1e1]
	weights = style_weights + content_weights
	loss_fns = [GramMSELoss().to(device)] * len(style_layers) + \
		[nn.MSELoss().to(device)] * len(content_layers)
	return loss_layers, weights, loss_fns, style_layers, content_layers


def train(loader, target_texture, device, num_epochs, batch_sz, learning_rate, image_sz, max_iter, print_every, save_every, outf, del_lr,viz):
	descriptor_net = create_descriptor(device)
	generator_net = create_generator(device)
	generator_net.train()
	I = Normalize_gradients.apply

	loss_layers, weights, loss_fns, style_layers, content_layers = get_losses(
		device)

	for epoch in range(num_epochs):
		print("Epoch ", epoch)
		style_targets = [GramMatrix()(A).detach()
						 for A in descriptor_net(target_texture, style_layers)]
		start = time.time()
		for i, data in enumerate(loader):
			data, inp = create_noise_and_normalize(data, image_sz, 1)
			for index, dat in enumerate(data):
				data[index] = dat.to(device)
			for index, dat in enumerate(inp):
				inp[index] = dat.to(device)
			iter = 0
			lr = learning_rate
			optimizer_style = Adam(generator_net.parameters(), lr)

			while iter < max_iter:
				y = generator_net(data)
				y_clone = y.clone().cpu()
				content_targets = [A.detach() for A in descriptor_net(
					inp[0], content_layers)]
				targets = style_targets + content_targets

				out = descriptor_net(y, loss_layers)
				optimizer_style.zero_grad()
				layer_losses = [weights[a] * loss_fns[a]
								(I(A), targets[a]) for a, A in enumerate(out)]
				sl = sum(layer_losses[:len(style_layers)]) * (1 / (batch_sz))
				cl = sum(layer_losses[len(style_layers):])*(1 / (batch_sz))
				single_loss = (1 / (batch_sz)) * sum(layer_losses)
				if iter%3 ==0:
					single_loss.backward()
				else:
					sl.backward()
				optimizer_style.step()

				# print(y.shape) [batch_sz, 3, 1024, 1024]
				if i % print_every == 0 :

					if _verbose:
						print('Style Loss: ', sl.item())
					if _verbose:
						print('Content Loss: ', cl.item())
					with open(os.path.join(outf, 'losses.txt'),'a') as txt:
						txt.write(str(sl.item())+','+str(i) + '\n')
						txt.write(str(cl.item())+','+str(i) + '\n')
					# if iter == 0: viz[0].add_img(utils.tensor_to_np(inp[0].clone().squeeze(0)), str(i)+'--'+str(iter))
					viz[0].add_img(utils.tensor_to_np(y.squeeze(0)), str(i)+'--'+str(iter)+'y')
					viz[1].plot('Style Loss', 'Style', 'Style Loss', iter + (max_iter*i), sl.item())
					viz[1].plot('Content Loss', 'Content', 'Content Loss', iter + (max_iter*i), cl.item())
					viz[2].add_txt('Time since start:' + str(time.time()-start), 'Time')
					viz[2].add_txt('Latest Loss:' + str(single_loss.item()), 'Loss')

				if i % save_every == 0 and (iter == max_iter-1 or iter == 0):
					# print(y.squeeze(0).shape, data[0].squeeze(0).shape) 3,1024,1024
					y_img = utils.tensor_to_img(y_clone.squeeze(0))
					input_img = utils.tensor_to_img(data[0].squeeze(0))
					y_img.save(os.path.join(outf,'y',str(i)+'-'+str(iter)+'y.png'))
					input_img.save(os.path.join(outf,'x',str(i)+'-'+str(iter)+'x.png'))
				iter+=1
				del out, single_loss
				if device == torch.device('cuda'):
					torch.cuda.empty_cache()

				if iter % del_lr == 0 and iter != 0:
					del optimizer_style
					lr = 0.9*lr
					optimizer_style = Adam(generator_net.parameters(), lr)






			del data, y
			if device == torch.device('cuda'):
				torch.cuda.empty_cache()


			if i%1000 == 0: torch.save(generator_net.state_dict(),os.path.join(outf,'Trained',str(i)+'Gen_net.pt'))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir',      type=str,
						default='./data/names', help='data_directory')
	parser.add_argument('--cuda', help="increase output verbosity",
						action="store_true")
	parser.add_argument('--num_epoch', type=int,
						default=1, help='data_directory')
	parser.add_argument('--batch_size', type=int,
						default=1, help='data_directory')
	parser.add_argument('--max_iter', type=int,
						default=3, help='data_directory')
	parser.add_argument('--texture', type=str,
						default='null', help='data_directory', required=True)
	parser.add_argument('--lr', type=float,
						default=0.05, help='data_directory')
	parser.add_argument('--print_every', type=int,
						default=30, help='data_directory')
	parser.add_argument('--save_every', type=int,
						default=30, help='data_directory')
	parser.add_argument('--del_lr', type=int,
						default=40, help='data_directory')
	parser.add_argument('--verbose',action="store_true",
						help='verbose for debugging')
	parser.add_argument('--viz',action="store_true",
						help='verbose for debugging')

	args = parser.parse_args()

	# Initializing the codebase
	global _verbose
	_verbose = True if args.verbose else False
	device = torch.device('cuda') if args.cuda else torch.device('cpu')
	image_sz = 1024

	viz = [VisdomImgUpload('image_viewer'),VisdomPlot(env_name='Plots'),VisdomText(env_name='Texts')]


	time_info = datetime.datetime.now()
	out_folder_name = 'Trained_models/' + time_info.strftime("%Y-%m-%d") + '_' + time_info.strftime("_%H%M") \
					+ args.texture[9:-4]
	if not os.path.exists(out_folder_name):
		if not os.path.exists('Trained_models'):
			os.mkdir('Trained_models')
		os.mkdir(out_folder_name)
		os.mkdir(os.path.join(out_folder_name,'y'))
		os.mkdir(os.path.join(out_folder_name,'x'))
		os.mkdir(os.path.join(out_folder_name,'Trained'))
	copyfile(__file__,
			out_folder_name + '/code.txt')

	loader, target = create_data(
		args.data_dir, args.texture, device,  args.batch_size)
	train(loader, target, device, args.num_epoch, args.batch_size, args.lr,
		  image_sz, args.max_iter, args.print_every, args.save_every, out_folder_name, args.del_lr, viz)


if __name__ == '__main__':
	main()
