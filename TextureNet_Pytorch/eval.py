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






def create_generator(device, state_dict):
	gen = Pyramid2D(ch_in = 6)
	params = list(gen.parameters())
	total_parameters = 0
	for p in params:
		total_parameters = total_parameters + p.data.numpy().size
	print('Generator''s total number of parameters = ' + str(total_parameters))
	gen.load_state_dict(torch.load(state_dict))
	gen.eval()
	return gen.to(device)


def create_data(data_dir, device, batch_sz):
	testimg = utils.get_img(data_dir)

	return testimg


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


def eval(device, content_img, image_sz,outf,state_dict):
	generator_net = create_generator(device,state_dict)

	data, inp = create_noise_and_normalize(content_img, image_sz, 1)

	for index, dat in enumerate(data):
		data[index] = dat.to(device)
	for index, dat in enumerate(inp):
		inp[index] = dat.to(device)

	y = generator_net(data)

	y_img = utils.tensor_to_img(y.squeeze(0))
	y_img.save(os.path.join(outf))

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--cuda', help="increase output verbosity",
						action="store_true")

	parser.add_argument('--batch_size', type=int,
						default=1, help='data_directory')
	parser.add_argument('--max_iter', type=int,
						default=3, help='data_directory')

	parser.add_argument('--load_model', type=str,
						default='null', help='data_directory', required=True)
	parser.add_argument('--load_img', type=str,
						default='null', help='data_directory', required=True)
	parser.add_argument('--outfile', type=str,
						default='null', help='data_directory', required=True)

	parser.add_argument('--del_lr', type=int,
						default=40, help='data_directory')
	parser.add_argument('--verbose',action="store_true",
						help='verbose for debugging')


	args = parser.parse_args()

	# Initializing the codebase
	global _verbose
	_verbose = True if args.verbose else False
	device = torch.device('cuda') if args.cuda else torch.device('cpu')
	image_sz = 1024


	time_info = datetime.datetime.now()


	testimg = create_data(
		args.load_img, device,  args.batch_size)
	eval(device,testimg, image_sz, args.outfile,args.load_model)


if __name__ == '__main__':
	main()
