from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

import os
import argparse

class MSCOCO(Dataset):

	def __init__(self, data_src):

		self.X = []
		self.img_size_hr = 800
		self.imgtrans = {
			'unregular32': transforms.Compose([
				transforms.Resize((32,32)),
				transforms.ToTensor(),


			]),
			'unregular64': transforms.Compose([
				transforms.Resize((64,64)),
				transforms.ToTensor(),


			]),
			'unregular128': transforms.Compose([
				transforms.Resize((128,128)),
				transforms.ToTensor(),


			]),
			'unregular256': transforms.Compose([
				transforms.Resize((256,256)),
				transforms.ToTensor(),


			]),
			'unregular512': transforms.Compose([
				transforms.Resize((512,512)),
				transforms.ToTensor(),


			]),
			'unregular1024': transforms.Compose([
				transforms.Resize((1024,1024)),
				transforms.ToTensor(),

			]),
			'regular32': transforms.Compose([
				transforms.Resize((32,32)),
				transforms.ToTensor(),

				transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
									 std=[1, 1, 1]),
				transforms.Lambda(
					lambda x: x.mul_(255))
			]),
			'regular64': transforms.Compose([
				transforms.Resize((64,64)),
				transforms.ToTensor(),

				transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
									 std=[1, 1, 1]),
				transforms.Lambda(
					lambda x: x.mul_(255))
			]),
			'regular128': transforms.Compose([
				transforms.Resize((128,128)),
				transforms.ToTensor(),

				transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
									 std=[1, 1, 1]),
				transforms.Lambda(
					lambda x: x.mul_(255))
			]),
			'regular256': transforms.Compose([
				transforms.Resize((256,256)),
				transforms.ToTensor(),

				transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
									 std=[1, 1, 1]),
				transforms.Lambda(
					lambda x: x.mul_(255))
			]),
			'regular512': transforms.Compose([
				transforms.Resize((512,512)),
				transforms.ToTensor(),

				transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
									 std=[1, 1, 1]),
				transforms.Lambda(
					lambda x: x.mul_(255))
			]),
			'regular1024': transforms.Compose([
				transforms.Resize((1024,1024)),
				transforms.ToTensor(),

				transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
									 std=[1, 1, 1]),
				transforms.Lambda(
					lambda x: x.mul_(255))
			])  # https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
		}
		self.data_src = data_src
		for filename in os.listdir(self.data_src):
			self.X.append(filename)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		item = Image.open(self.data_src + self.X[idx]).convert('RGB')
		item32 = self.imgtrans['regular32'](item)
		item64 = self.imgtrans['regular64'](item)
		item128 = self.imgtrans['regular128'](item)
		item256 = self.imgtrans['regular256'](item)
		item512 = self.imgtrans['regular512'](item)
		item1024 = self.imgtrans['regular1024'](item)
		# item32 = self.imgtrans['unregular32'](item)
		# item64 = self.imgtrans['unregular64'](item)
		# item128 = self.imgtrans['unregular128'](item)
		# item256 = self.imgtrans['unregular256'](item)
		# item512 = self.imgtrans['unregular512'](item)
		# item1024 = self.imgtrans['unregular1024'](item)
		item = (item1024,item512,  item256, item128,  item64,item32 )
		return item


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir',      type=str,
						default='./data/names', help='data_directory')
	parser.add_argument("--test", help="increase output verbosity",
						action="store_true")
	args = parser.parse_args()

	if args.test:
		D = MSCOCO(args.data_dir)
		print(D[0][1].shape, len(D))


if __name__ == '__main__':
	main()
	# python dataset.py --data_dir /media/omegashenr01n/System1/Users/Shobhit/Documents/DL/COCO/train2014/ --test
