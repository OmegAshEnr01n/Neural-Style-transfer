import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.autograd import Variable


def load_image(filename, batch_sz):
    input_torch = Variable(norm(prep(Image.open(filename)))).unsqueeze(
        0).repeat(batch_sz, 1, 1, 1).cuda()
    return input_torch


def tensor_to_img(tensor):
    x = postpa(tensor)
    x[x > 1] = 1
    x[x < 0] = 0
    x = x.transpose(0, 2)
    x = x.transpose(0, 1)
    x = x.detach().cpu().numpy()
    x = 255 * x
    x = x.astype(np.uint8)
    img = postpb(x)
    return img


def tensor_to_np(tensor):
    x = postpa(tensor)
    x[x > 1] = 1
    x[x < 0] = 0
    x = torch.nn.functional.adaptive_avg_pool2d(x, (512, 512))
    x = x.detach().cpu().numpy()
    x = 255 * x
    x = x.astype(np.uint8)

    return x


def get_img(data_src):
    item = Image.open(data_src).convert('RGB')
    item32 = imgtrans['regular32'](item)
    item64 = imgtrans['regular64'](item)
    item128 = imgtrans['regular128'](item)
    item256 = imgtrans['regular256'](item)
    item512 = imgtrans['regular512'](item)
    item1024 = imgtrans['regular1024'](item)
    item = [item1024.unsqueeze(0), item512.unsqueeze(0),  item256.unsqueeze(0), item128.unsqueeze(0),  item64.unsqueeze(0), item32.unsqueeze(0)]
    return item


imgtrans = {
    'old': transforms.Compose([
        transforms.RandomResizedCrop(500),
        transforms.ToTensor()
    ]),
    'regular32': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                             std=[1, 1, 1]),
        transforms.Lambda(
            lambda x: x.mul_(255))
    ]),
    'regular64': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                             std=[1, 1, 1]),
        transforms.Lambda(
            lambda x: x.mul_(255))
    ]),
    'regular128': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                             std=[1, 1, 1]),
        transforms.Lambda(
            lambda x: x.mul_(255))
    ]),
    'regular256': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                             std=[1, 1, 1]),
        transforms.Lambda(
            lambda x: x.mul_(255))
    ]),
    'regular512': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                             std=[1, 1, 1]),
        transforms.Lambda(
            lambda x: x.mul_(255))
    ]),
    'regular1024': transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                             std=[1, 1, 1]),
        transforms.Lambda(
            lambda x: x.mul_(255))
    ])  # https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
}

postpa = transforms.Compose([
    transforms.Lambda(lambda x: x.mul_(1. / 255)),
    # add imagenet mean
    transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                         std=[1, 1, 1]),

])

postpb = transforms.Compose([transforms.ToPILImage()])


prep = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),

])

norm = transforms.Compose([

    # subtract imagenet mean
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                         std=[1, 1, 1]),
    transforms.Lambda(lambda x: x.mul_(255)),
])
