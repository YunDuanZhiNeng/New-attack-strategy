"""
Created on Sun Mar 24 17:51:08 2019

@author: aamir-mustafa
"""
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet_model_frize import *  # Imports the ResNet Model
import argparse

"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""
parser = argparse.ArgumentParser("Prototype Conformity Loss Implementation")
parser.add_argument('--epsilon', type=float, default=0.03, help="epsilon")  ##
parser.add_argument('--scale', type=float, default=1, help="epsilon")  ##
parser.add_argument('--attack', type=str, default='fgsm')
parser.add_argument('--file-name', type=str, default='Models_Softmax/CIFAR10_Softmax.pth.tar')
parser.add_argument('--outputs-name', type=str, default='outputs')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

num_classes = 10

model = frize_resnet(num_classes=num_classes, depth=110)
if True:
    model = nn.DataParallel(model).cuda()

# Loading Trained Model
softmax_filename = args.file_name
checkpoint = torch.load(softmax_filename)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loading Test Data (Un-normalized)
transform_test = transforms.Compose([transforms.ToTensor(), ])

testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                       download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, pin_memory=True,
                                          shuffle=False, num_workers=8)

# Mean and Standard Deiation of the Dataset
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t


# Attacking Images batch-wise
def attack(model, criterion, img, label, dif_scale, eps, attack_type, iters, out_adv_18=None):
    global out
    max_out, max_out_17, max_out_16, max_out_15 = dif_scale
    max_out_18 = 1.
    adv = img.detach()
    adv.requires_grad = True

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

        noise = 0
    for j in range(iterations):
        out_adv_256_15, out_adv_256_16, out_adv_256_17, out_adv_256_18, out_adv = model(
            normalize(adv.clone()))
        out = out_adv
        if args.outputs_name == 'outputs':
            out = out_adv
        elif args.outputs_name == 'outputs256_15':
            out = (out_adv_256_15 / max_out_15 + out_adv_256_16 / max_out_16 + out_adv_256_17 / max_out_17 +
                   out_adv / max_out) / 4.
        elif args.outputs_name == 'outputs256_16':
            out = (out_adv_256_16 / max_out_16 + out_adv_256_17 / max_out_17 + out_adv / max_out) / 3.
        elif args.outputs_name == 'outputs256_17':
            out = (out_adv_256_17 / max_out_17 + out_adv / max_out) / 2.
        elif args.outputs_name == 'outputs256_18':
            out = (out_adv + out_adv_256_18) / 2.
        loss = criterion(out, label)
        loss.backward()

        if attack_type == 'mim':
            adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            noise = adv.grad

        adv.data = adv.data + step * noise.sign()
        # adv.data = adv.data + step * adv.grad.sign()

        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach()


# Loss Criteria
criterion = nn.CrossEntropyLoss()
adv_acc, adv_acc_256_15, adv_acc_256_16, adv_acc_256_17, adv_acc_256_18 = 0, 0, 0, 0, 0
clean_acc, clean_acc_256_15, clean_acc_256_16, clean_acc_256_17, clean_acc_256_18 = 0, 0, 0, 0, 0

maxk = max((10,))
max_out, max_out_18, max_out_17, max_out_16, max_out_15 = 0, 0, 0, 0, 0
eps = args.epsilon  # 8 / 255  # Epsilon for Adversarial Attack
std_max_out = []
std_max_out_18 = []
std_max_out_17 = []
std_max_out_16 = []
std_max_out_15 = []
for _, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)
    pred_val_out, pred_id_out = model(normalize(img.clone().detach()))[-1].topk(maxk, 1, True, True)
    std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1]).cpu().numpy())

    pred_val_out_17, pred_id_out_17 = model(normalize(img.clone().detach()))[-3].topk(maxk, 1, True, True)
    std_max_out_17.extend((pred_val_out_17[:, 0] - pred_val_out_17[:, 1]).detach().cpu().numpy())

    pred_val_out_16, pred_id_out_16 = model(normalize(img.clone().detach()))[-4].topk(maxk, 1, True, True)
    std_max_out_16.extend((pred_val_out_16[:, 0] - pred_val_out_16[:, 1]).detach().cpu().numpy())

    pred_val_out_15, pred_id_out_15 = model(normalize(img.clone().detach()))[-5].topk(maxk, 1, True, True)
    std_max_out_15.extend((pred_val_out_15[:, 0] - pred_val_out_15[:, 1]).detach().cpu().numpy())

max_out = np.mean(std_max_out)
max_out_17 = np.mean(std_max_out_17)
max_out_16 = np.mean(std_max_out_16)
max_out_15 = np.mean(std_max_out_15)
max_out_scale = [max_out, max_out_17, max_out_16, max_out_15]
min_max_out = np.min(max_out_scale)

# if max_out>10.:
#    max_out_scale=np.array(max_out_scale)/min_max_out
# else:
#    max_out_scale=np.array(max_out_scale)/max_out

value_clean_acc = [1, 1, 1, 1]
max_out_scale = [1, 1, 1, 1]
print('value_clean_acc : ', value_clean_acc)
print('max_out : ', max_out, 'max_out_17 : ', max_out_17, 'max_out_16 : ', max_out_16, 'max-out_15 : ', max_out_15)
print('args.scale : ', args.scale)
for i, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)
    clean_acc_256_15 += torch.sum(model(normalize(img.clone().detach()))[15].argmax(dim=-1) == label).item()
    clean_acc_256_16 += torch.sum(model(normalize(img.clone().detach()))[16].argmax(dim=-1) == label).item()
    clean_acc_256_17 += torch.sum(model(normalize(img.clone().detach()))[17].argmax(dim=-1) == label).item()
    clean_acc_256_18 += torch.sum(model(normalize(img.clone().detach()))[18].argmax(dim=-1) == label).item()
    clean_acc += torch.sum(model(normalize(img.clone().detach()))[19].argmax(dim=-1) == label).item()

    adv = attack(model, criterion, img, label, dif_scale=max_out_scale, eps=eps, attack_type=args.attack, iters=10)

    adv_acc_256_15 += torch.sum(model(normalize(adv.clone().detach()))[15].argmax(dim=-1) == label).item()
    adv_acc_256_16 += torch.sum(model(normalize(adv.clone().detach()))[16].argmax(dim=-1) == label).item()
    adv_acc_256_17 += torch.sum(model(normalize(adv.clone().detach()))[17].argmax(dim=-1) == label).item()
    adv_acc_256_18 += torch.sum(model(normalize(adv.clone().detach()))[18].argmax(dim=-1) == label).item()
    adv_acc += torch.sum(model(normalize(adv.clone().detach()))[19].argmax(dim=-1) == label).item()
    right = torch.sum(model(normalize(adv.clone().detach()))[19].argmax(dim=-1) == label).item()

print('{0}\tepsilon :{1:.3%}\n acc :{2:.3%}\t acc 256_18 :{3:.3%}\t acc 256_17 :{4:.3%}\t  acc 256_16 :{5:.3%}\n '
      'acc 256_15 :{6:.3%}\t  Adv :{7:.3%}\t    Adv 256_18:{8:.3%}\t   Adv 256_17 :{9:.3%}\t   Adv 256_16 :{10:.3%}\n '
      'Adv 256_15 :{11:.3%}\t '.format(args.attack, args.epsilon, clean_acc / len(testset),
                                       clean_acc_256_18 / len(testset), clean_acc_256_17 / len(testset),
                                       clean_acc_256_16 / len(testset), clean_acc_256_15 / len(testset),
                                       adv_acc / len(testset), adv_acc_256_18 / len(testset),
                                       adv_acc_256_17 / len(testset), adv_acc_256_16 / len(testset),
                                       adv_acc_256_15 / len(testset)))
