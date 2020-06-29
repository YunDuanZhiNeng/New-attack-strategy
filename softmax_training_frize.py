"""
Created on Wed Jan 23 10:15:27 2019

@author: aamir-mustafa
This is Part 1 file for replicating the results for Paper: 
    "Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks"
Here a ResNet model is trained with Softmax Loss for 164 epochs.
"""

# Essential Imports
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utils import AverageMeter, Logger
from resnet_model_frize import *  # Imports the ResNet Model
from cutout import Cutout
from autoaugment import ImageNetPolicy, CIFAR10Policy
import torch.nn.functional as F

parser = argparse.ArgumentParser("Softmax Training for CIFAR-10 Dataset")
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--schedule', type=int, nargs='+', default=[81, 122, 140],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max-epoch', type=int, default=164)
parser.add_argument('--t-max', type=int, default=164)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')  # gpu to be used
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--filename', type=str, default='robust_model.pth.tar')  # gpu to be used
parser.add_argument('--save-filename', type=str, default='Softmax')  # gpu to be used

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


def prevent_overflow(output):  # -----------------------------
    max_output, _ = output.topk(1, 1, True, True)
    output -= max_output.float()
    return output


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + 'CIFAR-10_OnlySoftmax' + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # Data Loading
    num_classes = 10
    print('==> Preparing dataset ')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        # Cutout(1, 16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, pin_memory=True,
                                              shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, pin_memory=True,
                                             shuffle=False, num_workers=1)  # args.workers)

    # Loading the Model

    model = frize_resnet(num_classes=num_classes, depth=110, filename=args.filename)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.KLDivLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=1e-5)
    start_time = time.time()
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print('LR: %f' % (model_lr.get_lr()[-1]))

        train(trainloader, model, criterion, criterion2, optimizer, epoch, use_gpu, num_classes, model_lr)

        if args.eval_freq > 0 and (epoch + 0) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")  # Tests after every 10 epochs
            acc256_15, acc256_16, acc256_17, acc256_18, acc, err, total = test(model, testloader, use_gpu, num_classes,
                                                                               epoch)
            print("Accuracy256_15 (%): {}\n Accuracy256_16 (%): {}\t Accuracy256_17 (%): {}\t "
                  "Accuracy256_18 (%): {}\t  Accuracy (%): {}\t Error rate (%): {} "
                  "Total {}".format(acc256_15, acc256_16, acc256_17, acc256_18, acc, err, total))

            checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                          'optimizer_model': optimizer.state_dict(), }
            torch.save(checkpoint, 'Models_Softmax/frize_' + str(args.save_filename) + '_' + str(epoch) + '_' + str(
                float(acc256_15)) + '.pth.tar')

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(trainloader, model, criterion, criterion2, optimizer, epoch, use_gpu, num_classes, model_lr):
    model.train()

    losses256_15 = AverageMeter()  # 15
    losses256_16 = AverageMeter()  # 16
    losses256_17 = AverageMeter()  # 17
    losses256_18 = AverageMeter()
    losses_outputs = AverageMeter()
    losses = AverageMeter()

    # Batch-wise Training
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs256_15, outputs256_16, outputs256_17, outputs256_18, outputs = model(data)

        soft_target = outputs.detach()
        T = 10
        outputs_S_256_15 = F.log_softmax(outputs256_15 / T, dim=1)
        outputs_S_256_16 = F.log_softmax(outputs256_16 / T, dim=1)
        outputs_S_256_17 = F.log_softmax(outputs256_17 / T, dim=1)
        outputs_S_256_18 = F.log_softmax(outputs256_18 / T, dim=1)
        outputs_T = F.softmax(soft_target / T, dim=1)

        alpha = 0.0

        loss_xent256_15 = alpha * criterion(outputs256_15, labels) + \
                          (1 - alpha) * criterion2(outputs_S_256_15, outputs_T) * T * T  # 15
        loss_xent256_16 = alpha * criterion(outputs256_16, labels) + \
                          (1 - alpha) * criterion2(outputs_S_256_16, outputs_T) * T * T  # 16
        loss_xent256_17 = alpha * criterion(outputs256_17, labels) + \
                          (1 - alpha) * criterion2(outputs_S_256_17, outputs_T) * T * T  # 17
        loss_xent256_18 = alpha * criterion(outputs256_18, labels) + \
                          (1 - alpha) * criterion2(outputs_S_256_18, outputs_T) * T * T  # 18
        loss_xent_outputs = criterion(outputs, labels)

        loss_xent = loss_xent256_15 + loss_xent256_16 + loss_xent256_17 + loss_xent256_18

        optimizer.zero_grad()
        loss_xent.backward()
        optimizer.step()

        losses256_15.update(loss_xent256_15.item(), labels.size(0))  # 15
        losses256_16.update(loss_xent256_16.item(), labels.size(0))  # 16
        losses256_17.update(loss_xent256_17.item(), labels.size(0))  # 17
        losses256_18.update(loss_xent256_18.item(), labels.size(0))  # 18
        losses_outputs.update(loss_xent_outputs.item(), labels.size(0))
        losses.update(loss_xent.item(), labels.size(0))  # AverageMeter() has this param

        if (batch_idx + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.3f} ({:.3f}) Loss256_15 {:.3f} ({:.3f}) Loss256_16 {:.3f} ({:.3f})  "
                  "Loss256_17 {:.3f} ({:.3f})  Loss256_18 {:.3f} ({:.3f}) Loss_outputs {:.3f} ({:.3f})  " \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, losses256_15.val, losses256_15.avg,
                          losses256_16.val, losses256_16.avg, losses256_17.val, losses256_17.avg, losses256_18.val,
                          losses256_18.avg, losses_outputs.val, losses_outputs.avg))
    model_lr.step()  # ##################


def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct256_15, correct256_16, correct256_17, correct256_18, correct, total = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs256_15, outputs256_16, outputs256_17, outputs256_18, outputs = model(data)

            predictions256_15 = outputs256_15.data.max(1)[1]
            predictions256_16 = outputs256_16.data.max(1)[1]
            predictions256_17 = outputs256_17.data.max(1)[1]
            predictions256_18 = outputs256_18.data.max(1)[1]
            predictions = outputs.data.max(1)[1]

            total += labels.size(0)
            correct256_15 += (predictions256_15 == labels.data).sum()
            correct256_16 += (predictions256_16 == labels.data).sum()
            correct256_17 += (predictions256_17 == labels.data).sum()
            correct256_18 += (predictions256_18 == labels.data).sum()
            correct += (predictions == labels.data).sum()
    acc256_15 = correct256_15 * 100. / total
    acc256_16 = correct256_16 * 100. / total
    acc256_17 = correct256_17 * 100. / total
    acc256_18 = correct256_18 * 100. / total
    acc = correct * 100. / total

    err = 100. - acc
    return acc256_15, acc256_16, acc256_17, acc256_18, acc, err, total


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
