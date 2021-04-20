import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn

import os
import os.path as osp
import tqdm
import argparse
import json
import logging
import time

from classification.image_produce import ImageDataset
from models.model import resnet50, Classifier
from models.discriminator import FCDiscriminator


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cnn_path', default='/home/ains/An/histologic/TJ_CH/lung_proj/DA/cnn.json', metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format')
parser.add_argument('--save_path', default='/home/ains/An/histologic/TJ_CH/lung/256_patch_DA/result/', metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--pre_trainpath', default='/home/ains/An/histologic/SLFCD-master/256_patches/model_save/best.ckpt', metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument("--learning-rate", type=float, default=1e-3, help="Base learning rate for training with polynomial decay.")
parser.add_argument("--learning-rate-D", type=float, default=1e-3, help="Base learning rate for discriminator.")
parser.add_argument('--pre_train', default=True, type=bool, help='Path to the saved models')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0', type=str, help='comma'
                    ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                    ' and GPU_1, default 0.')

args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.device_ids)) if torch.cuda.is_available() else torch.device('cpu')

with open(args.cnn_path, 'r') as f:
    cnn = json.load(f)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        # assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = 1

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        zeros = torch.zeros([batch_size, labels_length]).cuda()
        label_onehot = zeros.scatter_(1, new_label, 1)
        # label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, cnn['epoch'], 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, cnn['epoch'], 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def valid_epoch(summary, model, loss_fn, dataloader_valid, classifier2):
    model.eval()
    classifier2.eval()
    steps = len(dataloader_valid)
    batch_size = dataloader_valid.batch_size
    dataiter_valid = iter(dataloader_valid)

    loss_sum = 0
    acc_sum = 0
    with torch.no_grad():
        for step in range(steps):
            data_valid, target_valid = next(dataiter_valid)
            data_valid = Variable(data_valid.float().cuda(non_blocking=True))
            target_valid = Variable(target_valid.float().cuda(non_blocking=True))

            feature = model(data_valid)
            output = classifier2(feature)
            output = torch.squeeze(output) # important
            loss = loss_fn(output, target_valid)

            probs = output.sigmoid()
            predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
            acc_data = (predicts == target_valid).type(
                torch.cuda.FloatTensor).sum().data * 1.0 / batch_size
            loss_data = loss.data

            loss_sum += loss_data
            acc_sum += acc_data

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary


def main():
    """Create the model and start the training."""

    cudnn.enabled = True

    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    with open(os.path.join(args.save_path, 'cnn.json'), 'w') as f:
        json.dump(cnn, f, indent=1)

    # Create network
    model = resnet50(pretrained=False, num_classes=2).to(DEVICE)
    classifer1 = Classifier(num_classes=1).to(DEVICE)
    classifer2 = Classifier(num_classes=2).to(DEVICE)

    # init D
    model_D = FCDiscriminator(in_ch=2048).to(DEVICE)

    batch_size = cnn['batch_size']
    strain_dataset = ImageDataset(cnn['data_source_train'], cnn['image_size'], cnn['crop_size'], cnn['normalize'])
    ttrain_dataset = ImageDataset(cnn['data_target_train'], cnn['image_size'], cnn['crop_size'], cnn['normalize'])
    svalid_dataset = ImageDataset(cnn['data_source_valid'], cnn['image_size'], cnn['crop_size'], cnn['normalize'])
    tvalid_dataset = ImageDataset(cnn['data_target_valid'], cnn['image_size'], cnn['crop_size'], cnn['normalize'])

    strain_dataloader = DataLoader(strain_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True)
    ttrain_dataloader = DataLoader(ttrain_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True)

    svalid_dataloader = DataLoader(svalid_dataset, batch_size=batch_size, num_workers=args.num_workers)
    tvalid_dataloader = DataLoader(tvalid_dataset, batch_size=batch_size, num_workers=args.num_workers)

    len_dataloader = min(len(strain_dataloader), len(ttrain_dataloader))

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
    optimizer.zero_grad()

    optimizer_D = optim.SGD(model_D.parameters(), lr=args.learning_rate_D, momentum=0.9)
    optimizer_D.zero_grad()

    class_loss = FocalLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1
    # set up tensor board
    summary_writer = SummaryWriter(args.save_path)
    summary_train = {'epoch': 0}
    summary_tvalid = {'loss': float('inf'), 'acc': 0}
    loss_valid_best = float('inf')

    for i_iter in range(cnn['epoch']):

        sdataiter_train = iter(strain_dataloader)
        tdataiter_train = iter(ttrain_dataloader)
        time_now = time.time()
        model.train()
        model.to(DEVICE)

        model_D.train()
        model_D.to(DEVICE)

        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        adjust_learning_rate(optimizer, i_iter)

        for step in range(len_dataloader):
            src_data, src_lb = next(sdataiter_train)
            trg_data, trg_lb = next(tdataiter_train)

            optimizer.zero_grad()
            optimizer_D.zero_grad()

            src_data, src_lb, trg_data, trg_lb = src_data.to(DEVICE), src_lb.to(DEVICE), \
                                                 trg_data.to(DEVICE), trg_lb.to(DEVICE)
            # .to(dtype=torch.long)
            # train G
            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # train with source
            src_feature = model(src_data)
            pred = classifer1(src_feature)
            pred = torch.squeeze(pred)
            loss1 = bce_loss(pred, src_lb)

            # proper normalization
            loss1.backward()

            probs = pred.sigmoid()
            predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
            acc_data = (predicts == src_lb).type(torch.cuda.FloatTensor).sum().data * 1.0 / batch_size
            loss_s_data = loss1.data

            # train with target
            trg_feature = model(trg_data)
            D_out = model_D(trg_feature)

            loss_adv_target1 = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(DEVICE))
            loss2 = 0.01 * loss_adv_target1

            loss2.backward()
            loss_adv_target_value1 += loss_adv_target1.item()

            # train D
            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            src_feature = src_feature.detach()

            D_out1 = model_D(src_feature)
            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(DEVICE))
            loss_D1 = loss_D1 / 2
            loss_D1.backward()
            loss_D_value1 += loss_D1.item()

            # train with target
            trg_feature = trg_feature.detach()

            D_out2 = model_D(trg_feature)
            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(DEVICE))
            loss_D2 = loss_D2 / 2
            loss_D2.backward()
            loss_D_value1 += loss_D2.item()

            optimizer.step()
            optimizer_D.step()

            time_spent = time.time() - time_now

            if step % 100 == 0:
                print('{}, Epoch : {}, Training Loss : {:.5f}, Training Acc : {:.3f}, Run Time : {:.2f}'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'] + 1, loss_s_data, acc_data, time_spent))

                summary_writer.add_scalar('train/s_loss', loss_s_data, summary_train['epoch'])
                summary_writer.add_scalar('train/acc', acc_data, summary_train['epoch'])


        torch.save({'epoch': summary_train['epoch'],
                    'state_dict1': model.state_dict(),
                    'state_dict2': classifer1.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))

        time_now = time.time()
        summary_valid = valid_epoch(summary_tvalid, model, bce_loss, tvalid_dataloader, classifer1)
        time_spent = time.time() - time_now
        print('{}, Epoch: {}, Validation Loss: {:.5f}, Validation ACC: {:.3f}, Run Time: {:.2f}'
                     .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                             summary_valid['loss'], summary_valid['acc'], time_spent))

        summary_writer.add_scalar('valid/loss', summary_valid['loss'], summary_train['epoch'])
        summary_writer.add_scalar('valid/acc', summary_valid['acc'], summary_train['epoch'])

        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']

            torch.save({'epoch': summary_train['epoch'],  'state_dict1': model.state_dict(),
                    'state_dict2': classifer1.state_dict()},
                       os.path.join(args.save_path, 'best.ckpt'))

        summary_train['epoch'] += 1
    summary_writer.close()


if __name__ == '__main__':
    main()
