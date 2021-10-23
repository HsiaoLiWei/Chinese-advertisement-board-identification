import numpy as np
import pandas as pd
import argparse
import torch, copy, tqdm, os, datetime, time, json
from torchvision import models
from torch import optim
from models import ResNet, Bottleneck, ArcMarginProduct, FocalLoss
import dataset
from dataset import TextClassData
from config import char_to_index

def resnet50(num_classes, pretrained=False):
    model = ResNet(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes)
    if pretrained:
        dict_ = models.resnet50(pretrained = pretrained)
        dict_ = dict_.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dict_.items():
            if k in model_dict and k.split('.')[0] != 'fc':
                pretrained_dict.update({k: v})
            else:
                break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def train(opt, model, metric_fc, criterion, optimizer, trainloader):
    running_loss = 0
    acc_max = float('-inf')
    for epoch in range(1, opt.epochs):
        model.train()
        if epoch in opt.lr_decay:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr * opt.lr_decay_rate
        for param_group in optimizer.param_groups:
            opt.lr = param_group['lr']
        train_bar = tqdm.tqdm(trainloader)
        correct = 0.
        total = 0.
        for idx,(inputs, labels) in enumerate(train_bar):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            logps = metric_fc(logps, labels)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, pred = torch.max(logps, 1)
            correct += (pred == labels).sum().cpu().data.numpy()
            total += inputs.size(0)
            train_bar.set_description( 
                            str(datetime.datetime.today()) + '  |' + \
                            "Epoch: %d/%d  |"%(epoch, opt.epochs) + \
                            "Batch: %s/%d  |"%(str(idx).zfill(len(str(len(trainloader)))), len(trainloader)) +\
                            "iter: %s/%d  |"%(str((epoch - 1)*len(trainloader) + idx).zfill(len(str(opt.epochs*len(trainloader)))), opt.epochs*len(trainloader)) +\
                            "lr: %0.6f  |"%(opt.lr) + \
                            "Train loss: %4.5f |"%(running_loss) + \
                            "Accuracy: {:.4f} %( {} / {} )".format(100.*float(correct / total), int(correct), int(total))
            )

        running_loss = 0
        accuracy = 100.*float(correct / total)
        if accuracy >= acc_max:
            torch.save(model,'%s/Text_%.4f_%d.pkl'%(opt.project, accuracy, epoch))
            acc_max = copy.copy(accuracy)
            print("Save Best!  %s"%('%s/Text_%.4f_%d.pkl'%(opt.project, accuracy, epoch)))


if __name__ in "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default = 400)
    parser.add_argument('--batch_size', type=int, default = 128, help='total batch size for all GPUs')
    parser.add_argument('--img_size', type=int, default = 224, help='image sizes')
    
    parser.add_argument('--TrainPath', type=str, default="../dataset/TrainImageAll.txt")
    parser.add_argument('--num_workers', type=int, default=16, help='maximum number of dataloader workers')
    parser.add_argument('--shuffle', type=bool, default=True, help='set to True to have the data reshuffled at every epoch')
    parser.add_argument('--pin_memory', type=bool, default=True, help='If True, the data loader will copy Tensors into CUDA pinned memory before returning them.')
    parser.add_argument('--drop_last', type=bool, default=True, help='set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.')

    parser.add_argument('--adam', action = 'store_false', help='use torch.optim.Adam() or torch.optim.SGD() optimizer')
    parser.add_argument('--lr', type=float, default = 1e-2, help='optimizer learning rate')
    parser.add_argument('--momentum', type=float, default = 0.9, help='optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default = 5e-4, help='optimizer weight_decay')
    parser.add_argument('--lr_decay', type=list, default = [82,200], help='When does the epoch decrease the learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default = 0.1, help='decrease rate')

    parser.add_argument('--focal_loss', action = 'store_true', help='use Focal loss or torch.nn.CrossEntropyLoss()')
    parser.add_argument('--gamma', type=float, default = 2, help='use Focal loss gamma')

    parser.add_argument('--num_classes', type=int, default = len(char_to_index), help='How many categories to predict')
    parser.add_argument('--pretrained_weight', type=bool, default=True, help='Whether to use ImageNet pretrained weight')
    parser.add_argument('--ArcFeature', type=int, default = 512, help='How many features are input to ArcMarginProduct')
    parser.add_argument('--ArcS', type=float, default = 30, help="feature scale s")
    parser.add_argument('--ArcM', type=float, default = 0.5, help="angular margin m")
    parser.add_argument('--easy_margin', type=bool, default = False)
    parser.add_argument('--project', default='./modelsArc', help='save model path')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    if not os.path.exists(opt.project):
        os.makedirs(opt.project)
    
    print("DataLoader...")
    dataset.size = opt.img_size
    train_data = TextClassData(path = opt.TrainPath, istrain = True, transform = True)
    trainloader = torch.utils.data.DataLoader(train_data, \
                                                batch_size = opt.batch_size, \
                                                shuffle = opt.shuffle, \
                                                num_workers = opt.num_workers, \
                                                pin_memory = opt.pin_memory, \
                                                drop_last = opt.drop_last
                                            )
    print("Load model parameters...")
    model = resnet50(num_classes = opt.ArcFeature, pretrained = opt.pretrained_weight)
    # model = torch.load("./modelsArc/Text_98.4700_139.pkl")
    model = model.cuda()
    metric_fc = ArcMarginProduct(opt.ArcFeature, opt.num_classes, s = opt.ArcS, m = opt.ArcM, easy_margin = opt.easy_margin)
    metric_fc = metric_fc.cuda()
    
    if opt.focal_loss:
        criterion = FocalLoss(gamma=opt.gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.adam:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr = opt.lr, weight_decay = opt.weight_decay)
    else:
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr = opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)
    
    print("Training...")
    train(opt, model, metric_fc, criterion, optimizer, trainloader)
    
