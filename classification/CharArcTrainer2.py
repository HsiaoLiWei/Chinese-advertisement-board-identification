import numpy as np
import pandas as pd
import copy, tqdm, os, datetime, time, argparse
import torch
from torch import nn
from torch import optim
from torchvision import models
from models import SEResNeXt, SEBottleneck, SEBottleneckX101, EfficientNetAll, ArcMarginProduct, FocalLoss
import dataset
from dataset import TextCharData
from config import char_to_index_Only

def ArcSEresnet50(num_classes, ArcFeature, pretrained=False):
    model = SEResNeXt(SEBottleneck, [3, 4, 6, 3], num_classes = num_classes, ArcFeature = ArcFeature)
    if pretrained:
        dict_ = models.resnext50_32x4d(pretrained = pretrained)
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

def ArcSEresnet101(num_classes, ArcFeature, pretrained=False):
    model = SEResNeXt(SEBottleneckX101, [3, 4, 23, 3], num_classes = num_classes, ArcFeature = ArcFeature)
    if pretrained:
        dict_ = models.resnext101_32x8d(pretrained = pretrained)
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

def train(opt, model, metric_fc, criterion, optimizer, trainloader, vaildloader):
    running_loss = 0
    acc_max = float('-inf')
    for epoch in range(1, opt.epochs):
        model.train()
        if epoch in opt.lr_decay: # effB5:[45,100]
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
            logps, embedding = model(inputs)
            embedding = metric_fc(embedding, labels)
            loss1 = criterion(embedding, labels)
            loss2 = criterionClass(logps, labels)
            loss = loss1 + loss2
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
                            "iter: %s/%d  |"%(str((epoch-1)*len(trainloader) + idx).zfill(len(str(opt.epochs*len(trainloader)))), opt.epochs*len(trainloader)) +\
                            "lr: %0.6f  |"%(opt.lr) + \
                            "Train loss: %4.5f |"%(running_loss) + \
                            "Accuracy: {:.4f} %( {} / {} )".format(100.*float(correct / total), int(correct), int(total))
            )

        running_loss = 0
        model.eval()
        with torch.no_grad():
            valid_loss=0
            accuracy=0
            correct = 0.
            total = 0.
            valid_bar = tqdm.tqdm(vaildloader)
            for idx,(inputs, labels) in enumerate(valid_bar):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs, labels = inputs.to(device), labels.to(device)
                logps,_ = model(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()
                
                _, pred = torch.max(logps, 1)
                correct += (pred == labels).sum().cpu().data.numpy()
                total += inputs.size(0)
                valid_bar.set_description('[Test]  Loss: {:.4f}    Accuracy: {:.4f} %( {} / {} )'\
                                        .format(valid_loss, 100.*float(correct / total), int(correct), int(total)))
            
            accuracy = 100.*float(correct / total)
            if accuracy >= acc_max:
                torch.save(model,'%s/Text_%.4f_%d.pkl'%(opt.project, accuracy, epoch))
                acc_max = copy.copy(accuracy)
                print("Save Best!  %s"%('%s/Text_%.4f_%d.pkl'%(opt.project, accuracy, epoch)))

if __name__ in "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default = 300)  # B5: 160, SE101: 300
    parser.add_argument('--batch_size', type=int, default = 64, help='total batch size') # B5: 32, SE101: 64
    parser.add_argument('--img_size', type=int, default = 224, help='image sizes, SE:224, B5:456')
    
    parser.add_argument('--TrainPath', type=str, default="../dataset/TrainImageOnlyChar.txt")
    parser.add_argument('--ValidPath', type=str, default="../dataset/ValidImageOnlyChar.txt")
    parser.add_argument('--num_workers', type=int, default=32, help='maximum number of dataloader workers')
    parser.add_argument('--shuffle', type=bool, default=True, help='set to True to have the data reshuffled at every epoch')
    parser.add_argument('--pin_memory', type=bool, default=True, help='If True, the data loader will copy Tensors into CUDA pinned memory before returning them.')
    parser.add_argument('--drop_last', type=bool, default=True, help='set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.')

    parser.add_argument('--adam', action = 'store_false', help='use torch.optim.Adam() or torch.optim.SGD() optimizer')
    parser.add_argument('--lr', type=float, default = 1e-2, help='optimizer learning rate')
    parser.add_argument('--momentum', type=float, default = 0.9, help='optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default = 5e-4, help='optimizer weight_decay')
    parser.add_argument('--lr_decay', type=list, default = [82,200], help='When does the epoch decrease the learning rate') # B5: [49, 100], SE101: [82,200]
    parser.add_argument('--lr_decay_rate', type=float, default = 0.1, help='decrease rate')

    parser.add_argument('--focal_loss', action = 'store_true', help='use Focal loss or torch.nn.CrossEntropyLoss()')
    parser.add_argument('--gamma', type=float, default = 2, help='use Focal loss gamma(Arc loss)')
    parser.add_argument('--gammaClass', type=float, default = 1.5, help='use Focal loss gamma(class loss)')

    parser.add_argument('--model_name', type=str, default = "ArcSEResNet101", help='ArcSEResNet50, ArcSEResNet101, EfficientNet')
    parser.add_argument('--num_classes', type=int, default = len(char_to_index_Only), help='How many categories to predict')
    parser.add_argument('--pretrained_weight', type=bool, default=True, help='Whether to use ImageNet pretrained weight')
    parser.add_argument('--EfficientNet_mode', type=str, default="efficientnet-b5", help='Whether to use ImageNet pretrained weight')
    parser.add_argument('--feature', type=int, default = 2048, help='B0:1280, B3:1536, B5:2048, B6:2304')
    parser.add_argument('--ArcFeature', type=int, default = 1024, help='How many features are input to ArcMarginProduct')
    parser.add_argument('--ArcS', type=float, default = 30, help="feature scale s")
    parser.add_argument('--ArcM', type=float, default = 0.5, help="angular margin m")
    parser.add_argument('--easy_margin', type=bool, default = False)
    parser.add_argument('--project', default='./modelsArc2', help='save model path')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    if not os.path.exists(opt.project):
        os.makedirs(opt.project)

    print("DataLoader...")
    dataset.size = opt.img_size
    train_data = TextCharData(path = opt.TrainPath, istrain = True, transform = True)
    valid_data = TextCharData(path = opt.ValidPath, istrain = False, transform = True)

    trainloader = torch.utils.data.DataLoader(train_data, \
                                                batch_size = opt.batch_size, \
                                                shuffle = opt.shuffle, \
                                                num_workers = opt.num_workers, \
                                                pin_memory = opt.pin_memory, \
                                                drop_last = opt.drop_last
                                            )

    vaildloader = torch.utils.data.DataLoader(valid_data, \
                                                batch_size = opt.batch_size, \
                                                shuffle = False, \
                                                num_workers = opt.num_workers, \
                                                pin_memory = opt.pin_memory, \
                                                drop_last = False
                                            )

    print("Load model parameters...")
    if opt.model_name == "ArcSEResNet50":
        model = ArcSEresnet50(num_classes = opt.num_classes, ArcFeature = opt.ArcFeature, pretrained = True)
    elif opt.model_name == "ArcSEResNet101":
        model = ArcSEresnet101(num_classes = opt.num_classes, ArcFeature = opt.ArcFeature, pretrained = True)
    elif opt.model_name == "EfficientNet":
        model = EfficientNetAll(mode = opt.EfficientNet_mode, \
                                advprop = False, \
                                num_classes = opt.num_classes, \
                                feature = opt.feature, \
                                ArcFeature = opt.ArcFeature
                                )
    # model = torch.load("modelsArcEfficientNetB5Only_sgd/Text_92.7875_51.pkl")

    model = model.cuda()
    if len(opt.device) > 2:
        model = nn.DataParallel(model)

    metric_fc = ArcMarginProduct(opt.ArcFeature, opt.num_classes, s = opt.ArcS, m = opt.ArcM, easy_margin = opt.easy_margin)
    metric_fc = metric_fc.cuda()
    
    if opt.focal_loss:
        criterion = FocalLoss(gamma = opt.gamma) #2.0
        criterionClass = FocalLoss(gamma = opt.gammaClass) #1.5
    else:
        criterion = torch.nn.CrossEntropyLoss()
        criterionClass = torch.nn.CrossEntropyLoss()

    if opt.adam:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr = opt.lr, weight_decay = opt.weight_decay)
    else:
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr = opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)
    
    print("Training...")
    train(opt, model, metric_fc, criterion, optimizer, trainloader, vaildloader)


    