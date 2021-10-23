import numpy as np
import argparse, tqdm, os
import torch
import torchvision.models as models
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import dataset
from dataset import TextCharData
from config import char_to_index_Only

def train(opt, model, train_loader, val_loader):
    cuda = True if torch.cuda.is_available() else False
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if cuda:
        criterion = criterion.cuda()
        model = model.cuda()

    """lr_scheduler"""
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - opt.lr_decay_epoch) / float(opt.lr_decay_epoch + 1)
        return lr_l
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)

    """training"""
    print('Start training!')
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
    max_acc = 0.
    for epoch in range(opt.initial_epoch, opt.n_epochs):
        model.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, opt.n_epochs), unit=" step")
        loss_total = 0
        acc = 0.
        correct_total = 0
        label_total = 0
        for image, label in train_loader:
            if cuda:
                image = image.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, label)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            acc = (correct_total / label_total) * 100
            loss_total += loss
            loss.backward()
            optimizer.step()

            pbar.update()
            pbar.set_postfix(
            loss=f"{loss_total:.4f}",
            Accuracy=f"{acc:.2f}%"
            )
        pbar.close()
        val_acc = validation(opt, model, val_loader)
        if max_acc <= val_acc:
            print('saved model!')
            max_acc = val_acc
            torch.save(model.state_dict(), '%s/model_epoch%d_acc%.2f.pth' % (opt.saved_model, epoch, max_acc))
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
    print('best ACC:%.2f' % (max_acc))

def validation(opt, model, val_loader):
    model.eval()
    cuda = True if torch.cuda.is_available() else False
    criterion = torch.nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="val", unit=" step")
    for image, label in val_loader:
        label_total = 0
        correct_total = 0
        loss_total = 0.
        with torch.no_grad():
            if cuda:
                image = image.cuda()
                label = label.cuda()
            pred = model(image)
            loss = criterion(pred, label)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            acc = (correct_total / label_total) * 100
            loss_total += loss
            pbar.update()
            pbar.set_postfix(
            loss=f"{loss_total:.4f}",
            Accuracy=f"{acc:.2f}%"
            )
    pbar.close()
    return acc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")
    parser.add_argument("--batch_size", type=int, default = 8, help="batch_size")
    parser.add_argument('--img_size', type=int, default = 224, help='image sizes, SE:224, B5:456')
    
    parser.add_argument('--TrainPath', default='../dataset/TrainImageOnlyChar.txt', help='path to train.txt')
    parser.add_argument('--ValidPath', default='../dataset/ValidImageOnlyChar.txt', help='path to validation.txt')
    parser.add_argument('--num_workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--shuffle', type=bool, default=True, help='set to True to have the data reshuffled at every epoch')
    parser.add_argument('--pin_memory', type=bool, default=True, help='If True, the data loader will copy Tensors into CUDA pinned memory before returning them.')
    parser.add_argument('--drop_last', type=bool, default=True, help='set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.')


    parser.add_argument("--lr", type=float, default=7e-4, help="adam: learning rate")
    parser.add_argument("--lr_decay_epoch", type=int, default=50, help="Start epoch")

    parser.add_argument('--model', default='resnext50_32x4d', help=' resnext50_32x4d or resnext101_32x8d')
    parser.add_argument('--saved_model', default='./modelsChar', help='path to model to continue training')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    os.makedirs(opt.saved_model, exist_ok=True)

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
    if opt.model == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
        in_feature = model.fc.in_features
        model.fc = nn.Linear(in_feature, len(char_to_index_Only))
    elif opt.model == 'resnext101_32x8d':
        model = models.resnext101_32x8d(pretrained=True)
        in_feature = model.fc.in_features
        model.fc = nn.Linear(in_feature, len(char_to_index_Only))
    
    train(opt, model, trainloader, vaildloader)


