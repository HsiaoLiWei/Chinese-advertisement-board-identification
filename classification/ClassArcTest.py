from sklearn import metrics
from PIL import Image
import joblib
import json, tqdm, os, copy
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import dataset
from dataset import TextClassData
from config import char_to_index
from models import classifier

def train_DL(trainloader, model):
    epochs = 15
    learning_rate = 1e-3
    classifierModel = classifier(len(char_to_index)).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifierModel.parameters(), lr = learning_rate, weight_decay = 5e-4)
    model.eval()
    classifierModel.train()
    acc_max = float('-inf')
    for epoch in range(1, epochs):
        if epoch in [4, 9]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * 0.1
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        running_loss = 0
        correct = 0.
        total = 0.
        train_bar = tqdm.tqdm(trainloader)
        for idx,(inputs, labels) in enumerate(train_bar):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)
            text_embeddings = model(inputs)
            logps = classifierModel(text_embeddings)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(logps, 1)
            correct += (pred == labels).sum().cpu().data.numpy()
            total += inputs.size(0)
            train_bar.set_description( 
                            "Epoch: %d/%d  |"%(epoch, epochs) + \
                            "Batch: %s/%d  |"%(str(idx).zfill(len(str(len(trainloader)))), len(trainloader)) +\
                            "iter: %s/%d  |"%(str((epoch-1)*len(trainloader) + idx).zfill(len(str(epochs*len(trainloader)))), epochs*len(trainloader)) +\
                            "lr: %0.6f  |"%(learning_rate) + \
                            "Train loss: %4.5f |"%(running_loss) + \
                            "Accuracy: {:.4f} %( {} / {} )".format(100.*float(correct / total), int(correct), int(total))
            )

        running_loss = 0
        accuracy = 100.*float(correct / total)
        if accuracy >= acc_max:
            torch.save(classifierModel,'./modelsArc/classifier.pkl')
            # torch.save(model.state_dict(),'./modelsArc/classifier.pkl')
            acc_max = copy.copy(accuracy)
            print("Save Best!")

def test(path, model, classifier):
    print('Testing ....')
    img_base = []
    with open(path, 'r', encoding="utf-8-sig") as f:
        imgs = f.readlines()
        # self.img_base = np.loadtxt(path, dtype = str)
        for i in imgs:
            i = i.split(' ')
            img_base += [[i[0], char_to_index[i[1].strip()]]]

    predicteds = []
    test_y = []
    model.eval()
    classifier.eval()
    with torch.no_grad():
        valid_bar = tqdm.tqdm(img_base)
        for img_path, label in valid_bar:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img = Image.open(img_path).convert("RGB")
            img = transforms.Resize((224,224))(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(img)
            img = img.to(device)
            img = img.view(-1, img.size(0), img.size(1), img.size(2))
            test_X = model(img)
            logps = classifier(test_X)
            _, pred = torch.max(logps, 1)
            predicteds += [ int(pred.cpu().data.numpy()[0]) ]

            test_y += [ label ]
            valid_bar.set_description('[Test]  Accuracy: {:.4f}'.format( metrics.accuracy_score(test_y, predicteds)))
        accuracy = metrics.accuracy_score(test_y, predicteds)
        print(accuracy)

if __name__ in "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    batch_size = 64
    
    dataset.size = 224
    train_data = TextClassData(path = '../dataset/TrainImageAll.txt', istrain = True, transform = True)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 16)
    model = torch.load('./modelsArc/Text_0.0000_1.pkl').cuda()

    ''' Fully connected Classifier '''
    train_DL(trainloader, model)
    # classifierModel = torch.load('./modelsArc/classifier.pkl').cuda()
    # test('./dataset/ValidImageChar.txt', model, classifierModel)


    