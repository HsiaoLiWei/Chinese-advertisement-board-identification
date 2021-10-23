'''
Modified Date: 2021/10/23

Author: Li-Wei Hsiao

mail: nfsmw308@gmail.com
'''
from PIL import Image
import tqdm, os, math, csv, copy, cv2, random
import joblib
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data.dataset import Dataset
from config import index_to_char_Only, index_to_char_New, index_to_char
from models import ResNet, Bottleneck, SEResNeXt, SELayerX, SEBottleneck, SEBottleneckX101, EfficientNetAll, classifier
from efficientnet_pytorch import EfficientNet

test_transforms = transforms.Compose([  transforms.Resize((224,224)),
                                        # transforms.CenterCrop(224),
                                        # transforms.RandomCrop(224),   # 224
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
                                    
test_transforms2 = transforms.Compose([  transforms.Resize((456,456)),
                                        # transforms.CenterCrop(224),
                                        # transforms.RandomCrop(224),   # 224
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

class TextData(Dataset):
    def __init__(self, path, transform=None):
        self.img_base = []
        with open(path, 'r', encoding="utf-8-sig") as f:
            self.imgs = f.readlines()
        for i in self.imgs:
            self.img_base += [i.strip().replace("\\", "/")]

    def __getitem__(self, index):
        path = self.img_base[index]
        img = Image.open(path).convert("RGB")
        img1 = test_transforms(img)
        img2 = test_transforms2(img)
        return path, img1, img2

    def __len__(self):
        return len(self.img_base)

if __name__ in "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    print('Determine whether it is a string......')
    pred_dict = {}
    pred_dict2 = {}
    public_name = np.loadtxt('../dataset/private.txt', dtype = str)
    for i in public_name:
        img_name = i.split('/')[-1]
        pred_dict[img_name] = []
    pred_dict2 = copy.deepcopy(pred_dict)

    print('Testing Text...')
    valid_data = TextData(path = '../dataset/single_private.txt', transform = True)
    vaildloader = torch.utils.data.DataLoader(valid_data, batch_size = 4, shuffle = False, num_workers = 8, pin_memory = False)
    
    model1 = models.resnext50_32x4d(pretrained=True)
    in_feature = model1.fc.in_features
    model1.fc = nn.Linear(in_feature, len(index_to_char_Only))
    model1.load_state_dict(torch.load('./private_model/model_epoch84_acc90.41.pth'))
    model1.cuda().eval()

    model2 = models.resnext101_32x8d(pretrained=True)
    in_feature = model2.fc.in_features
    model2.fc = nn.Linear(in_feature, len(index_to_char_Only))
    model2.load_state_dict(torch.load('./private_model/model_epoch92_acc90.92.pth'))
    model2.cuda().eval()

    modelArc2 = torch.load('./private_model/Text_91.9476_52.pkl').cuda().eval()
    modelArc2_2 = torch.load('./private_model/Text_91.8241_68.pkl').cuda().eval()


    modelArcB5_S6 = EfficientNetAll(mode = "efficientnet-b5", \
                                advprop = False, \
                                num_classes = len(index_to_char_New), \
                                feature = 2048, \
                                ArcFeature = 1024
                                )
    checkpoint = torch.load('./private_model/Text_93.6635_30.pkl', map_location=lambda storage, loc: storage.cuda(0))
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.state_dict().items()}
    modelArcB5_S6.load_state_dict(state_dict)
    modelArcB5_S6.cuda().eval()

    modelArcB5_S5 = EfficientNetAll(mode = "efficientnet-b5", \
                                advprop = False, \
                                num_classes = len(index_to_char_New), \
                                feature = 2048, \
                                ArcFeature = 1024
                                )
    checkpoint = torch.load('./private_model/Text_92.6507_51.pkl', map_location=lambda storage, loc: storage.cuda(0))
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.state_dict().items()}
    modelArcB5_S5.load_state_dict(state_dict)
    modelArcB5_S5.cuda().eval()

    modelArcB5_alun = EfficientNetAll(mode = "efficientnet-b5", \
                                advprop = False, \
                                num_classes = len(index_to_char_New), \
                                feature = 2048, \
                                ArcFeature = 1024
                                )
    checkpoint = torch.load('./private_model/Text_93.3547_7.pkl', map_location=lambda storage, loc: storage.cuda(0))
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.state_dict().items()}
    modelArcB5_alun.load_state_dict(state_dict)
    modelArcB5_alun.cuda().eval()

    # modelArcB5_S6 = torch.load('./private_model/Text_93.6635_30.pkl').cuda().eval()
    # modelArcB5_S5 = torch.load('./private_model/Text_92.6507_51.pkl').cuda().eval()
    # modelArcB5_alun = torch.load('./private_model/Text_93.3547_7.pkl').cuda().eval()
    
    ArcModel_Noisy = torch.load('./private_model/Text_98.4700_139.pkl').cuda().eval()
    classifier_Noisy = torch.load('./private_model/classifier9847.pkl').cuda().eval()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        valid_bar = tqdm.tqdm(vaildloader)
        for img_path, img, img1  in valid_bar:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img, img1 = img.to(device), img1.to(device)
            
            logps = softmax(model1(img))
            logps += softmax(model2(img))
            logps += softmax(modelArc2(img)[0])
            logps += softmax(modelArc2_2(img)[0])
            logps = torch.cat((logps, torch.zeros(logps.size(0), 1).cuda()), dim=1)
            logps = softmax(modelArcB5_S6(img1)[0])
            logps += softmax(modelArcB5_S5(img1)[0])
            logps += softmax(modelArcB5_alun(img1)[0])
            
            _, pred = torch.max(logps, 1)
            predicted = [ index_to_char_New[str(int(i))] for i in pred.cpu().data.numpy() ] 
            
            test_X = ArcModel_Noisy(img)  
            logps = classifier_Noisy(test_X)
            _, pred2 = torch.max(logps, 1)
            predicted2 = [ index_to_char[str(int(i))] for i in pred2.cpu().data.numpy() ] 
            
            for path, pred, pred2 in zip(img_path, predicted, predicted2):
                img_name = path.split('/')[-1].split('_')
                imgP = img_name[0] + '_' + img_name[1] + '_' +img_name[2] + '.png'
                if pred_dict[imgP] != "###":
                    pred = pred if pred != "" else "###"
                    pred_dict[imgP].append(pred)

                if pred_dict2[imgP] != "###":
                    pred2 = pred2 if pred2 != "" else "###"
                    pred_dict2[imgP].append(pred2)


    print('Organize the dictionary...')
    pred_list = []
    for (img_name, pred), (img_name2, pred2) in zip(pred_dict.items(), pred_dict2.items()):
        pred_str = ''
        # pred_str2 = ''
        if pred2 == ["###"]:
            pred_list.append("###")
        elif pred == []:
            pred_list.append("###")
        else:
            for str_ in pred:
                pred_str += str_
            pred_list.append(pred_str)
    
    print('Save result...')
    with open('../dataset/private/Task2_Private_String_Coordinate.csv', newline='', encoding="utf-8") as csvfile:
        with open('./private.csv', 'w', encoding="utf-8-sig", newline = "") as result_csv:
            rows = csv.reader(csvfile, delimiter=':')
            writer = csv.writer(result_csv)
            for i,row in enumerate(rows):
                pred_result = pred_list[i]
                result = row[0].split(',')
                result.append(pred_result)
                # print(result)
                pred_result = "###" if pred_result == "" else pred_result
                writer.writerow([result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], pred_result])
