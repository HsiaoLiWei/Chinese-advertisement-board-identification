'''
Modified Date: 2021/10/23

Author: Li-Wei Hsiao

mail: nfsmw308@gmail.com
'''
import argparse, cv2, time, json, csv, copy
from pathlib import Path
from PIL import Image
import numpy as np
from numpy import random
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
import torchvision


import sys, os
sys.path.append("..")
sys.path.append("../classification")
from classification.config import index_to_char_Only, index_to_char_New, index_to_char
from classification.models import SEBottleneckX101, SELayerX, SEResNeXt, Bottleneck, ResNet, EfficientNetAll, classifier
import efficientnet_pytorch

import models
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

class TextData(Dataset):
    def __init__(self, img_list):
        self.img_base = img_list

    def __getitem__(self, index):
        img = self.img_base[index]
        img = Image.fromarray(np.uint8(img))

        img1 = torchvision.transforms.Resize((224,224))(img)
        img1 = torchvision.transforms.ToTensor()(img1)
        img1 = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(img1)

        img2 = torchvision.transforms.Resize((456,456))(img)
        img2 = torchvision.transforms.ToTensor()(img2)
        img2 = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(img2)

        return img1, img2

    def __len__(self):
        return len(self.img_base)

def Text_model():
    model1 = torchvision.models.resnext50_32x4d(pretrained=True)
    in_feature = model1.fc.in_features
    model1.fc = nn.Linear(in_feature, len(index_to_char_New)-1)
    model1.load_state_dict(torch.load('../classification/private_model/model_epoch84_acc90.41.pth'))
    model1.cuda().eval()

    model2 = torchvision.models.resnext101_32x8d(pretrained=True)
    in_feature = model2.fc.in_features
    model2.fc = nn.Linear(in_feature, len(index_to_char_New)-1)
    model2.load_state_dict(torch.load('../classification/private_model/model_epoch92_acc90.92.pth'))
    model2.cuda().eval()

    modelArc2 = torch.load('../classification/private_model/Text_91.9476_52.pkl').cuda().eval()
    modelArc2_2 = torch.load('../classification/private_model/Text_91.8241_68.pkl').cuda().eval()

    ArcModel_Noisy = torch.load('../classification/private_model/Text_98.4700_139.pkl').cuda().eval()
    classifier_Noisy = torch.load('../classification/private_model/classifier9847.pkl').cuda().eval()

    # modelArcB5_S6 = EfficientNetAll(mode = "efficientnet-b5", \
    #                             advprop = False, \
    #                             num_classes = len(index_to_char_New), \
    #                             feature = 2048, \
    #                             ArcFeature = 1024
    #                             )
    # checkpoint = torch.load('../classification/private_model/Text_93.6635_30.pkl', map_location=lambda storage, loc: storage.cuda(0))
    # state_dict = {k.replace("module.", ""): v for k, v in checkpoint.state_dict().items()}
    # modelArcB5_S6.load_state_dict(state_dict)
    # modelArcB5_S6.cuda().eval()

    # modelArcB5_S5 = EfficientNetAll(mode = "efficientnet-b5", \
    #                             advprop = False, \
    #                             num_classes = len(index_to_char_New), \
    #                             feature = 2048, \
    #                             ArcFeature = 1024
    #                             )
    # checkpoint = torch.load('../classification/private_model/Text_92.6507_51.pkl', map_location=lambda storage, loc: storage.cuda(0))
    # state_dict = {k.replace("module.", ""): v for k, v in checkpoint.state_dict().items()}
    # modelArcB5_S5.load_state_dict(state_dict)
    # modelArcB5_S5.cuda().eval()

    # modelArcB5_alun = EfficientNetAll(mode = "efficientnet-b5", \
    #                             advprop = False, \
    #                             num_classes = len(index_to_char_New), \
    #                             feature = 2048, \
    #                             ArcFeature = 1024
    #                             )
    # checkpoint = torch.load('../classification/private_model/Text_93.3547_7.pkl', map_location=lambda storage, loc: storage.cuda(0))
    # state_dict = {k.replace("module.", ""): v for k, v in checkpoint.state_dict().items()}
    # modelArcB5_alun.load_state_dict(state_dict)
    # modelArcB5_alun.cuda().eval()

    return model1, model2, modelArc2, modelArc2_2, ArcModel_Noisy, classifier_Noisy #, modelArcB5_S6, modelArcB5_S5, modelArcB5_alun

def detect():
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    os.makedirs("./out", exist_ok=True)

    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # object detection (yolov5)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # classification
    model1, model2, modelArc2, modelArc2_2, ArcModel_Noisy, classifier_Noisy = Text_model()
    softmax = nn.Softmax(dim=1)

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    for path, img, im0s, vid_cap in dataset:
        draw_img = cv2.imread(path)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        
        t1 = time_synchronized()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = Path(path), '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            stringText = ""
            stringText2 = ""
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                labels = []
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    xyxy += [names[int(cls)]]
                    labels += [[int(i.cpu().data.numpy()) if type(i) != str else i for i in xyxy ]]

                    # plot rectangle
                    color = colors[int(cls)] or [random.randint(0, 255) for _ in range(3)]
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(draw_img, c1, c2, color, thickness=3, lineType=cv2.LINE_AA)

                if len(labels) != 1:
                    if np.std(np.array(labels)[:,0].astype('int')) < np.std(np.array(labels)[:,1].astype('int')):
                        labels = sorted(labels, key=lambda x:x[1])
                    else:
                        labels = sorted(labels, key=lambda x:x[0])

                for idx, xyxy in enumerate( labels ):
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    img = Image.fromarray(np.uint8(img))
                    img1 = torchvision.transforms.Resize((224,224))(img)
                    img1 = torchvision.transforms.ToTensor()(img1)
                    img1 = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(img1)
                    img1 = img1.view(-1, img1.size(0), img1.size(1), img1.size(2))
                    img1 = img1.float().to(device)

                    # img2 = torchvision.transforms.Resize((456,456))(img)
                    # img2 = torchvision.transforms.ToTensor()(img2)
                    # img2 = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(img2)
                    # img2 = img2.view(-1, img2.size(0), img2.size(1), img2.size(2))
                    # img2 = img2.float().to(device)
                    
                    logps = softmax(model1(img1))
                    logps += softmax(model2(img1))
                    logps += softmax(modelArc2(img1)[0])
                    logps += softmax(modelArc2_2(img1)[0])
                    logps = torch.cat((logps, torch.zeros(logps.size(0), 1).cuda()), dim=1)
                    # logps = softmax(modelArcB5_S6(img2)[0])
                    # logps += softmax(modelArcB5_S5(img2)[0])
                    # logps += softmax(modelArcB5_alun(img2)[0])

                    _, pred = torch.max(logps, 1)
                    predicted = index_to_char_New[str(int(pred.cpu().data.numpy()[0]))]

                    test_X = ArcModel_Noisy(img1)  
                    logps = classifier_Noisy(test_X)
                    _, pred2 = torch.max(logps, 1)
                    predicted2 = index_to_char[str(int(pred2.cpu().data.numpy()[0]))]

                    stringText += predicted
                    stringText2 += predicted2

                del(labels)

                t2 = time_synchronized()
                if len(stringText2) == stringText2.count("#"):
                    print('%sDone. (%.3fs) %s' % (s, t2 - t1, "###"))
                    cv2.imwrite("./out/" + path.replace('\\','/').split("/")[-1][:-4] + "_###.png", draw_img)
                else:
                    print('%sDone. (%.3fs) %s' % (s, t2 - t1, stringText))
                    cv2.imencode(".png", draw_img)[1].tofile(u"./out/" + path.replace('\\','/').split("/")[-1][:-4] + "_%s.png"%(stringText))
            else:
                t2 = time_synchronized()
                print('%sDone. (%.3fs) %s' % (s, t2 - t1, "###"))
                cv2.imwrite("./out/" + path.replace('\\','/').split("/")[-1][:-4] + "_###.png", draw_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/expm/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./example', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default = 480, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()