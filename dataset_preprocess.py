'''

Modified Date: 2021/10/23

Author: Li-Wei Hsiao

mail: nfsmw308@gmail.com

train: ./dataset/train
public: ./dataset/public
private: ./dataset/private
'''
import json, os, tqdm, random
import numpy as np
import cv2
import pandas as pd
import time

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def Cal_area_2poly(img, original_bbox, prediction_bbox):
    im = np.zeros(img.shape[:2], dtype = "uint8")
    im1 = np.zeros(img.shape[:2], dtype = "uint8")
    original_grasp_mask = cv2.fillPoly(im, original_bbox.reshape((-1,4,2)), 255)
    prediction_grasp_mask = cv2.fillPoly(im1, prediction_bbox.reshape((-1,4,2)), 255)
    masked_and = cv2.bitwise_and(original_grasp_mask, prediction_grasp_mask, mask=im)
    masked_or = cv2.bitwise_or(original_grasp_mask, prediction_grasp_mask)
    or_area = np.sum(np.float32(np.greater(masked_or, 0))) # 有沒有比0大
    and_area = np.sum(np.float32(np.greater(masked_and,0)))
    IOU = and_area/or_area
    return IOU

def train_valid_get_imageClassification():
    trainJsonPath = './dataset/train/json/'
    trainImgPath = './dataset/train/img/'
    trainSampleImg = './dataset/imgClass/'
    Alltext = './dataset/TrainImageAll.txt'
    trainTxt = './dataset/TrainImage.txt'
    validTxt = './dataset/ValidImage.txt'
    if not os.path.exists(trainSampleImg):
        os.makedirs(trainSampleImg)
    json_files = os.listdir(trainJsonPath)
    outTxt = []
    UNIQUE_CHAR = set(',')
    for file in tqdm.tqdm(json_files):
        with open(trainJsonPath + file, encoding="utf-8") as f:
            data = json.load(f)
        for idx, value in enumerate(data['shapes']):
            label = value['label']
            points = np.array(value['points'])
            group_id = value['group_id']
            shape_type = value['shape_type']
            if group_id == 1 or group_id == 255:
                img = cv2.imread(trainImgPath + file.split('.')[0] + '.jpg')
                xmax, xmin = max(points[:,0]), min(points[:,0])
                ymax, ymin = max(points[:,1]), min(points[:,1])
                # if ymin < 0 or xmin < 0 :
                #     print(label) 
                #     out = four_point_transform(img, points)
                #     cv2.imshow('outXY',out)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                xmin = xmin if xmin > 0 else 0
                ymin = ymin if ymin > 0 else 0
                out = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(trainSampleImg + file.split('.')[0] + '_' + str(idx).zfill(2) + '.jpg', out)
                if label == "": label = "###"
                outTxt += ['.' + trainSampleImg + file.split('.')[0] + '_' + str(idx).zfill(2) + '.jpg ' + label]
                if not label in UNIQUE_CHAR:
                    UNIQUE_CHAR.add(label)
                # cv2.imshow('out',out)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    np.savetxt(Alltext, np.array(outTxt), fmt='%s', delimiter=' ', newline='\n', encoding='utf-8')

    dict_ = {}
    for label in outTxt:
        path, label = label.split(" ")
        if dict_.get(label, None) == None:
            dict_[label] = [path + " %s"%(label)]
        else:
            dict_[label] += [path + " %s"%(label)]

    train = []
    valid = []
    for key, value in dict_.items():
        random.shuffle(value)
        if len(value) == 1:
            train += value
        elif len(value) == 2:
            train += [ value[0] ]
            valid += [ value[1] ]
        else:
            train += value[:int(len(value)*0.9)]
            valid += value[int(len(value)*0.9):]

    np.savetxt(trainTxt, np.array(train), fmt='%s', delimiter=' ', newline='\n', encoding='utf-8')
    np.savetxt(validTxt, np.array(valid), fmt='%s', delimiter=' ', newline='\n', encoding='utf-8')

    char_to_index = {x:y for x, y in zip(
        sorted(list(UNIQUE_CHAR)), range(len(UNIQUE_CHAR)+1))}
    index_to_char = {y:x for x, y in zip(
        sorted(list(UNIQUE_CHAR)), [str(_) for _ in range(len(UNIQUE_CHAR)+1)])}

    with open('./classification/configs/index_to_char.json', 'w') as f:
        json.dump(index_to_char, f)
    with open('./classification/configs/char_to_index.json', 'w') as f:
        json.dump(char_to_index, f)

def train_valid_get_imageChar():
    trainJsonPath = './dataset/train/json/'
    trainImgPath = './dataset/train/img/'
    trainSampleImg = './dataset/imgChar/'
    Alltext = './dataset/TrainImageOnlyCharAll.txt'
    trainTxt = './dataset/TrainImageOnlyChar.txt'
    validTxt = './dataset/ValidImageOnlyChar.txt'
    if not os.path.exists(trainSampleImg):
        os.makedirs(trainSampleImg)
    json_files = os.listdir(trainJsonPath)
    outTxt = []
    UNIQUE_CHAR = []
    for file in tqdm.tqdm(json_files):
        with open(trainJsonPath + file, encoding="utf-8") as f:
            data = json.load(f)
        for idx, value in enumerate(data['shapes']):
            label = value['label']
            points = np.array(value['points'])
            group_id = value['group_id']
            shape_type = value['shape_type']
            if group_id == 1 or group_id == 4:
                if len(UNIQUE_CHAR) == 1187:
                    UNIQUE_CHAR.append("N")
                if len(label) == 1:
                    if not label in UNIQUE_CHAR:
                        UNIQUE_CHAR.append(label)
            if group_id == 1:
                img = cv2.imread(trainImgPath + file.split('.')[0] + '.jpg')
                xmax, xmin = max(points[:,0]), min(points[:,0])
                ymax, ymin = max(points[:,1]), min(points[:,1])
                if ymin < 0 or xmin < 0 :
                    print(trainImgPath + file.split('.')[0] + '.jpg', "  label:", label, " Coordinates are less than 0.") 
                #     out = four_point_transform(img, points)
                #     cv2.imshow('outXY',out)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                xmin = xmin if xmin > 0 else 0
                ymin = ymin if ymin > 0 else 0
                out = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(trainSampleImg + file.split('.')[0] + '_' + str(idx).zfill(2) + '.jpg', out)
                if label == "": label = "###"
                outTxt += ['.' + trainSampleImg + file.split('.')[0] + '_' + str(idx).zfill(2) + '.jpg ' + label]
                
                # cv2.imshow('out',out)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    np.savetxt(Alltext, np.array(outTxt), fmt='%s', delimiter=' ', newline='\n', encoding='utf-8')

    dict_ = {}
    for label in outTxt:
        path, label = label.split(" ")
        if dict_.get(label, None) == None:
            dict_[label] = [path + " %s"%(label)]
        else:
            dict_[label] += [path + " %s"%(label)]
    train = []
    valid = []
    for key, value in dict_.items():
        random.shuffle(value)
        if len(value) == 1:
            train += value
        elif len(value) == 2:
            train += [ value[0] ]
            valid += [ value[1] ]
        else:
            train += value[:int(len(value)*0.85)]
            valid += value[int(len(value)*0.85):]
    np.savetxt(trainTxt, np.array(train), fmt='%s', delimiter=' ', newline='\n', encoding='utf-8')
    np.savetxt(validTxt, np.array(valid), fmt='%s', delimiter=' ', newline='\n', encoding='utf-8')


    char_to_index = {x:y for x, y in zip(
        list(UNIQUE_CHAR), range(len(UNIQUE_CHAR)+1))}
    index_to_char = {y:x for x, y in zip(
        list(UNIQUE_CHAR), [str(_) for _ in range(len(UNIQUE_CHAR)+1)])}

    with open('./classification/configs/index_to_char_Only.json', 'w') as f:
        json.dump(index_to_char, f)
    with open('./classification/configs/char_to_index_Only.json', 'w') as f:
        json.dump(char_to_index, f)

    char_to_index.setdefault("###",len(UNIQUE_CHAR))
    index_to_char.setdefault(len(UNIQUE_CHAR),"###")

    with open('./classification/configs/index_to_char_New.json', 'w') as f:
        json.dump(index_to_char, f)
    with open('./classification/configs/char_to_index_New.json', 'w') as f:
        json.dump(char_to_index, f)

def train_valid_detection_get_bbox():
    trainJsonPath = './dataset/train/json/'
    trainImgPath = './dataset/train/img/'
    trainSample = './dataset/datasetYolo/'
    trainTxt = './dataset/TrainBBox.txt'
    validTxt = './dataset/ValidBBox.txt'
    if not os.path.exists(trainSample):
        os.makedirs(trainSample)
    json_files = os.listdir(trainJsonPath)
    random.shuffle(json_files)
    for json_files, savePath, folder in [(json_files[:int(len(json_files)*0.9)], trainTxt, 'train'), (json_files[int(len(json_files)*0.9):], validTxt, 'Val')]:
        imgs = []
        if not os.path.exists(trainSample + folder + '/images/'):
            os.makedirs(trainSample + folder + '/images/')
        if not os.path.exists(trainSample + folder + '/labels/'):
            os.makedirs(trainSample + folder + '/labels/')
        
        for file in tqdm.tqdm(json_files):
            with open(trainJsonPath + file, encoding="utf-8") as f:
                data = json.load(f)
            img = cv2.imread(trainImgPath + file.split('.')[0] + '.jpg')
            imgs.append(trainSample + folder + '/images/' + file.split('.')[0] + '.jpg')
            
            for idx, value in enumerate(data['shapes']):
                label = value['label']
                points = np.array(value['points'])
                group_id = value['group_id']
                shape_type = value['shape_type']
                if group_id == 0 :
                    bboxs = []
                    bboxs_char = []
                    for idx1, value1 in enumerate(data['shapes']):
                        if value1['group_id'] == 1 :
                            IOU = Cal_area_2poly(img, points, np.array(value1['points']))
                            if IOU > 0.001 and value1['label'] in label:
                                bboxs += [np.array(value1['points'])]
                                bboxs_char += [value1['label']]
                    if bboxs == []:
                        print(trainImgPath + file.split('.')[0] + '.jpg', "  label;", label, " No Bounding Box.") 
                        # out = four_point_transform(img, points)
                        # cv2.imshow('out',out)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        continue
                    vbboxs = np.vstack(np.array(bboxs))
                    xmax, xmin = max(vbboxs[:,0]), min(vbboxs[:,0])
                    ymax, ymin = max(vbboxs[:,1]), min(vbboxs[:,1])
                    if ymin < 0 or xmin < 0 :
                        print(trainImgPath + file.split('.')[0] + '.jpg', "  label:", label, " Coordinates are less than 0.") 
                        # out = four_point_transform(img, points)
                        # cv2.imshow('outXY',out)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    xmin = xmin if xmin > 0 else 0
                    ymin = ymin if ymin > 0 else 0

                    size = 0
                    xmin = xmin - size if xmin - size > 0 else 0
                    ymin = ymin - size if ymin - size > 0 else 0
                    xmax = xmax + size if xmax + size < img.shape[1] else img.shape[1]
                    ymax = ymax + size if ymax + size < img.shape[0] else img.shape[0]

                    out = img[ymin:ymax, xmin:xmax]
                    # cv2.imshow('out',out)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    cv2.imwrite( trainSample + folder + '/images/' + file.split('.')[0] + '_' + str(idx).zfill(2) + '.jpg', out)
                    
                    labels = []
                    for contour, char in zip(bboxs, bboxs_char):
                        xMin, xMax = min(contour[:,0]), max(contour[:,0])
                        yMin, yMax = min(contour[:,1]), max(contour[:,1])
                        xMin, xMax = xMin - xmin, xMax - xmin
                        yMin, yMax = yMin - ymin, yMax - ymin
                        w, h = (xMax - xMin)/out.shape[1], (yMax - yMin)/out.shape[0]
                        x, y = xMin/out.shape[1] + w/2, yMin/out.shape[0] + h/2
                        labels += [[1, x,y,w,h]]
                        if x > 1 or y > 1:
                            print("stop")
                    # cv2.drawContours(img, np.array(bboxs), -1, (0,0,255), 3)
                    # cv2.imshow('out',img)
                    # # cv2.imshow('mask',mask)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    labels = np.array(labels)
                    if labels.ndim == 1:
                        labels = labels[np.newaxis, :]
                    np.savetxt(trainSample + folder + '/labels/' + file.split('.')[0] + '_' + str(idx).zfill(2) + '.txt', np.array(labels), fmt='%0.5f', delimiter=' ', newline='\n')

                # points = points.reshape((-1,4,2))
                # cv2.polylines(img, points, isClosed=True, color = (0,0,255), thickness=5)
                # cv2.imshow('img',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        np.savetxt(savePath, np.array(imgs), fmt='%s', delimiter=' ', newline='\n')

def test_bbox():
    for path in os.listdir('./dataset/datasetYolo/train/images/'):
        img = cv2.imread('./dataset/datasetYolo/train/images/' + path)
        labels = np.loadtxt('./dataset/datasetYolo/train/labels/' + path[:-4] + '.txt')
        red_color = (0, 0, 255) # BGR
        if len(labels.shape) == 1: labels = labels[np.newaxis, :]
        for label, x,y,w,h in labels:
            x0 = int((x - w/2)*img.shape[1])
            x1 = int((x + w/2)*img.shape[1])
            y0 = int((y - h/2)*img.shape[0])
            y1 = int((y + h/2)*img.shape[0])
            cv2.rectangle(img, (x0, y0), (x1, y1), red_color, 3, cv2.LINE_AA)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def private_img_get_preprocess():
    publicPath = './dataset/privateImg/'
    if not os.path.exists(publicPath):
        os.makedirs(publicPath)
    eval_save_txt = []
    csv_table = pd.read_csv('./dataset/private/Task2_Private_String_Coordinate.csv', header=None).values
    count = 0
    oriPath = ""
    for idx,(path,x00,x01,x10,x11,x20,x21,x30,x31)  in enumerate(tqdm.tqdm(csv_table)):
        img = cv2.imread('./dataset/private/img_private/' + path + '.jpg')
        img = four_point_transform(img, np.array([[x00,x01],[x10,x11],[x20,x21],[x30,x31]]))
        if oriPath == path or idx == 0:
            count +=1
        else:
            count = 0
        cv2.imwrite(publicPath + path + '_' + str(count) + '.png', img)
        eval_save_txt.append(['.%s/%s_%d.png'%(publicPath, path, count)])
        oriPath = path
    np.savetxt('./dataset/private.txt', eval_save_txt, fmt='%s')

def image_transform(path, points):
    img = cv2.imread(path)
    out = four_point_transform(img, points)
    cv2.imwrite(path[:-4] + '_transform.jpg', out)

if __name__ in "__main__":
    train_valid_get_imageClassification()   # 生成的資料庫辨識是否是文字的 function
    train_valid_get_imageChar()             # 生成的資料庫辨識該圖像是哪個文字的 function
    train_valid_detection_get_bbox()         # 生成的資料庫判斷文字位置的 function
    private_img_get_preprocess()            # 生成預處理的資料庫，之後利用 yolo 抓出char位置，最後放入模型辨識
    # test_bbox()                             # 查看BBOX有沒有抓對
    # image_transform('./img_10065.jpg', np.array([ [169,593],[1128,207],[1166,411],[142,723] ])) # 將輸入圖片與要截取的四邊座標轉成正面
