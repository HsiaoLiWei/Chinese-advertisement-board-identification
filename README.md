# Chinese-Advertisement-Board-Identification(Pytorch)  
- Competition URL : https://tbrain.trendmicro.com.tw/Competitions/Details/16  (Private 5th place)

# 1.Propose method
## The model
- We first calibrate the direction of the image according to the given coordinates by points transformation algorithm to magnify the font of the characters, which improves the prediction result of the model. Next, we apply pre-trained Yolov5 to predict the box location of the characters, and use sort box location algorithm to sort the order of those located characters. With this, we can not only obviate the problem of string disorder, but also filter out images that contains no characters using Yolov5. Then, we perform two types of classification for each located character box. The first type of classification is to determine whether it is a character. If it is not, we directly label it as "###"; and if it is a character, we perform the second classifiation to recognize the character in the located box.
<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/modelall.png" width=40% height=40%>

- This is our proposed training method for CNN that improves the precision on character recognition by incorporating ArcMargin, FCN, and Focal loss. By using these two types of loss to determine the backend, the classification model can further distinguish the difference between features (The choice of CNN model can be optional to any classification architecture).
<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/modelclass.png" width=40% height=40%>

## Data augmentation
- Random Mosaic

| Input image | Mosaic size = 2 | Mosaic size = 4 | Mosaic size = 6 | Mosaic size = 8 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/%E5%8E%9F%E5%9C%96.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/Mosaic2.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/Mosaic4.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/Mosaic6.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/Mosaic8.jpg" width=50% height=50%>|

- Random scale Resize

| Input image | 56x56 to 224x224 | 38x38 to 224x224 | 28x28 to 224x224 | 18x18 to 224x224 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/%E5%8E%9F%E5%9C%96.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/56_56.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/38_38.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/28_28.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/18_18.jpg" width=50% height=50%>|

- Random ColorJitter

| Input image | brightness=0.5 | contrast=0.5 | saturation=0.5 | hue=0.5 | brightness=0.5  contrast=0.5  saturation=0.5  hue=0.5 |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/%E5%8E%9F%E5%9C%96.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/brightness.jpg" width=35% height=35%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/contrast.jpg" width=40% height=40%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/saturation.jpg" width=37.5% height=37.5%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/hue.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/ColorJitter.jpg" width=27.5% height=27.5%>|

# 2.Demo
- Four points transformation

| Input image | After transformation |
|:----------:|:----------:|
|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img_10065.jpg" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img_10065_transform.jpg" width=50% height=50%>|

- Predicted results

| Input image | YoloV5 Text detection | Text classification |
|:----------:|:----------:|:----------|
|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10000_3.png)|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10000_3_%E9%9B%BB%E6%A9%9F%E5%86%B7%E6%B0%A3%E6%AA%A2%E9%A9%97.png)|電機冷氣檢驗|
|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10013_3.png" width=20% height=20%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10013_3_%E7%A5%A5%E6%BA%96%E9%90%98%E9%8C%B6%E6%99%82%E8%A8%88.png" width=20% height=20%>|祥準鐘錶時計|
|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10028_5.png" width=50% height=50%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10028_5_%E8%96%91%E6%AF%8D%E9%B4%A8.png" width=50% height=50%>|薑母鴨|
|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10028_6.png" width=20% height=20%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10028_6_%E8%96%91%E6%AF%8D%E9%B4%A8.png" width=20% height=20%>|薑母鴨|
|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10005_6.png" width=30% height=30%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10005_6_%23%23%23.png" width=30% height=30%>|###|
|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10005_8.png" width=30% height=30%>|<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10005_8_%23%23%23.png" width=30% height=30%>|###|

# 3.Competition results
- Our proposed method combined the training model with ArcMargin and Focal loss
- The training of the two models, SEResNet101 and EfficientNet, has not ended before the end of the competition. Therefore, the above results which are the 46th epoch could be more accurately

- Final score = 1_N.E.D - (1 - Precision)
- Arc Focal loss = ArcMargin + Focal loss(γ=2) 、 Class Focal loss = FCN + Focal loss(γ=1.5)
- Public dataset scores 

| Model type | Loss function | Final score | Precision | Recall | Normalization Edit Distance(N.E.D.)|
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| ResNeXt50  | Cross entropy |      0.69742|       0.9447|       0.8884|       0.7527|
| ResNeXt101 | Cross entropy |      0.71608|       0.9631|       0.9076|       0.7530|
| SEResNet101| Cross entropy |      0.80967|       0.9984|       0.9027|       0.8112|
| SEResNet101| Focal loss(γ=2) |    0.82015|       0.9986|       0.9032|       0.8215|
| SEResNet101| Arc Focal loss(γ=2)<br>+ Class Focal loss(γ=1.5) | 0.85237|       0.9740|       0.9807|       0.8784|
| EfficientNet-b5| Arc Focal loss(γ=2)<br>+ Class Focal loss(γ=1.5) | 0.82234|       0.9797|       0.9252|      0.8426|

- Public dataset ensemble scores   

| Model type | Final score | Precision | Recall | Normalization Edit Distance(N.E.D.) |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ResNeXt50+ResNeXt101|      0.82532|       0.9894|       0.9046|       0.8359|
| ResNeXt50+ResNeXt101<br>+SEResNet101|      0.86804|       0.9737|       0.9759|       0.8943|
| ResNeXt50+ResNeXt101<br>+SEResNet101+EfficientNet-b5|      **0.87167**|       **0.9740**|       **0.9807**|       **0.8977**|

- Private dataset ensemble scores   

| Model type | Final score | Precision | Recall | Normalization Edit Distance(N.E.D.) |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ResNeXt50+ResNeXt101<br>+SEResNet101|      0.8682|       0.9718|       0.9782|       0.8964|
| ResNeXt50+ResNeXt101<br>+EfficientNet-b5|      0.8727|       0.9718|       0.9782|       0.9009|
| ResNeXt50+ResNeXt101<br>+SEResNet101+EfficientNet-b5|      **0.8741**|       **0.9718**|       **0.9782**|       **0.9023**|

# 4.Computer equipment
- System: Windows10、Ubuntu20.04
- Pytorch version: Pytorch 1.7 or higher
- Python version: Python 3.6
- Testing:  
CPU: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz  
RAM: 16GB  
GPU: NVIDIA GeForce RTX 2060 6GB  

- Training:  
CPU: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz  
RAM: 256GB  
GPU: NVIDIA GeForce RTX 3090 24GB  

# 5.Download pretrained models
- After downloading the pre-trained model of YoLoV5(m), put the model on the path `./yoloV5/runs/train/expm/weights/`.
https://drive.google.com/drive/folders/1cfoWKvoh9zOzg0njvs1WJOOrnhqiZsY5?usp=sharing  

- After downloading the pre-trained model of classification，put the model on the path `./classification/private_model/`.
https://drive.google.com/drive/folders/1CBMReE3JznmqY9cujOODxZVkvzaPpjVb?usp=sharing  

- Download the model which is provided by the official, then put the model on the path `./yoloV5/`. 
https://drive.google.com/drive/folders/1Ykd3-PxwKFrqryjAGKNiVP6eIvV5yu9r?usp=sharing

# 6.Testing
## Model evaulation -- Get the predicted results by inputting images
- First, move your path to the `yoloV5`
```bash
$ cd ./yoloV5
```
- Please download the pre-trained model before you run "Text_detection.py" file. Then, put your images under the path `./yoloV5/example/`.
- There are some examples under the folder `example`. The predicted results will save on the path `./yoloV5/out/` after you run the code. The predicted results are on the back of filename. If no words or the images are not clear enough, the model will predict "###". Otherwise, it will show the predicted results.
- **Note!!** You need to verify that the input image is the same as the given image under the folder "example". If the image is not a character image, you could provide the four points coordinate of the image, then deploy the function of image transform, which is in the file "dataset_preprocess.py".
- **Note!!** The model of the text classification does not add the model of "EfficientNet-b5". If you would like to use it, you need to revise the code and de-comment by yourself.
```bash
$ python3 Text_detection.py

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.75, device='', img_size=480, iou_thres=0.6, save_conf=False, save_txt=False, source='./example', view_img=False, weights='./runs/train/expm/weights/best.pt')
Fusing layers... 
image 1/12 example\img_10000_2.png: 160x480 6 Texts, Done. (0.867s) 法國康達石油
image 2/12 example\img_10000_3.png: 160x480 6 Texts, Done. (0.786s) 電機冷氣檢驗
image 3/12 example\img_10000_5.png: 96x480 7 Texts, Done. (0.998s) 見達汽車修理廠
image 4/12 example\img_10002_5.png: 64x480 12 Texts, Done. (1.589s) 幼兒民族芭蕾成人有氧韻律
image 5/12 example\img_10005_1.png: 480x96 6 Texts, Done. (0.790s) 中山眼視光學
image 6/12 example\img_10005_3.png: 480x352 Done. (0.000s) ###
image 7/12 example\img_10005_6.png: 480x288 Done. (0.000s) ###
image 8/12 example\img_10005_8.png: 480x288 1 Texts, Done. (0.137s) ###
image 9/12 example\img_10013_3.png: 480x96 6 Texts, Done. (0.808s) 祥準鐘錶時計
image 10/12 example\img_10017_1.png: 480x64 7 Texts, Done. (0.917s) 國立臺灣博物館
image 11/12 example\img_10028_5.png: 160x480 3 Texts, Done. (0.399s) 薑母鴨
image 12/12 example\img_10028_6.png: 480x128 3 Texts, Done. (0.411s) 薑母鴨
```
## Image transform

- Change the main of "dataset_preprocess.py" to execute the function "image_transform()"
```python
def image_transform(path, points):
    img = cv2.imread(path)
    out = four_point_transform(img, points)
    cv2.imwrite(path[:-4] + '_transform.jpg', out)

if __name__ in "__main__":
    # train_valid_get_imageClassification()   # 生成的資料庫辨識是否是文字的 function
    # train_valid_get_imageChar()             # 生成的資料庫辨識該圖像是哪個文字的 function
    # train_valid_detection_get_bbox()         # 生成的資料庫判斷文字位置的 function
    # private_img_get_preprocess()            # 生成預處理的資料庫，之後利用 yolo 抓出char位置，最後放入模型辨識
    # test_bbox()                             # 查看BBOX有沒有抓對
    image_transform('./img_10065.jpg', np.array([ [169,593],[1128,207],[1166,411],[142,723] ])) # 將輸入圖片與要截取的四邊座標轉成正面
```
# 6.Training
- The folder should be put under the fold "./dataset/" first, then unzip the .zip file provided by the official
- The training data preprocessing can be running after you unzip the file.
```bash
$ python3 dataset_preprocess.py
```
## YoloV5 training and evaluation
- Follow the instructions provided by the Yolov5 official to do the pre-processing of the data, and you can train after you finish.
- The data pre-processing of Yolov5 has been written in the function "train_valid_detection_get_bbox()", which is in the file `dataset_preprocess.py`. Therefore, you can get the training data after you run the file `dataset_preprocess.py`.
- After that, move you path to `./yoloV5/`.
```bash
$ cd ./yoloV5
```
- After modifying the hyperparameters under the file `train.py`, you can start training. Please download the [pre-trained models](# 5.Download pretrained models) before training.
```bash
$ python3 train.py
```
- After training, You need to modify the path of the model to evaluate the performance of the model. And tune the parameters of "conf-thres" and "iou-thres" values according to your own model. We evaluate our model using the private dataset. If you want to use another dataset, please modify the path by yourself.
```bash
$ python3 detect.py
```
- Finally, please move path to `classification`. 
```bash
$ cd ../classification
```
- Run the results of the text classification. Please modify the code if you revise any path or filename
```bash
$ python3 Ensemble.py
```
## Text or ### classification Training
- Please move path to `classification`.
```bash
$ cd ./classification
```
- The data pre-processing of classification has beeb written in the function "train_valid_get_imageClassification()", which is in the file `dataset_preprocess.py`. Therefore, you can get the training data after you run the file `dataset_preprocess.py`.
- Model training.
```bash
$ python3 ClassArcTrainer.py
```
- You need to modify the path by yourself to fine-tune the last classifier. use the best model which is in the folder `./modelsArc/` and modify the 111th line of `ClassArcTest.py`. After that, you can run the code.
```bash
$ python3 ClassArcTest.py
```
## Text recognition Training
- Please move to path `classification`
```bash
$ cd ./classification
```
- The data pre-processing of classification has beeb written in the function "train_valid_get_imageChar()", which is in the file `dataset_preprocess.py`. Therefore, you can get the training data after you run the file `dataset_preprocess.py`.
- Train the model we provided.
```bash
$ python3 CharArcTrainer2.py
```
- Train the model of resnext50 or resnext101.
```bash
$ python3 CharTrainer.py
```
- **Please run the code of `detect.py` to extract the word bounding box before evaluation. After that, you should modify the path in `Ensemble.py` to use the model you trained.

# References
[1] https://github.com/ultralytics/yolov5  
[2] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py  
[3] https://github.com/lukemelas/EfficientNet-PyTorch  
[4] https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py  
[5] https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/  
[6] https://tw511.com/a/01/30937.html  
[7] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4690-4699).  
[8] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).  
[9] Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1492-1500).  
