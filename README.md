# Chinese-Advertisement-Board-Identification(Pytorch)  
- Competition URL : https://tbrain.trendmicro.com.tw/Competitions/Details/16  

# 1.Propose method
- 我們針對官方給定的四點座標，先做四點校正(points tranforms)演算法，使用此演算法的原因是為了得到轉正的圖片，讓字體面積更大，也能讓模型更好的預測結果。校正好字串的圖片後，用已訓練好的Yolov5模型預測單字框的位置，框選出單字位置後，使用單字框排序(Sort Box location)演算法，排出所有文字的順序，這樣一來，找出的字串就不會有順序不對的問題，也可以透過Yolov5過濾到許多沒文字的圖片。接下來針對所有的單字框做兩種分類，第一種分類是否是文字，若不是文字，直接標出"###"; 若是文字，則使用單字分類器辨識文字
<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/modelall.png" width=25% height=25%>

- 這是我們提出來用於在CNN上能更準確分辨出文字的訓練方式，透過ArcMargin 、 FCN and Focal loss這兩者loss決定Backend，讓模型更能將特徵的差異性分辨更好，CNN的部份只要是分類架構都可以替換
<img src="https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img/modelclass.png" width=25% height=25%>

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
- After downloading the pre-trained model of YoLoV5(m), put the model on the path "./yoloV5/runs/train/expm/weights/".
https://drive.google.com/drive/folders/1cfoWKvoh9zOzg0njvs1WJOOrnhqiZsY5?usp=sharing  

- After downloading the pre-trained model of classification，put the model on the path "./classification/private_model/".
https://drive.google.com/drive/folders/1CBMReE3JznmqY9cujOODxZVkvzaPpjVb?usp=sharing  

- Download the model which is provided by the official, then put the model on the path "./yoloV5/". 
https://drive.google.com/drive/folders/1Ykd3-PxwKFrqryjAGKNiVP6eIvV5yu9r?usp=sharing

# 6.Testing
## Model evaulation -- Get the predicted results by inputting images
- First, move your path to the "yoloV5"
```bash
$ cd ./yoloV5
```
- Please download the pre-trained model before you run "Text_detection.py" file. Then, put your images under the path "./yoloV5/example/".
- There are some examples under the folder "example". The predicted results will save on the path "./yoloV5/out/" after you run the code. The predicted results are on the back of filename. If no words or the images are not clear enough, the model will predict "###". Otherwise, it will show the predicted results.
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
- Text classification的模型沒有加入EfficientNet-b5，若想要使用的話，需要自行解註解與修改程式

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
