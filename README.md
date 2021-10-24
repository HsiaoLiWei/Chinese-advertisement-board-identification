# Chinese-advertisement-board-identification(Pytorch)
- 中文廣告刊板之中文字辨識，搭配yoloV5抓取ROI中的中文單字位置後，辨識中文單字  
- 競賽連結:https://tbrain.trendmicro.com.tw/Competitions/Details/16  

# 1.Demo
- 四點校正

| Input |
|:----------:|
|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img_10065.jpg)|
| transform |
|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/img_10065_transform.jpg)|

- 模型預測結果

| Input | YoloV5 Text detection | Text classification |
|:----------:|:----------:|:----------|
|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10000_3.png)|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10000_3_%E9%9B%BB%E6%A9%9F%E5%86%B7%E6%B0%A3%E6%AA%A2%E9%A9%97.png)|電機冷氣檢驗|
|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10013_3.png)|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10013_3_%E7%A5%A5%E6%BA%96%E9%90%98%E9%8C%B6%E6%99%82%E8%A8%88.png)|祥準鐘錶時計|
|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10028_5.png)|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10028_5_%E8%96%91%E6%AF%8D%E9%B4%A8.png)|薑母鴨|
|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10028_6.png)|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10028_6_%E8%96%91%E6%AF%8D%E9%B4%A8.png)|薑母鴨|
|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10005_6.png)|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10005_6_%23%23%23.png)|###|
|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/example/img_10005_8.png)|![image](https://github.com/HsiaoLiWei/Chinese-advertisement-board-identification/blob/main/yoloV5/out/img_10005_8_%23%23%23.png)|###|

# 2.Inference
- 我的 Propose methmod 是將訓練模型導入Argmargin + Focal loss計算模型的loss，SEResNet101跟EfficientNet在比賽截止前還沒訓練結束，所以上面的數據是把第46個epoch結果放上去而已，說不定效果會更好  

- Final score = 1_N.E.D - (1 - Precision)

- Public dataset 的上傳分數 

| Model type | Loss function | Final score | Precision | Recall | Normalization Edit Distance(N.E.D.)|
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| ResNeXt50  | Cross entropy |      0.69742|       0.9447|       0.8884|       0.7527|
| ResNeXt101 | Cross entropy |      0.71608|       0.9631|       0.9076|       0.7530|
| SEResNet101| Cross entropy |      0.80967|       0.9984|       0.9027|       0.8112|
| SEResNet101| Focal loss(γ=2) |    0.82015|       0.9986|       0.9032|       0.8215|
| SEResNet101| Focal loss(γ=2) + Cross entropy | 0.85237|       0.9740|       0.9807|       0.8784|
| EfficientNet-b5| Focal loss(γ=2) + Cross entropy | 0.82234|       0.9797|       0.9252|      0.8426|

- Public dataset ensemble 的上傳分數  

| Model type | Final score | Precision | Recall | Normalization Edit Distance(N.E.D.) |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ResNeXt50+ResNeXt101|      0.82532|       0.9894|       0.9046|       0.8359|
| ResNeXt50+ResNeXt101  +SEResNet101|      0.86804|       0.9737|       0.9759|       0.8943|
| ResNeXt50+ResNeXt101  +SEResNet101+EfficientNet-b5|      **0.87167**|       **0.9740**|       **0.9807**|       **0.8977**|

- Private dataset ensemble 的上傳分數  

| Model type | Final score | Precision | Recall | Normalization Edit Distance(N.E.D.) |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ResNeXt50+ResNeXt101  +SEResNet101|      0.8682|       0.9718|       0.9782|       0.8964|
| ResNeXt50+ResNeXt101  +EfficientNet-b5|      0.8727|       0.9718|       0.9782|       0.9009|
| ResNeXt50+ResNeXt101  +SEResNet101+EfficientNet-b5|      **0.8741**|       **0.9718**|       **0.9782**|       **0.9023**|

# 3.Computer equipment
- System: Windows10、Ubuntu20.04
- Pytorch version: Pytorch 1.7 and higher
- Python version: Python 3.6
- Testing:  
CPU: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz  
RAM: 16GB  
GPU: NVIDIA GeForce RTX 2060 6GB  

- Training:  
CPU: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz  
RAM: 256GB  
GPU: NVIDIA GeForce RTX 3090 24GB  

# 4.Download Pretrain models
- 將訓練好的yoloV5 m 模型下載後，放到 ./yoloV5/runs/train/expm/weights/  
https://drive.google.com/drive/folders/1cfoWKvoh9zOzg0njvs1WJOOrnhqiZsY5?usp=sharing  

- 將訓練好的classification模型下載後，放到 ./classification/private_model/  
https://drive.google.com/drive/folders/1CBMReE3JznmqY9cujOODxZVkvzaPpjVb?usp=sharing  

- 將官方提供的yoloV5預訓練模型下載放到./yoloV5/  
https://drive.google.com/drive/folders/1Ykd3-PxwKFrqryjAGKNiVP6eIvV5yu9r?usp=sharing

# 5.Testing
- 先將路徑移到yoloV5底下
```bash
$ cd ./yoloV5
```
- 執行 Text_detection.py 檔案前，請先載好pretrain model，模型輸入圖片請放在 ./yoloV5/example/ 底下，example資料夾底下有圖片範例，執行結束後，預測結果會存在 ./yoloV5/out/，檔名後面會有預測結果，如果是沒有單字或判斷不清楚，會給###，如果有文字，就會顯示預測結果
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
# Reference
[1] https://github.com/ultralytics/yolov5  
[2] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py  
[3] https://github.com/lukemelas/EfficientNet-PyTorch  
[4] https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py  
[5] https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/  
[6] https://tw511.com/a/01/30937.html  
[7] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4690-4699).  
[8] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).  
[9] Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1492-1500).  
