# Chinese-advertisement-board-identification(Pytorch)
中文廣告刊板之中文字辨識，搭配yoloV5抓取ROI中的中文單字位置後，辨識中文單字  
競賽連結:https://tbrain.trendmicro.com.tw/Competitions/Details/16  

# Computer equipment
System environment: Windows10、Ubuntu20.04

Testing:  
CPU: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz  
RAM: 16GB  
GPU: NVIDIA GeForce RTX 2060 6GB  

Training:  
CPU: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz  
RAM: 256GB  
GPU: NVIDIA GeForce RTX 3090 24GB  

# Download Pretrain models
將訓練好的yoloV5 m 模型下載後，放到 ./yoloV5/runs/train/expm/weights/  
https://drive.google.com/drive/folders/1cfoWKvoh9zOzg0njvs1WJOOrnhqiZsY5?usp=sharing  

將訓練好的classification模型下載後，放到 ./classification/private_model/  
https://drive.google.com/drive/folders/1CBMReE3JznmqY9cujOODxZVkvzaPpjVb?usp=sharing  

將官方提供的yoloV5預訓練模型下載放到./yoloV5/  
https://drive.google.com/drive/folders/1Ykd3-PxwKFrqryjAGKNiVP6eIvV5yu9r?usp=sharing

# Testing
先將路徑移到yoloV5底下
```bash
$ cd ./yoloV5
```
執行 Text_detection.py 檔案前，請先載好pretrain model，模型輸入圖片請放在 ./yoloV5/example/ 底下，example資料夾底下有圖片範例，執行結束後，預測結果會存在 ./yoloV5/out/，檔名後面會有預測結果，如果是沒有單字或判斷不清楚，會給###，如果有文字，就會顯示預測結果
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
