# Chinese-advertisement-board-identification
中文廣告刊板之中文字辨識，搭配yoloV5抓取ROI中的中文單字位置後，辨識中文單字

# Computer equipment
Testing:
CPU: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
RAM: 16GB
GPU: NVIDIA GeForce RTX 2060 6GB

Training:
CPU: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
RAM: 128GB
GPU: NVIDIA GeForce RTX 3090 24GB

# Pretrain models
將訓練好的yoloV5 m 模型下載後，放到 ./yoloV5/runs/train/expm/weights/
https://drive.google.com/drive/folders/1cfoWKvoh9zOzg0njvs1WJOOrnhqiZsY5?usp=sharing

將訓練好的classification模型下載後，放到 ./classification/private_model/
https://drive.google.com/drive/folders/1CBMReE3JznmqY9cujOODxZVkvzaPpjVb?usp=sharing

# Testing
先將路徑移到yoloV5底下

$ cd ./yoloV5

執行 Text_detection.py 檔案，模型輸入圖片請放在 ./yoloV5/example/ 底下，example資料夾底下有圖片範例，執行結束後，預測結果會存在 ./yoloV5/out/，檔名後面會有預測結果，如果是沒有單字或判斷不清楚，會給###，如果有文字，就會顯示預測結果

$ python3 Text_detection.py


