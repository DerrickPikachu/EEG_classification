# EEG_classification
LAB2 in NYCU DLP. Use CNN to classify the EEG data.


## Train a network
- 直接利用python3 main.py執行main.py
- 選擇好想要的架構以後，會自動開始進行訓練
- 每個epoch都會output testing accuracy

## Show the comparison figure
- 直接利用python3 train.py執行train.py
- 選擇想要訓練的network strcut，接著會開始對三種不同的activation function進行訓練
- 訓練完成後，會產生一張圖檔，圖檔存於專案資料夾中(如果選擇訓練EEG則會叫做EEGNet.png，而DeepConvNet則會叫做DeepConvNet.png)

## Test the model
- 以已經儲存好的model test。
- 有六種不同的model可以使用，分別是:
  1. EEG_relu
  2. EEG_leakyRelu
  3. EEG_elu
  4. DCN_relu
  5. DCN_leakyRelu
  6. DCN_elu
- 以python3 testModel.py執行
- 必須先將測的model架構建立好，根據要練的對象做出對應的選擇
- 再輸入要test的model，如EEG_relu
- 輸入完畢以後，會開始進行測試

## Environment
- OS: Ubuntu 18.04
- python: 3.6.9
