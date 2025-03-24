# 中平研のミーバイ検出AIカメラのコード類です

## ディレクトリ構成
```
.
├── code
|   ├── fish_deteciton.py・・・・・・・・・bagファイルの検出を行うコード
|   └── fish_analysis.py・・・・・・・検出したデータを解析するコード
├── model
|   ├── best.pt ・・・・・・・・・制度は悪いが数多く検出してくれるモデル
|   └── model_fish.pt・・・・・・・制度は高いが少ない数を検出してくれるモデル
└── README.mb・・・・・・・このファイル
```
## bagファイルについて
bagファイルは中平研にあるssdに保存しています。  
またbagファイルは以下のサイトからダウンロード可能なIntel RealSense Viewerにより再生できます。  
https://www.intelrealsense.com/sdk-2/
