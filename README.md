#ailia MODELsのダウンロードサイト (ailia SDKのインストール方法も説明あり)
　https://github.com/ailia-ai/ailia-models/blob/master/TUTORIAL_jp.md

#推論用を入手してください

イノシシ動画を"animal01.mp4"として使っています。
同様の動画を著作権フリーのサイトから入手し、同じ名前に変更した上でフォルダー内に保存してください。

#クロスライン検知スクリプトの起動コマンド

　python3 crossline_trigger.py (yoloxのみ)

　python3 crossline_trigger_**.py (yolox以外は名前を変えています)

#yolox用のスクリプト (ailia-models-master/object_detection/yolox/の下に置く);

 yolox_ex.py (外部からアクセス用のAPIを挿入済み)
 
 crossline_trigger.py (これがメインのプログラムになる)

#dab-detr用のスクリプト (ailia-models-master/object_detection/dab-detr/の下に置く);

  dab_detr_ex.py (外部からアクセス用のAPIを挿入済み)
  
  crossline_trigger_detr.py (これがメインのプログラムになる)


