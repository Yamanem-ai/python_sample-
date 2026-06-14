#ailia MODELsのダウンロードサイト (ailia SDKのインストール方法も説明あり)
　https://github.com/ailia-ai/ailia-models/blob/master/TUTORIAL_jp.md

#推論用サンプル動画を入手してください

イノシシ動画を"animal01.mp4"として使っています。著作権フリーですが、２次利用不可のため、
同様の動画を著作権フリーのサイトから入手し、同じ名前に変更した上でフォルダー内に保存してください。

#クロスライン検知スクリプトの起動コマンド

　  python3 crossline_trigger.py (yoloxのみ)

　  python3 crossline_trigger_**.py (yolox以外は名前を変えています)

#01 yolox用のスクリプト (ailia-models-master/object_detection/yolox/の下に置く);

   yolox_ex.py (外部からアクセス用のAPIを挿入済み)
 
   crossline_trigger.py (これがメインのプログラムになる)

#02 dab-detr用のスクリプト (ailia-models-master/object_detection/dab-detr/の下に置く);

    dab_detr_ex.py (外部からアクセス用のAPIを挿入済み)
  
    crossline_trigger_detr.py (これがメインのプログラムになる)

#03 dense_prediction_transformer用のスクリプト(ailia-models-master/image_segmentation/dense_prediction_transformers/の下におく);

    dense_prediction_transformers_ex.py (外部からアクセス用のAPIを挿入済み)

    crossline_trigger_dpt.py (メインのプログラム)

#04 segment_anything_2用のスクリプト(ailia-models-master/image_segmentation/segment-anything-2/の下におく);
  
    segment_anything_2_ex.py (外部からアクセス用のAPIを挿入済み)
  
    gui_tracker.py (メインのプログラム)

#05 strong_sort用のスクリプト (ailia-models-master/object_tracking/strong_sort/の下に置く);

    strongsort_ex_A.py (カメラA用のローカルID追跡用)

    strongsort_ex_B.py (カメラB用のローカルID追跡用)

    global_tracker.py (グローバルID追跡用)

    multi_camera.py (メインのプログラム)

#06 bytetrack用のスクリプト (ailia-models-master/object_tracking/bytetrack/の下に置く);

    bytetrack_ex.py (外部からのアクセス用のAPIを挿入済み)

    people_counter.py (メインのプログラム)

#07 facemesh_v2用のスクリプト (ailia-models-master/face_recognition/facemesh_v2/の下に置く);

    facemesh_v2_ex.py (外部からのアクセス用のAPIを挿入済み)

    dms_monitor.py (メインのプログラム: 3D顔面POSE無し)

    dms_monitor_plus.py (メインのプログラム: 3D顔面POSE有り)

#08 clip用のスクリプト　(ailia-models-master/image_classification/clip/の下におく);

    clip_ex.py (外部からのアクセス用のAPIを挿入済み)

    monitor.py (メインのプログラム)

    yolox_ex.py (呼び出されるので、フォルダー内に置いておく yoloxのonnxファイルは起動時にダウンロードされます)

    yolox_utils.py (呼び出されるので、フォルダー内に置いておく)

   headphone2.mov (サンプルとして置いておきます)
   
