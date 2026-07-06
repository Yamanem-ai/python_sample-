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

#09 clap用のスクリプト (ailia-models-master/audio_processing/clap/の下に置く):

    clap_ex.py (外部からのアクセス用のAPIを挿入すみ)

    key_open.py (メインのプログラム)

#10 padim用のスクリプト (ailia-models-master/anomaly_detection/padim/の下に置く):

    padim_api.py (外部からのアクセス用のAPIを挿入済み)

    post_meeting_check.py (メインのプログラム)

#11 sensevoice&gemma3用のスクリプト (ailia-models-master/audio_processing/sensevoice/の下に置く):

    example_ailia_llm.py (gemma3を単体で動かすデモアプリ)

    gemma_api.py (外部からのアクセス用のAPIを挿入済み)

    sensevoice_api.py (外部からのアクセス用のAPIを挿入済み)

    voice_assistant.py (メインのプログラム)

#12 voice_assistant_bertvits2用のスクリプト&ファイル (ailia-models-master/audio_processing/voice_assistant_bertvits2を作り、配下に置く)

    まずは、bert-vits2フォルダー内で、python3 bert-vits2.pyを起動してください。

    起動後、onnxファイルや辞書フォルダ (unidic-lite)がダウンロードされるので、このフォルダーをコピペあるいは名前を変更してvoice_assistant_bertbits2としてください

    tokenizerフォルダー (すでに配置されていたら、無視してください)
    
    symbols.py (すでに配置されていたら、無視してください)

    vits2utils.py (すでに配置されていたら、無視してください)
   
    bert_vits2_api.py (外部からのアクセス用のAPIを挿入済み)

    gemma_api.py (外部からのアクセス用のAPIを挿入済み)

    sensevoice_api.py (外部からのアクセス用のAPIを挿入済み)

    voice_assistant_plus.py (メインのプログラム)

    voice_assistant_gui.py (メインのプログラム)
