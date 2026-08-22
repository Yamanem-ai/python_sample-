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

    前回使用したsensevoiceフォルダーに入っている以下のフォルダーを持ってくる。"funasr_ailia" "s2t_config" "tokenizer" "vad_config" 

    まずは、bert-vits2フォルダー内で、python3 bert-vits2.pyを起動してください。

    起動後、onnxファイルや辞書フォルダ (unidic-lite)がダウンロードされるので、これらをvoice_assistant_bertbits2に移動してください。

    tokenizerフォルダー にbert-vits2.pyのtokenizerフォルダーの中身を追加してください　(clap-htsat-fused, deberta-v2-large-japanese-char-wwm)
    
    symbols.py (すでに配置されていたら、無視してください)

    vits2utils.py (すでに配置されていたら、無視してください)
   
    bert_vits2_api.py (外部からのアクセス用のAPIを挿入済み)

    gemma_api.py (外部からのアクセス用のAPIを挿入済み)

    sensevoice_api.py (外部からのアクセス用のAPIを挿入済み)

    voice_assistant_plus.py (メインのプログラム)

    voice_assistant_gui.py (メインのプログラム)

#13 voice_assistant_gptsovits用のスクリプト&ファイル (ailia-models-master/audio_processing/voice_assistant_gptsovitsを作り、配下に置く)

    前々回使用したsensevoiceフォルダーに入っている以下のフォルダーを持ってくる。"funasr_ailia" "s2t_config" "tokenizer" "vad_config" 
   
    example_ailia_voice.py　(gpt_sovitsのailia_voiceバージョン)

    gemma_api.py (外部からのアクセス用のAPIを挿入済み)

    sensevoice_api.py (外部からのアクセス用のAPIを挿入済み)

    gpt_sovits_ailia_api.py (外部から呼び出しようにAPI化)

    voice_assistant_plus.py (メインのプログラム)

    voice_assistant_gui.py (メインのプログラム)

#14 voice_assistant_ailia用のスクリプト　(ailia-models-master/audio_processing/voice_assistant_ailiaを作り、配下に置く)

    準備しておくフォルダーは、前回と同様　"funasr_ailia" "s2t_config" "tokenizer" "vad_config" 

    gemma_api.py (外部からのアクセス用のAPIを挿入済み)

    sensevoice_api.py (外部からのアクセス用のAPIを挿入済み)

    gpt_sovits_ailia_api.py (外部から呼び出しようにAPI化)

    voice_assistant_gui.py (メインのプログラム)

#15 lightglue用のスクリプト (ailia-models-master/3d_pointcloud/lightglue/の下に置く)

    camera_path.py (カメラの軌跡確認用スクリプト)

    desk2.MOV (YouTubeで使用していた動画)

    extract_frames.py (サンプル動画から画像を10枚切り出すスクリプト)

    lightglue_api.py (外部から呼び出し用にAPI化)

    reconstruct_pair.py (画像1ペアから3D生成を確認するスクリプト)

    merge_pointcloud.py (メインプログラム)

    view_ply.py (.plyファイルを描画させるスクリプト)

#16 vggt用のスクリプト　(vggt-mainの下に置く)

    このサンプルはailia-modelsに含まれていません。ご自身でGithubからダウンロードしてください。
    
    Githubサイト：https://github.com/facebookresearch/vggt

    以下、動画で使用したサンプル；

    framesフォルダー (サンプルの画像を３枚)

    demo_viser.py (オリジナルスクリプトのcuda周りを修正したもの)

    demo_open3d.py (結果表示をviserではなく、open3dで直接描画するように修正したスクリプト)

#17 BEVFormerのスクリプト (ailia-models-master/automonous_driving/bevformer/の下に置く)

    このファイルの他、nuscenesのデータをダウンロードし、動画で説明しているように必要なフォルダーを配置してください。

    bevformer_nuscenes.py (Nuscenes dataのローダーおよび結果を動画で表示させるスクリプト)

#18 BEVFormer_pytorchのスクリプト (mmdetection3d/BEVFormer/tools/analysis_tools/の下に置く)

    demo_bevformer.py (nuscenesデータローダー入り)

    コマンド例１(Longバージョン、カメラ映像compose、imshow版、v2-base);

    python3 tools/analysis_tools/demo_bevformer.py projects/configs/bevformerv2/bevformerv2-r50-t1-base-24ep.py --checkpoint work_dirs/bevformer_v2-R50-t1-base/epoch_24.pth --samples 1000

    コマンド例２ (Longバージョン、カメラ映像compose, imshow版、tiny-fp16)

    python3 tools/analysis_tools/demo_bevformer.py projects/configs/bevformer_fp16/bevformer_tiny_fp16.py --checkpoint work_dirs/bevformer_tiny_fp16/bevformer_tiny_fp16_epoch_24.pth --samples 1000

    注意：動画ではwork_dirsについて説明していませんでした。BEVFormerのGithubにModel Zooがあり、対象のモデルをダウンロードできます。コマンド通りのフォルダーを作成して配置してください。

#19 Whisper_transcribeのスクリプト (ailia-models-master/audio_processing/whisper/の下に置く)

    whisper_api.py (外部から呼び出しようにAPI化)

    realtime_processing.py (メインのGUIプログラム)
    
