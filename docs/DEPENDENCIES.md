# 依存関係と確認環境

公開準備時に、制作に使った既存環境で確認したパッケージのバージョンです。新規環境の自動構築を保証するロックファイルではありません。

| パッケージ | バージョン |
|---|---|
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| numpy | 2.4.2 |
| Pillow | 12.2.0 |
| matplotlib | 3.11.0 |
| insightface | 1.0.1 |
| onnxruntime-gpu | 1.20.1 |
| scikit-image | 0.26.0 |
| facenet-pytorch | 2.6.0 |
| opencv-python | 5.0.0.93 |
| requests | 2.34.2 |
| tqdm | 4.68.4 |

facenet-pytorch 2.6.0の依存指定はこの環境のPyTorch等より古いため、制作環境では `--no-deps` で導入しています。READMEの導入例は既存ComfyUI向けです。PythonバージョンやGPU構成による互換性は別途確認してください。

画像ギャラリーの日本語フォントはWindowsの游ゴシック・メイリオを優先します。フォントは同梱していません。Linux/macOSでは日本語の表示品質を未確認です。

## 第三者プロジェクト

- [InsightFace](https://github.com/deepinsight/insightface)：顔検出・特徴抽出・姿勢。コードと学習済みモデルの利用条件は異なります。[License節](https://github.com/deepinsight/insightface#license)参照。buffalo_lのモデルは同梱しません。
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)：MTCNNによる平均顔表示。FaceNet backendを選ぶとInceptionResnetV1のVGGFace2学習済み重みも使います。第三者モデル・データの条件は各配布元を確認してください。
- [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)：動画の読み込み。
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)：JSON結果の表示。
- [ONNX Runtime CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)：CUDA/cuDNNとの対応関係。

ここで挙げた第三者コード・モデルを、この配布物にコピーして同梱してはいません。インストール時・初回実行時に別途取得する場合があります。
