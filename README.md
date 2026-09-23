# ComfyUI-RogoAI-IDeval — CSIM / 本人一致度の評価

ComfyUI custom nodes and a workflow for evaluating face identity similarity using cosine similarity (CSIM), identity consistency across video frames, and facial skin-color changes.

Features include reference building from multiple photos, threshold calibration, image and video evaluation, visual reports, and JSON output. The default face-embedding backend is InsightFace `buffalo_l`.

Similarity scores are not probabilities of identity. Pretrained models have separate licensing terms; see the model usage notes below. Setup and usage instructions are currently provided in Japanese.

「その顔・何点？」シリーズの付録で紹介した、顔の本人一致度と動画内の本人・肌色の安定性を調べるComfyUIワークフローです。

独自コードはMITライセンスです。リポジトリの [LICENSE](LICENSE) を参照してください。使用する第三者ライブラリ・学習済みモデルには、それぞれの利用条件が適用されます。

## できること

- 基準にする人物の複数写真から、顔の特徴の基準を作成。
- 同一人物と別人のスコアを調べ、本人範囲・判定保留・別人範囲の物差しを作成。
- フォルダー内の画像を評価し、結果画像とJSONを出力。
- 動画全体の一致度を時系列で評価し、指定区間の本人保持率も集計。
- 本人評価と組み合わせて、顔の肌色の変化をCIEDE2000色差で評価。

点数は本人である確率ではありません。顔検出失敗・顔の向き・遮蔽も併せて確認してください。現在の顔特徴抽出では、複数の顔が検出されたときは最大の顔を選びます。**集合写真の全員検索や追跡は、このワークフローだけでは実装していません。** 写真整理・通知・好みの学習・画像生成アプリは、別の機能を組み合わせる応用案です。

## 配布ファイル

- `workflows/rogoai_ideval_v3_reference.json`：ComfyUIに読み込むUI形式のワークフロー。API投稿用JSONではありません。
- `__init__.py` / `nodes.py` / `nodes_v2.py`：必要な独自ノード。v3の機能も `nodes_v2.py` に含まれます。
- `docs/WORKFLOW_JA.md`：入力、各モード、結果の読み方。
- `docs/DEPENDENCIES.md`：導入環境、モデルと第三者ライセンスの確認先。

写真・動画・評価結果・顔の特徴ベクトル・学習済みモデルは同梱しません。サンプルの `D:\CSIM_demo` は説明用パスであり、各自の環境に変更してください。

## 必要なノード

| ノード群 | 入手先 |
|---|---|
| RogoAI ID Reference / Ruler / Image Folder / Eval Image / Eval Video / Color Eval Video | このリポジトリ |
| VHS_LoadVideo | [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |
| easy showAnything | [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) |
| SaveImage / MarkdownNote | ComfyUI本体・フロントエンド |

## インストール

1. このリポジトリのフォルダーを、ComfyUIの `custom_nodes/ComfyUI-RogoAI-IDeval` として配置します。以前の `ComfyUI_RogoAI_IDeval` と重複して読み込まないでください。
2. VideoHelperSuiteとComfyUI-Easy-Useを導入します。
3. **ComfyUIが実際に使っているPython** で依存ライブラリを導入します。下の `python` はそのPythonの実行ファイルに置き換えてください。作業ディレクトリはこのノードのフォルダーです。

```powershell
python -m pip install -r requirements.txt
python -m pip install --no-deps facenet-pytorch==2.6.0
```

`facenet-pytorch` はInsightFaceを選んだ場合も、画面確認用の平均顔を作るMTCNNに必要です。`--no-deps` は既存ComfyUIのPyTorch等を古い版へ変更しないためです。PyTorch、torchvision、NumPy、Pillow等が必要なので、[動作環境](docs/DEPENDENCIES.md)も参照してください。`pip check` に古い依存バージョン制約の警告が出る場合があります。どのComfyUI環境でも互換性が保証されるわけではありません。

ONNX Runtimeは環境に合わせて **CPU版・GPU版の一方だけ** を使います。動作中の環境に両方を追加しないでください。

```powershell
# CPUで使う場合の例
python -m pip install onnxruntime
```

制作環境のGPU版は `onnxruntime-gpu==1.20.1` です。GPU利用時はCUDA/cuDNNとの互換性を確認してください。`auto` は状況に応じてCPUを使う場合があります。

4. ComfyUIを再起動し、ワークフローJSONを画面へドラッグして読み込みます。

## 最初の実行

1. 入力・出力フォルダーを準備し、ノードのパスを書き換えます。
2. Referenceには、本人だけの顔が確認できる写真を複数用意します。leave-one-outには検出成功が最低2枚必要です。制作例は23枚で、正面・斜め・表情の違いを含みます。23枚は必須枚数ではありません。
3. 別人画像を `different_people_dir` に設定します。画像評価には別途、調べたい画像を指定します。
4. 動画評価ではLoad Videoへ自分の動画を指定します。`select_your_video.mp4` は未同梱の目印なので、そのまま実行できません。
5. 読み込むフレーム列のFPSとID Eval Videoの `frames_per_second` を一致させます。設定例は24fpsですが、すべての動画が24fpsではありません。
6. 初期表示はすべての系統が有効です。画像だけ／動画だけを実行するときは、使わない評価系統とその保存・表示ノードをまとめて無効化します。ReferenceとRulerは共通で使います。

最初は少数の画像や短い動画で確認してください。画像のSaliency `all` と動画全フレーム処理は計算量が大きくなります。

## 初期設定と保存先

| 項目 | 配布ワークフローの設定 |
|---|---|
| backend | `insightface_buffalo_l` |
| 同一人物の校正 | `leave_one_out_reference` |
| 判定方式 | `distribution` |
| 区間集計 | 20〜30秒 |
| 動画の代表画像 | 5秒間隔、最大12サンプル |
| 色評価 | `sync_id_eval` / `first_stable_frames` / 同一人物条件あり |

20〜30秒は動画を切り取る設定ではなく、全体評価の中で別途集計する区間です。0秒からの代表画像が出ることは正常です。30秒より短い動画では区間設定も見直してください。

画像はComfyUIの `output/<日付>/ID_Eval/`、JSONは各評価ノードの `json_output_path` に保存します。配布例のJSON保存先は `D:\CSIM_demo\results` です。画像とJSONの保存設定は別々です。

## モデル・データの扱い

初回実行時、必要なモデルがない場合には依存ライブラリがダウンロードすることがあります。モデルファイル自体は配布していません。

InsightFaceのコードと学習済みモデルでは利用条件が異なります。公式は提供モデルを非商用研究目的と説明しており、`buffalo_l` の利用許諾窓口も案内しています。[公式のLicense節](https://github.com/deepinsight/insightface#license)を確認してください。独自コードのライセンスを決めても、モデルの条件は変わりません。

ReferenceのJSONには顔の特徴ベクトルと元ファイルのパスが含まれます。実行済みワークフローにも結果表示が残る場合があります。自分の素材で実行した後のJSONやワークフローを、そのまま公開しないでください。

## 確認範囲

既存制作環境のコードと採用ワークフローから公開用コピーを準備しています。配布コピーでは個人環境依存の予備パスだけを修正し、採点ロジックは変更していません。公開準備時の構文・結線・ノード定義の検証と、新規PCでの導入・実画像の推論検証は別です。新規環境へのインストールと推論の再実行は未検証です。
