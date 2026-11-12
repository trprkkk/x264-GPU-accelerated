# hybrid-x264-gpu

x264 (0.165.x, commit `0480cb05fa188d37ae87e8f4fd8f1aea3711f7ee`) をベースに、
**CUDA による整数精度動き探索 (ME) の候補生成**を追加した CPU/GPU ハイブリッド
H.264 エンコーダです。

- GPU は NVENC のような固定機能ではなく、汎用 CUDA コアで動き探索の候補を計算します。
- **最終的な符号化判断はすべて CPU 側の x264 コードが行います。** GPU が返すのは
  マクロブロックごとの整数精度 MV 候補 (最大 4 個/参照/リスト) のみで、x264 本体が
  本物のコスト関数 (SATD + λ·MV コスト、RDO) で評価し直します。
- `--gpu-me 0` (既定) は GPU を一切使わず、**stock x264 とビット単位で同一**の出力を
  生成します (検証済み)。

---

## 1. 特徴

| 項目 | 内容 |
|---|---|
| 対象 | H.264/AVC (x264 フォーク、High profile / 8-bit ビルドで GPU 有効) |
| GPU 利用 | 汎用 CUDA カーネル (NVENC/QSV 等の固定機能は不使用) |
| 画質 | GPU は候補を「追加」するだけ。CPU が本物のコストで選び直すため、実測でも SSIM 差は ±0.00005 以内 |
| 互換性 | 出力は通常の H.264 ストリーム。MPC-HC / ffmpeg 等でデコード可能 (検証済み) |
| 対応解像度 | 任意 (x264 準拠。MB 境界までパディングして GPU 解析。1080p も GPU 加速) |
| 速度 | `--me esa` / `--me tesa` で **+17〜+47%** (2 測定セッション。下記ベンチマーク) |
| 既定動作 | `--gpu-me 0`。従来と完全に同一・ビット一致 |
| フォールバック | CUDA 無し / 非対応条件では警告を出して自動的に CPU 動作へ |

### モード一覧

| `--gpu-me` | 動作 | 用途 |
|---|---|---|
| `0` | GPU オフ (既定) | 従来と同じ。ビット一致 |
| `1` | 16x16 ME に GPU 候補を追加してから通常探索 | 画質重視。候補が追加されるため画質は同等以上 |
| `2` | 全パーティションに GPU 候補を追加し、CPU 探索範囲を ±4 に縮小 | **推奨**。esa/tesa で +28〜29% 高速化 (画質はほぼ同等) |
| `3` | 全パーティションに GPU 候補を追加し、CPU のフルペル探索を省略 | 最速。画質はほぼ同等 (±0.00005 SSIM 程度の変動) |

---

## 2. 必要環境

- Windows x64
- Visual Studio (C++ ツールセット。動作確認: VS 18 Community / cl 19.51)
- CUDA Toolkit 13.3 (`nvcc` と `cudart_static.lib`) — 動作確認: 13.3.73
- NASM 2.16.03 (`toolchain/nasm-2.16.03` に配置済み。別バージョンでも可)
- Git Bash + `mingw32-make` (MSYS2/MinGW の make でも可)
- NVIDIA GPU (Compute Capability 7.5 でビルド確認: RTX 2060 SUPER)

GPU 対応条件 (これ以外は自動で CPU フォールバック):

- 8-bit エンコード (`--output-depth 8`)
- プログレッシブ (インターレースは非対応)

解像度は x264 が受け付ける任意のサイズに対応します。フレームプレーンは
MB 境界までパディングされて解析されるため、1920x1080 のような 16 の倍数で
ない解像度でも GPU 加速されます。

---

## 3. ビルド手順

```bash
# 1) 環境変数 (CUDA の場所。configure-msvc.sh / make-msvc.sh の既定値と同じ)
export CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"

# 2) configure (build-msvc/ を作成し config.mak / config.h に CUDA 設定を追記)
cd hybrid-x264-gpu
./configure-msvc.sh

# 3) ビルド
./make-msvc.sh          # -j8 で並列ビルド
# 生成物: build-msvc/x264.exe
```

> 補足: `x264/tools/msvsdepend.sh` が MSVC の `/showIncludes` 出力からヘッダ依存
> (`.d`) を生成します。本フォークでは日本語ロケール (CP932) の出力パターンと、
> ビット深度定義 (`-DBIT_DEPTH`) なしでスキャンが中断する問題の両方に対応済みで、
> ヘッダ変更時の差分ビルドが正しく再コンパイルされます。
>
> CUDA カーネルのターゲット GPU は環境変数 `CUDA_ARCH` で指定できます
> (既定 `sm_75` = Turing。Ampere なら `sm_86` など。`configure-msvc.sh` を参照)。

Windows のコマンドプロンプトから一括で行う場合は同梱のバッチを使用します
(内部で `vcvars64.bat` を呼び、Git Bash 経由で make を実行します)。

| スクリプト | 内容 |
|---|---|
| `build-gpu.bat` | configure → clean → フルビルド |
| `rebuild-gpu.bat` | configure 省略で clean → フルビルド |
| `inc-gpu.bat` | 差分ビルドのみ |
| `selftest.ps1` | 受け入れテスト (ビット一致・GPU モード・デコード・フォールバック・決定性) |
| `det_battery.ps1` | 決定性バッテリー (mode 1/2/3 反復実行とフレームタイプ列の検証) |
| `bench720.ps1` | 720p ベンチマーク (medium/slow/esa/tesa × mode 0〜3) |
| `bench-placebo.ps1` | placebo プリセットのベンチマーク (mode 0〜3) |
| `bench-nvenc.ps1` | NVENC 比較 (同一ビットレート・SSIM/PSNR 測定、`-CrfList` で点数指定可) |
| `tools/bdrate.py` | bench-nvenc の CSV から BD-rate (Bjontegaard) を計算 |

CUDA を使わない CPU のみのビルドにしたい場合は、`configure-msvc.sh` 末尾の
CUDA 追記ブロック (HAVE_CUDA / NVCC / CUDAFLAGS / CUDALIBS) を削除してください。
`x264/common/gpu.cu` はリンクされず、`--gpu-me` は無効になります。

---

## 4. 使い方

```bash
x264 --preset slow --crf 23 --gpu-me 2 input.yuv --input-res 1280x720 -o out.264
```

### オプション

| オプション | 既定値 | 説明 |
|---|---|---|
| `--gpu-me <0-3>` | `0` | GPU 動き探索モード (上記の表) |
| `--gpu-me-range <int>` | `64` | 粗探索の全解像度換算サーチ範囲 (4〜128) |
| `--gpu-me-radius <int>` | `2` | 粗探索候補周辺の全解像度リファイン半径 (0〜8) |
| `--gpu-me-refs <int>` | `2` | GPU 解析する参照フレーム数 (0 = すべて、最大 4) |
| `--gpu-me-device <int>` | `0` | CUDA デバイス番号 |

### 推奨構成

- **`--me esa` / `--me tesa` (高画質志向) との組み合わせが最も効果的です。**
  - `--gpu-me 2`: 約 +17〜29% 高速化 (SSIM 差は ±0.00002 程度)
  - `--gpu-me 3`: 約 +39〜47% 高速化 (SSIM 差は −0.00006 程度)
- **`--preset placebo` (tesa 24 + subme 11) でも効果があります。**
  - `--gpu-me 2`: 約 +12% 高速化 (画質ほぼ同等)
  - `--gpu-me 3`: mode 2 とほぼ同じ速度 (画質は −0.00004 程度)
- `--preset medium / slow` (hex/umh) では、GPU 解析コストに対して CPU 側の
  フルペル探索が軽いため、速度面では `--gpu-me 0` が最速です。
  画質を上げたい場合のみ `--gpu-me 2` を使ってください。

### 動作ログ

CUDA が使えた場合は起動時に次のように表示されます。

```
x264 [info]: CUDA ME: NVIDIA GeForce RTX 2060 SUPER (sm_75)
```

非対応条件では警告を出して CPU 動作に切り替わります。

```
x264 [warning]: CUDA ME: interlaced coding is not supported, falling back to CPU
x264 [warning]: CUDA ME: only supported in the 8-bit build, falling back to CPU
x264 [warning]: CUDA ME: unable to initialise CUDA, falling back to CPU
```

---

## 5. ベンチマーク

### 測定環境

| 項目 | 内容 |
|---|---|
| CPU | AMD Ryzen 5 5600 (6-core) |
| GPU | NVIDIA GeForce RTX 2060 SUPER 8GB (sm_75) |
| ドライバ | 32.0.15.9174 |
| CUDA | 13.3.73 |
| コンパイラ | MSVC 19.51 (VS 18 Community) |
| x264 | 0.165.x (`0480cb05fa188d`) + 本フォーク |

### 測定条件

- 入力: `testdata/motion720p.yuv` (1280x720, 240 frames, 30fps, 実写映像由来)
- エンコード: CRF 23, `--threads 1`, `--psnr --ssim`
- 比較: 同一ビルドで `--gpu-me 0/1/2/3` を切り替え (mode 0 は stock x264 とビット一致)
- 測定: `bench720.ps1` / `bench-placebo.ps1` / `bench-nvenc.ps1` を最終ビルドで実行

### 結果 (fps / SSIM)

| プリセット | mode 0 (GPU off) | mode 1 | mode 2 | mode 3 |
|---|---:|---:|---:|---:|
| medium (hex) | **41.25** / 0.9908696 | 36.26 / 0.9908802 | 36.17 / 0.9908833 | 37.20 / 0.9908659 |
| slow (umh) | **33.60** / 0.9908497 | 29.88 / 0.9908587 | 28.65 / 0.9908526 | 29.59 / 0.9908134 |
| esa 24 | 21.82 / 0.9908928 | — | 27.85 / 0.9908811 (+28%) | 32.08 / 0.9908641 (+47%) |
| tesa 24 | 18.63 / 0.9909359 | — | 23.98 / 0.9909279 (+29%) | 27.12 / 0.9908816 (+46%) |
| placebo (tesa) | 1.67 / 0.9912204 | 1.66 / 0.9912342 | 1.87 / 0.9912083 (+12%) | 1.86 / 0.9911766 (+11%) |

> 上表の測定 (セッション A) では esa/tesa/placebo で mode 2 が「高速化 + 画質ほぼ同等」、
> mode 3 が「大幅高速化 + 画質ほぼ同等」を両立しています。medium/slow では GPU 解析
> コストが相対的に大きくなるため mode 0 が最速です。

### x264 本家バイナリとの直接比較 (セッション B、同一条件で同時実行)

`x264-baseline.exe` (本家 x264 0.165.x、MSVC 19.51) と同一条件・同一セッションで
実行した結果 (720p 240 フレーム、CRF 23、`--threads 1`、esa/tesa 24):

| 設定 | 実行 | fps | SSIM_Y | 出力サイズ | SHA-1 |
|---|---|---:|---:|---:|---|
| esa 24 | 本家 x264 | 19.14 | 0.9908928 | 649,976 | `C3475626...` |
| esa 24 | 本フォーク mode 0 | 18.60 | 0.9908928 | 649,976 | `C3475626...` (バイト一致) |
| esa 24 | 本フォーク mode 2 | **23.16 (+21%)** | 0.9908811 | 648,798 | — |
| esa 24 | 本フォーク mode 3 | **26.64 (+39%)** | 0.9908641 | 645,404 | — |
| tesa 24 | 本家 x264 | 16.27 | 0.9909359 | 659,883 | `C1AEDE8A...` |
| tesa 24 | 本フォーク mode 0 | 16.71 | 0.9909359 | 659,883 | `C1AEDE8A...` (バイト一致) |
| tesa 24 | 本フォーク mode 2 | **19.03 (+17%)** | 0.9909279 | 658,315 | — |
| tesa 24 | 本フォーク mode 3 | **23.27 (+43%)** | 0.9908816 | 660,063 | — |

> mode 0 の出力は本家とバイト単位で一致 (SHA-1 同一)。mode 2/3 は SSIM 差
> −0.00001〜−0.00005 で、同等品質のまま高速化しています。fps はマシン負荷で
> 変動するため、セッション A (上表) とは絶対値が異なります。

### GPU 解析コスト (mode 1、参照キャッシュ有効時)

720p・238 フレーム解析時 (`--preset slow` = 4 refs) の GPU 処理内訳
(ms/フレーム、`X264_GPU_TIMING=1` で計測可能):

| 処理 | ms/frame |
|---|---:|
| 現フレーム転送 | 0.230 |
| 参照フレーム転送 (キャッシュミス時のみ) | 0.094 |
| ダウンサンプル | 0.278 |
| 粗探索カーネル | 0.567 |
| リファインカーネル | 0.930 |
| 結果転送 (D2H) | 0.137 |
| 同期 | 0.001 |
| **合計** | **2.238** |

参照フレームはデバイス上にキャッシュされるため、2 フレーム目以降は転送が
ほぼ発生しません (キャッシュ導入前は約 7.0 ms/frame でした)。

### GPU 固定機能 (NVENC) との画質比較

同じ入力・近いビットレートで、本フォーク (`--preset placebo --gpu-me 2`、
240 frames) と NVENC (Turing / RTX 2060 SUPER、ffmpeg `h264_nvenc` の最高品質設定:
`-preset p7 -tune hq -rc cbr -multipass fullres -spatial-aq 1 -temporal-aq 1 -bf 3
-b_ref_mode middle -rc-lookahead 32`) を比較しました。

| レート帯 | 本フォーク (CPU/GPU ハイブリッド) | NVENC 固定機能 |
|---|---|---|
| 低 (~430 kbps) | **430.8 kb/s**, SSIM_Y 0.987865, PSNR_y 42.70 (min 40.35) | 450.0 kb/s, SSIM_Y 0.981257, PSNR_y 40.06 (min 36.81) |
| 低中 (~580 kbps) | **584.7 kb/s**, SSIM_Y 0.991293, PSNR_y 44.60 (min 42.27) | 618.2 kb/s, SSIM_Y 0.988034, PSNR_y 42.13 (min 37.68) |
| 中 (~820 kbps) | **810.6 kb/s**, SSIM_Y 0.993658, PSNR_y 46.42 (min 44.22) | 860.7 kb/s, SSIM_Y 0.990661, PSNR_y 43.32 (min 38.26) |
| 高 (~1300 kbps) | **1293.2 kb/s**, SSIM_Y 0.995695, PSNR_y 48.64 (min 46.88) | 1333.7 kb/s, SSIM_Y 0.994278, PSNR_y 46.56 (min 42.38) |
| 高高 (~1700 kbps) | **1689.1 kb/s**, SSIM_Y 0.996465, PSNR_y 49.69 (min 47.95) | 1743.2 kb/s, SSIM_Y 0.995299, PSNR_y 47.85 (min 43.86) |

- すべてのレート帯で、本フォークは **NVENC より少ないビット量で高い SSIM/PSNR** を
  同時に達成しました (PSNR min も大きく優位 = 品質が崩れるフレームが少ない)。
- **BD-rate (Bjontegaard、5 点・同一品質でのビットレート差)**:
  PSNR 基準 **−40.7%**、SSIM 基準 **−33.6%**
  (重複範囲 42.7〜47.8 dB / 0.988〜0.995、範囲外は除外。重複率 66% / 53% の部分比較)。
- 品質は ffmpeg の `ssim` / `psnr` フィルタ (元 YUV とデコード結果の raw 比較) で
  同一条件で測定しています。

---

## 6. 検証済み事項

| 項目 | 結果 |
|---|---|
| `--gpu-me 0` が stock `x264-baseline.exe` とビット一致 | ✅ (720p medium / slow / placebo / esa 24 / tesa 24、1080p medium で SHA-1 一致) |
| x264 本家との同一セッション直接比較 | ✅ mode 2/3 が SSIM 同等のまま esa +21%/+39%、tesa +17%/+43% |
| GPU モード出力のデコード | ✅ ffmpeg で `-f null -` エラーなし (720p / 1080p) |
| 1080p (1920x1080) GPU モード | ✅ CUDA 有効のまま動作 (mode 2、反復実行バイト一致、フレームタイプ mode 0 と一致、mode 0 は stock とビット一致) |
| `--threads 4` / `--sliced-threads` | ✅ |
| B フレーム / B-pyramid / 重み付き予測 | ✅ (preset medium/slow は既定で有効) |
| 非 16 倍数解像度 (1920x1080 / 1280x718) | ✅ MB 境界パディングで GPU 加速 (mode 2 が CUDA 有効で動作) |
| 10-bit 出力 | ✅ 警告を出して CPU フォールバック |
| インターレース | ✅ 警告を出して CPU フォールバック |
| GPU モードの再現性 (同一入力・同一設定) | ✅ mode 1/2/3 の反復実行が SHA-1 一致 (240 frames placebo) |
| フレームタイプ決定が GPU モードに依存しない | ✅ `--gpu-me 0/1/2/3` で I/P/B 列が完全一致、stock とも一致 (240 frames placebo) |
| NVENC (固定機能) との画質比較 | ✅ 5 レート帯すべてで少ないビットで高 SSIM/PSNR (BD-rate −33.6% / −40.7%) |
| ヘッダ変更時の差分ビルド | ✅ `.d` が全インクルードヘッダを記録 (msvsdepend.sh 修正) |
| `selftest.ps1` (受け入れテスト 17 項目) | ✅ 全合格 |
| フルクリーンビルド (configure から) | ✅ 再現確認済み |

---

## 7. 内部設計

### 処理フロー

```
x264_encoder_encode()
  └─ reference_build_list() で参照リスト確定
      └─ gpu_me_analyse()                       [encoder/encoder.c]
          ├─ x264_gpu_frame_alloc()             フレームごとの候補バッファ
          └─ x264_gpu_analyze()                 [common/gpu.cu]
              ├─ 現フレームをアップロード
              ├─ 4x ダウンサンプル (downsample4)
              ├─ 参照フレームごとに:
              │   ├─ 参照キャッシュを検索 (frame ポインタ + i_frame がキー)
              │   ├─ 未キャッシュなら転送 + ダウンサンプル
              │   ├─ 粗探索 (coarse_search, 1/4 解像度・2 段グリッド)
              │   ├─ リファイン (refine_search, 全解像度 ±radius)
              │   └─ 候補 4 個を D2H コピー
              └─ 同期して完了
  └─ スライス符号化 (CPU)
      └─ x264_me_search_ref()                   [encoder/me.c]
          ├─ GPU 候補を取得し COST_MV で本物のコストを計算
          ├─ mode 1: 通常探索を継続
          ├─ mode 2: 探索範囲を ±4 に縮小して局所探索
          └─ mode 3: フルペル探索をスキップ
          └─ 以降のサブペル探索・RDO・エントロピー符号化はすべて元の x264
```

### 設計上の不変条件

- **GPU 候補はルックアヘッド (低解像度 ME) には一切渡されません。**
  `x264_me_search_ref()` 側が `m->i_list >= 0` を要求し、`encoder/slicetype.c` の
  低解像度探索は `i_list = -1` を設定します。これによりフレームタイプ (I/P/B) の
  決定は GPU モードに依存しません (`--gpu-me 0/1/2/3` でフレームタイプ列が完全一致)。
- **候補は追加のみ。** 最終的な MV 決定・モード判定は既存 x264 のコスト関数
  (SATD + λ·MV コスト、RDO) が行うため、GPU の近似 SAD が画質を直接劣化させる
  ことはありません。
- **再現性。** 同一入力・同一設定の反復実行はバイト単位で同一の出力になります
  (mode 0 は stock x264 とビット一致)。

### カーネル

| カーネル | 内容 |
|---|---|
| `downsample4` | 4x4 平均で 1/4 解像度の輝度プレーンを作成 |
| `coarse_search` | 1/4 解像度で ±range/4 を 2 段グリッド探索 (2 ピクセル刻み → 周囲 8 近傍)。4x4 SAD で上位 4 候補を保持 |
| `refine_search` | 粗探索候補の周囲 ±radius を全解像度で再探索。16x16 を 2 ピクセル間引きの 64 サンプル SAD でランキングし、候補を確定 |

候補の確定順位付けは GPU の近似 SAD で行いますが、**最終評価は CPU の
`fpelcmp` (SATD) + MV コスト + RDO** です。このため GPU 側の近似が
画質を直接悪化させることはありません。

### 参照フレームキャッシュ

- デバイスメモリ上に最大 8 面 (`X264_GPU_CACHE`) の参照プレーン (+ 1/4 解像度版) を保持。
- キーは「プレーン先頭ポインタ + `x264_frame_t::i_frame`」。x264 はフレーム
  オブジェクトを使い回すため、`i_frame` を併用して世代を識別します。
- LRU ではなくリング置換。参照は毎フレーム再利用されるため、ヒット率は非常に高い。

### スレッド安全性

- GPU モジュールは `std::mutex` で解析を直列化。解析はメインスレッドから
  フレーム開始前に 1 回だけ実行されます。
- 候補バッファ (`x264_gpu_frame_t`) はソースフレームオブジェクトごとに割り当て
  られ、`x264_frame_delete()` で解放されます。
- スライス/フレームスレッドはホスト側の候補配列 (読み取り専用) を参照します。

### 実装ファイル

| ファイル | 変更内容 |
|---|---|
| `common/gpu.h` / `common/gpu.cu` | 新規。CUDA モジュール本体 (カーネル、転送、キャッシュ、API) |
| `x264.h` | `x264_param_t.analyse` に `i_gpu_me*` を追加 |
| `common/base.c` | 既定値・オプション解析・パラメータ文字列 |
| `common/frame.h` / `frame.c` | フレームごとの候補バッファ (`gpu_me`) と解放処理 |
| `common/common.h` | エンコーダごとの GPU コンテキスト (`gpu_ctx`) |
| `encoder/encoder.c` | GPU コンテキストの生成/破棄、`gpu_me_analyse()`、フォールバック警告 |
| `encoder/me.h` / `encoder/analyse.c` | `x264_me_t` に `i_list` / `i_xoff` / `i_yoff` を追加し、候補検索用の情報を設定 |
| `encoder/me.c` | GPU 候補の注入 (mode 1/2/3)。`i_list >= 0` を必須条件とする |
| `encoder/slicetype.c` | ルックアヘッドの低解像度 ME に `i_list = -1` を設定し、GPU 候補の注入対象外にする |
| `Makefile` | `common/gpu.o` のビルド (nvcc) と `cudart_static.lib` のリンク |
| `x264/tools/msvsdepend.sh` | Windows 差分ビルドのヘッダ依存生成を修正 (日本語ロケールの `/showIncludes` 対応、ビット深度定義の付与、失敗時のビルド中断) |
| `configure-msvc.sh` / `make-msvc.sh` | CUDA 設定の追記とライブラリパス設定 |
| `dev/gpu_me_bench.cu` | 開発用の単体 CUDA ベンチマーク (エンコーダ本体にはリンクされません) |
| `bench720.ps1` / `bench-results/` | ベンチマーク実行スクリプトと全測定ログ (NVENC 比較含む) |

---

## 8. 既知の制限と今後の課題

- **8-bit のみ**。10-bit / 12-bit エンコードでは GPU は使われません (CPU フォールバック)。
- **インターレース非対応** (CPU フォールバック)。
- **x265/HEVC 版は未収録**。本フォークは x264 (H.264) のみです。
- 解析はフレーム先頭で**同期実行**のため、medium/slow のような CPU 探索が軽い
  プリセットでは相対的にオーバーヘッドが目立ちます。行単位の非同期パイプライン化
  (GPU 解析と CPU 符号化のオーバーラップ) は今後の課題です。
- 参照キャッシュは 8 面固定。`--gpu-me-refs` を大きくしすぎた場合や、参照の
  入れ替わりが多い場合はキャッシュミスが増えます。
- 候補数は 4 個固定 (`X264_GPU_CANDS`)。変更する場合は `common/gpu.cu` の
  `GK` と候補バッファサイズが連動します。

---

## 9. ベンチマークの再現方法

```bash
# 評価用クリップ (1280x720 / 240 frames / 30fps) を作る例
ffmpeg -ss 20 -i source.webm -frames:v 240 -vf scale=1280:720 \
       -pix_fmt yuv420p -f rawvideo testdata/motion720p.yuv

# 実行 (PowerShell)
powershell -NoProfile -ExecutionPolicy Bypass -File bench720.ps1
```

`bench720.ps1` は medium / slow / esa / tesa の各設定で mode 0〜3 を実行し、
fps・ビットレート・SSIM・PSNR・ファイルサイズを `testdata/*.log` と
`bench-results/` に出力します。placebo プリセットの比較は `bench-placebo.ps1`
(同じく mode 0〜3) で再現できます。

NVENC との画質比較を再現する場合:

```powershell
# ffmpeg (NVENC 対応ビルド) のパスを指定して実行
powershell -NoProfile -ExecutionPolicy Bypass -File bench-nvenc.ps1 `
           -Ffmpeg "C:\path\to\ffmpeg.exe"
# 既定では 240 フレームで placebo + --gpu-me 2 と NVENC (p7/最高品質設定) を比較
# 結果はコンソールと bench-results/nvenc-comparison/repro_summary.csv に出力
# -CrfList "26,23,20,16,14" のように点数を増やすと BD-rate の重複範囲が広がる

# BD-rate (Bjontegaard) を CSV から計算 (Python 3 が必要)
py -3 tools/bdrate.py bench-results/nvenc-comparison/repro_summary.csv
```

> `tools/bdrate.py` は acceleratedencode_GPU (MIT) の同種ツールの手法を参考に、
> 本プロジェクト用に再実装したものです。品質カーブの重複範囲外は除外し、
> 重複率を併記します。

受け入れテスト (ビット一致・全 GPU モードの動作・デコード・フォールバック・決定性):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File selftest.ps1
# ffmpeg / ffprobe が PATH に無い場合:
#   ... -File selftest.ps1 -Ffmpeg C:\path\to\ffmpeg.exe
```

> ビット一致検証に使う本家 x264 (`x264-baseline.exe`) はリポジトリに含まれません。
> 本フォークと同じコミット (`0480cb05fa188d37ae87e8f4fd8f1aea3711f7ee`) の本家
> x264 をビルドし、プロジェクト直下に `x264-baseline.exe` として置くと、
> `selftest.ps1` / `det_battery.ps1` のビット一致チェックが有効になります。

決定性バッテリー (mode 1/2/3 の反復一致と、全モードのフレームタイプ列の検証):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File det_battery.ps1
# ffprobe が PATH に無い場合: -Ffprobe C:\path\to\ffprobe.exe
```

GPU 解析の内訳を確認したい場合:

```bash
X264_GPU_TIMING=1 x264 --preset slow --crf 23 --gpu-me 1 input.yuv ... -o out.264
# 終了時に ms/frame の内訳が標準出力に表示されます
```

---

## 10. ライセンス

x264 本体と同じ **GPL v2 以降**です (`x264/COPYING` を参照)。
本フォークの追加部分 (`common/gpu.h`, `common/gpu.cu` および各差分) も
GPL v2 以降で配布されます。
