# hybrid-x264-gpu

x264 (0.165.x, commit `0480cb05fa188d37ae87e8f4fd8f1aea3711f7ee`) をベースに、
**CUDA / OpenCL による整数精度動き探索 (ME) の候補生成**を追加した CPU/GPU ハイブリッド
H.264 エンコーダです。

- GPU は NVENC のような固定機能ではなく、汎用 CUDA / OpenCL コアで動き探索の候補を計算します。
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
| GPU 利用 | 汎用 CUDA / OpenCL カーネル (NVENC/QSV 等の固定機能は不使用) |
| 画質 | GPU は候補を「追加」するだけ。CPU が本物のコストで選び直すため、実測でも SSIM 差は ±0.00005 以内 |
| 互換性 | 出力は通常の H.264 ストリーム。MPC-HC / ffmpeg 等でデコード可能 (検証済み) |
| 対応解像度 | 任意 (x264 準拠。MB 境界までパディングして GPU 解析。1080p も GPU 加速) |
| 速度 | `--me esa` / `--me tesa` で **+17〜+47%** (2 測定セッション。下記ベンチマーク) |
| 既定動作 | `--gpu-me 0`。従来と完全に同一・ビット一致 |
| フォールバック | CUDA 不可の GPU は OpenCL へ自動切替。GPU 自体が無い / 非対応条件では警告を出して自動的に CPU 動作へ |

### モード一覧

| `--gpu-me` | 動作 | 用途 |
|---|---|---|
| `0` | GPU オフ (既定) | 従来と同じ。ビット一致 |
| `1` | 16x16 ME に GPU 候補を追加してから通常探索 | 画質重視。候補が追加されるため画質は同等以上 |
| `2` | 全パーティションに GPU 候補を追加し、CPU 探索範囲を ±4 に縮小 | **推奨**。esa/tesa で +28〜29% 高速化 (品質ゲート通過) |
| `3` | 全パーティションに GPU 候補を追加し、CPU のフルペル探索を省略 | 最速。品質は内容依存 (BD-rate +0.4〜+1.9%、下記品質ゲート参照) |
| `4` | GPU 候補を追加し、CPU 探索範囲を ±1 に縮小 | 速度と品質のバランス (mode 2 と 3 の中間) |

---

## 2. 必要環境

- Windows x64
- Visual Studio (C++ ツールセット。動作確認: VS 18 Community / cl 19.51)
- CUDA Toolkit 13.3 (`nvcc`、`cudart_static.lib`、および OpenCL バックエンド用の CL ヘッダ) — 動作確認: 13.3.73
- NASM 2.16.03 (`toolchain/nasm-2.16.03` に配置済み。別バージョンでも可)
- Git Bash + `mingw32-make` (MSYS2/MinGW の make でも可)
- NVIDIA GPU: CUDA バックエンドは Turing〜Blackwell (sm_75/80/86/87/89/90/100/120)。
  Kepler〜Volta (sm_35〜sm_70) は CUDA Toolkit 13 がコンパイル非対応のため、
  OpenCL バックエンドが担当します (NVIDIA ドライバ同梱の OpenCL を使用)
- OpenCL 1.2+ ランタイム (任意。`--gpu-me-backend 2` 用。動作確認: NVIDIA OpenCL 3.0)

GPU 対応条件 (これ以外は自動で CPU フォールバック):

- 8-bit エンコード (`--output-depth 8`)
- プログレッシブ (インターレースは非対応)

GPU 世代ごとのバックエンド (既定の自動選択。実行時に解析が失敗した場合は
もう一方のバックエンドへ 1 回だけ切り替え、警告を表示します):

| GPU 世代 | CUDA | OpenCL | 備考 |
|---|---|---|---|
| Kepler〜Volta (sm_35〜sm_70) | – | ✅ | CUDA 13 が非対応 (OpenCL で動作) |
| Turing (sm_75) | ✅ | ✅ | 動作確認: RTX 2060 SUPER |
| Ampere (sm_80/86/87) | ✅ | ✅ | SASS を同梱 |
| Ada (sm_89) | ✅ | ✅ | SASS を同梱 |
| Hopper (sm_90) | ✅ | ✅ | SASS を同梱 |
| Blackwell (sm_100/120) | ✅ | ✅ | SASS を同梱 |
| それ以降の世代 | ✅ | ✅ | compute_75 PTX を JIT |

解像度は x264 が受け付ける任意のサイズに対応します。フレームプレーンは
MB 境界までパディングされて解析されるため、1920x1080 のような 16 の倍数で
ない解像度でも GPU 加速されます。

---

## 3. ビルド手順

```bash
# 1) 環境変数 (CUDA の場所。configure-msvc.sh / make-msvc.sh の既定値と同じ)
export CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"

# 2) configure (build-msvc/ を作成し config.mak / config.h に GPU 設定を追記)
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
> CUDA バックエンドは既定で Turing〜Blackwell (sm_75/80/86/87/89/90/100/120) の
> SASS と、将来 GPU 用の PTX (compute_75) を 1 つのバイナリに同梱します。
> 対象は環境変数 `CUDA_ARCHS` / `CUDA_PTX` で指定できます
> (`configure-msvc.sh` を参照)。CUDA Toolkit 13 は sm_75 未満をコンパイル
> できないため、Kepler〜Volta は OpenCL バックエンドが担当します
> (ドライバ同梱の OpenCL を使用。実行時に CUDA が使えない場合は自動で切替)。
> OpenCL バックエンドは CL ヘッダ (CUDA Toolkit 同梱) を使ってビルドされ、
> `OpenCL.dll` は実行時に動的ロードします (無い環境でも起動し、OpenCL
> バックエンドだけが無効になります)。

> ビルド済みの Windows x64 バイナリ (`x264.exe`) は
> [Releases](https://github.com/trprkkk/x264-GPU-accelerated/releases) で配布しています。
> 単体で動作し (同梱 DLL 不要。依存は Windows 標準の SHELL32 / KERNEL32 のみ)、
> NVIDIA ドライバがあれば CUDA バックエンド、その他の OpenCL ドライバでも
> OpenCL バックエンドが使えます。どちらも無い環境では警告を出して CPU エンコードに
> 切り替わります。

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
| `bench-quality.ps1` | 品質ゲート用ベンチマーク (クリップ×モード×CRF×radius で SSIM/PSNR/VMAF) |
| `bench-nvenc.ps1` | NVENC 比較 (同一ビットレート・SSIM/PSNR 測定、`-CrfList` で点数指定可) |
| `tools/quality-gate.py` | bench-quality の CSV から BD-rate を判定 (mode 0 基準、しきい値判定) |
| `tools/bdrate.py` | NVENC 比較 CSV から BD-rate (Bjontegaard) を計算 |

GPU (CUDA / OpenCL) を使わない CPU のみのビルドにしたい場合は、
`configure-msvc.sh` 末尾の GPU 追記ブロック (HAVE_CUDA / NVCC / CUDAFLAGS /
CUDALIBS / OPENCL_INC / OPENCL_LIB) を削除してください。
`x264/common/gpu*.c*` はリンクされず、`--gpu-me` は無効になります。

---

## 4. 使い方

```bash
x264 --preset slow --crf 23 --gpu-me 2 input.yuv --input-res 1280x720 -o out.264
```

### オプション

| オプション | 既定値 | 説明 |
|---|---|---|
| `--gpu-me <0-4>` | `0` | GPU 動き探索モード (上記の表) |
| `--gpu-me-backend <0-2>` | `0` | バックエンド: 0=自動 (CUDA→OpenCL)、1=CUDA、2=OpenCL。実行時の解析失敗時はもう一方へ 1 回だけ自動切替 (警告表示) |
| `--gpu-me-range <int>` | `64` | 粗探索の全解像度換算サーチ範囲 (4〜128) |
| `--gpu-me-radius <int>` | `2` | 粗探索候補周辺の全解像度リファイン半径 (0〜8) |
| `--gpu-me-refs <int>` | `2` | GPU 解析する参照フレーム数 (0 = すべて、最大 4) |
| `--gpu-me-device <int>` | `0` | GPU デバイス番号 |

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

GPU が使えた場合は起動時にバックエンドとデバイスを表示します。

```
x264 [info]: GPU ME: cuda / NVIDIA GeForce RTX 2060 SUPER (sm_75)
x264 [info]: GPU ME: opencl / NVIDIA GeForce RTX 2060 SUPER (OpenCL 3.0 CUDA)
```

非対応条件では警告を出して CPU 動作に切り替わります。

```
x264 [warning]: GPU ME: interlaced coding is not supported, falling back to CPU
x264 [warning]: GPU ME: only supported in the 8-bit build, falling back to CPU
x264 [warning]: GPU ME: no usable GPU backend, falling back to CPU
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
| OpenCL | NVIDIA OpenCL 3.0 (同 GPU。バックエンド比較用) |
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

### 品質ゲート (BD-rate、3クリップ × esa/tesa × CRF 3点)

`bench-quality.ps1` + `tools/quality-gate.py` で、各 GPU モードの BD-rate
(PSNR/SSIM/VMAF、基準は mode 0 = 本家 x264 とビット一致) を自動計測します。

- クリップ: motion720p (240f) / test720p (120f) / test1080p (60f)
- 条件: CRF 20/23/26、esa 24 / tesa 24、各比較 3 点の BD-rate

| モード | 最悪 BD-rate (PSNR / SSIM / VMAF) | 備考 |
|---|---|---|
| 1 | +0.30% / +0.22% / +0.66% | 候補追加のみ |
| **2 (推奨)** | **+0.28% / +0.37% / +0.75%** | 全クリップで実質同等 |
| 3 | +1.50% / +1.92% / +1.14% | test1080p でペナルティ大 |
| 4 | +1.50% / +1.92% / +1.14% | ESA では mode 3 のペナルティが半減 (1080p: SSIM +1.53%→+0.64%) |

- ゲート判定 (**per-comparison ≤ +2.0%、平均 ≤ +1.0%**): **PASS**。
  全 26 比較の平均は PSNR +0.23% / SSIM +0.28% / VMAF +0.06%。
  strict 判定 (>+0.5%) は 16 比較で、mode 3/4 の test1080p と motion720p の
  tesa SSIM に集中しています。
- mode 3/4 の 1080p ペナルティは候補カバレッジ由来で、リファイン半径の拡大
  (radius 3/4) では改善しないことを確認 (`bench-results/radius-experiment.txt`)。
  参照単位のプルーニングは実測で逆効果のため不採用 (`prune-experiment.txt`)。
- 生データ: `bench-results/quality/quality_summary.csv`、
  判定結果: `bench-results/quality-gate-baseline.txt`。

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
| 非同期プリフェッチ (`X264_GPU_ASYNC`) | ✅ 出力はビット一致のまま (全回帰ハッシュ一致)。GPU 待ち時間 ≈0.002 ms/frame、実測 +1% 前後 |
| OpenCL バックエンド (`--gpu-me-backend 2`) | ✅ CUDA とビット単位で同一ストリーム (720p/1080p、mode 2/3)。selftest で自動検証 |
| CUDA アーキテクチャ (配布バイナリ) | ✅ SASS sm_75/80/86/87/89/90/100/120 + compute_75 PTX を同梱 (cuobjdump で確認)。Kepler〜Volta は OpenCL が担当 |
| 実行時バックエンドフォールバック | ✅ 起動時プローブで非対応 GPU を検出、解析失敗時はもう一方へ 1 回切替 (出力はビット一致のまま)。selftest で自動検証 |
| 10-bit 出力 | ✅ 警告を出して CPU フォールバック |
| インターレース | ✅ 警告を出して CPU フォールバック |
| GPU モードの再現性 (同一入力・同一設定) | ✅ mode 1/2/3 の反復実行が SHA-1 一致 (240 frames placebo) |
| フレームタイプ決定が GPU モードに依存しない | ✅ `--gpu-me 0/1/2/3` で I/P/B 列が完全一致、stock とも一致 (240 frames placebo) |
| NVENC (固定機能) との画質比較 | ✅ 5 レート帯すべてで少ないビットで高 SSIM/PSNR (BD-rate −33.6% / −40.7%) |
| 品質ゲート (BD-rate × 3クリップ) | ✅ PASS (26比較。最悪 SSIM +1.92% は mode 3/4 の 1080p、mode 2 は +0.37% 以内) |
| ヘッダ変更時の差分ビルド | ✅ `.d` が全インクルードヘッダを記録 (msvsdepend.sh 修正) |
| `selftest.ps1` (受け入れテスト 27 項目) | ✅ 全合格 |
| フルクリーンビルド (configure から) | ✅ 再現確認済み |

---

## 7. 内部設計

### 処理フロー

```
x264_encoder_encode()
  └─ reference_build_list() で参照リスト確定
      └─ gpu_me_analyse()                       [encoder/encoder.c]
          ├─ 前フレームで先出しした解析が参照リストを覆っていれば採用
          │   (同期処理はスキップ。覆っていなければ下記の同期解析を実行)
          ├─ x264_gpu_frame_alloc()             フレームごとの候補バッファ
          └─ x264_gpu_analyze()                 [common/gpu.cu]
              ├─ 現フレームをアップロード (ピン留めステージング経由)
              ├─ 4x ダウンサンプル (downsample4)
              ├─ 参照フレームごとに:
              │   ├─ 参照キャッシュを検索 (frame ポインタ + i_frame がキー)
              │   ├─ 未キャッシュなら転送 + ダウンサンプル
              │   ├─ 粗探索 (coarse_search, 1/4 解像度・2 段グリッド)
              │   ├─ リファイン (refine_search, 全解像度 ±radius)
              │   └─ 候補 4 個を D2H コピー (ピン留めメモリ)
              └─ 同期して完了
  └─ スライス符号化 (CPU)
      └─ x264_me_search_ref()                   [encoder/me.c]
          ├─ GPU 候補を取得し COST_MV で本物のコストを計算
          ├─ mode 1: 通常探索を継続
          ├─ mode 2: 探索範囲を ±4 に縮小して局所探索
          ├─ mode 3: フルペル探索をスキップ
          ├─ mode 4: 探索範囲を ±1 に縮小して局所探索
          └─ 以降のサブペル探索・RDO・エントロピー符号化はすべて元の x264
  └─ gpu_me_prefetch()                          [encoder/encoder.c]
      (次フレームの解析を GPU に先出しし、NAL 出力と次フレームの CPU 前処理
       とオーバーラップさせる。既定で有効、`X264_GPU_ASYNC=0` で無効化)
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
- **先出し解析でも出力は不変。** 次フレーム向けの解析は最終的な参照リストを
  予測して先に走りますが、候補は参照フレームの同一性 (プレーン先頭ポインタ +
  `i_frame`) で照合し、**最終リストを覆っている場合のみ採用**します。覆って
  いなければ従来どおりの同期解析にフォールバックするため、ビットストリームは
  非同期の有無で変わりません (全回帰ハッシュで一致を確認)。

### バックエンド (CUDA / OpenCL)

`common/gpu.c` が公開 API のディスパッチャで、`common/gpu.cu` (CUDA) と
`common/gpu_cl.c` (OpenCL) が同一アルゴリズムを実装します。

- 選択: `--gpu-me-backend 0|1|2` (既定 0 = 自動で CUDA → OpenCL の順に試行)。
  使用中のバックエンドとデバイスは起動ログ (`GPU ME: <backend> / <device>`) に表示。
- **実行時フォールバック。** 起動時に CUDA カーネルが動かない GPU (非対応
  アーキテクチャ等) は作成時のプローブ起動で検出し、自動選択が OpenCL を
  選びます。エンコード中に解析が失敗した場合も、そのバックエンドを退役させて
  もう一方へ 1 回だけ切り替えます (保持していた候補バッファは解放して作り直し、
  `GPU ME: switched to the <backend> backend` の警告を表示)。どちらも
  使えない場合は従来どおり警告を出して CPU 動作になります。
- **両バックエンドの出力はビット単位で一致します。** カーネルは走査順・比較
  演算子まで同一で、候補は参照の同一性 (プレーン先頭ポインタ + `i_frame`) で
  照合されます。バックエンド名は SEI のパラメータ文字列に含めないため、
  ストリームをそのままハッシュ比較できます (selftest で自動検証)。切替が
  発生しても、切替後の出力は切替前と同じビットストリームになります。
- 転送は両バックエンドともピン留めステージングを使い、非同期プリフェッチが
  そのまま動作します。OpenCL は同一出力のまま CUDA よりわずかに低速です
  (実測: esa 24 mode 2・240 フレームで CUDA 8.82 s / OpenCL 9.17 s)。

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
| `common/gpu.h` / `common/gpu.c` | 新規。公開 API とバックエンド・ディスパッチャ (実行時フォールバックとフレーム所有権の振り分けを含む) |
| `common/gpu.cu` | 新規。CUDA バックエンド (カーネル、転送、キャッシュ) |
| `common/gpu_cl.c` | 新規。OpenCL バックエンド (CUDA と同一アルゴリズム・同一出力) |
| `x264.h` | `x264_param_t.analyse` に `i_gpu_me*` を追加 |
| `common/base.c` | 既定値・オプション解析・パラメータ文字列 |
| `common/frame.h` / `frame.c` | フレームごとの候補バッファ (`gpu_me`) と解放処理 |
| `common/common.h` | エンコーダごとの GPU コンテキスト (`gpu_ctx`) |
| `encoder/encoder.c` | GPU コンテキストの生成/破棄、`gpu_me_analyse()`、フォールバック警告 |
| `encoder/me.h` / `encoder/analyse.c` | `x264_me_t` に `i_list` / `i_xoff` / `i_yoff` を追加し、候補検索用の情報を設定 |
| `encoder/me.c` | GPU 候補の注入 (mode 1/2/3/4)。`i_list >= 0` を必須条件とする |
| `encoder/slicetype.c` | ルックアヘッドの低解像度 ME に `i_list = -1` を設定し、GPU 候補の注入対象外にする |
| `Makefile` | `common/gpu*.o` (ディスパッチャ / CUDA / OpenCL) のビルドと `cudart_static.lib` のリンク |
| `x264/tools/msvsdepend.sh` | Windows 差分ビルドのヘッダ依存生成を修正 (日本語ロケールの `/showIncludes` 対応、ビット深度定義の付与、失敗時のビルド中断) |
| `configure-msvc.sh` / `make-msvc.sh` | GPU 設定 (CUDA / OpenCL) の追記とライブラリパス設定 |
| `dev/gpu_me_bench.cu` | 開発用の単体 CUDA ベンチマーク (エンコーダ本体にはリンクされません) |
| `bench720.ps1` / `bench-results/` | ベンチマーク実行スクリプトと全測定ログ (NVENC 比較含む) |

---

## 8. 既知の制限と今後の課題

- **8-bit のみ**。10-bit / 12-bit エンコードでは GPU は使われません (CPU フォールバック)。
- **インターレース非対応** (CPU フォールバック)。
- **CUDA バックエンドは Turing (sm_75) 以降のみ**。Kepler〜Volta は CUDA
  Toolkit 13 がコンパイル非対応のため、OpenCL バックエンドで動作します
  (ドライバ同梱の OpenCL を使用)。
- **x265/HEVC 版は未収録**。本フォークは x264 (H.264) のみです。
- **OpenCL バックエンドは NVIDIA 実機のみ検証済み**。カーネルはポータブルな
  OpenCL C 1.2 の範囲のみ使用していますが、AMD/Intel 実機での検証は未実施です。
  OpenCL も同一出力のまま CUDA よりわずかに低速 (~4%)。
- 解析はフレーム先頭で**同期実行**でしたが、先出し解析 (`gpu_me_prefetch`) により
  GPU 待ちはほぼ隠蔽済み (0.002 ms/frame)。medium/slow のような CPU 探索が軽い
  プリセットでは、依然として GPU 解析の相対コストが大きくなります
  (mode 0 が最速)。
- 参照キャッシュは 8 面固定。`--gpu-me-refs` を大きくしすぎた場合や、参照の
  入れ替わりが多い場合はキャッシュミスが増えます。
- 候補数は 4 個固定 (`X264_GPU_CANDS`)。変更する場合は `common/gpu.cu` の
  `GK` と候補バッファサイズが連動します。
- **参照単位の努力量プルーニングは検討の上で不採用**。GPU の粗い SAD 比で
  劣る参照の CPU 探索を削ると、x264 のコスト閾値が反応して評価する分割が
  増え (+20% の ME 呼び出し)、実測で最大 −8% の逆効果でした (詳細:
  `bench-results/prune-experiment.txt`)。速度は既存の全体ノブ (mode 2 の
  探索半径、mode 3 のフルペル省略) で調整します。

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

OpenCL バックエンドの検証 (CUDA とハッシュ一致することを確認できます):

```powershell
# CUDA (または自動選択) と OpenCL で同一設定をエンコードして比較
build-msvc\x264.exe --preset placebo --crf 23 --threads 1 --gpu-me 2 `
    testdata\motion720p.yuv --input-res 1280x720 --fps 30 -o a.264
build-msvc\x264.exe --preset placebo --crf 23 --threads 1 --gpu-me 2 --gpu-me-backend 2 `
    testdata\motion720p.yuv --input-res 1280x720 --fps 30 -o b.264
Get-FileHash a.264, b.264 -Algorithm SHA1   # 同じハッシュになる
```

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

X264_ME_PROF=1 x264 --preset slow --crf 23 --gpu-me 0 input.yuv ... -o out.264
# モーションサーチ (サブペル含む) の合計時間とエンコード内シェアを表示します
# (--threads 1 での計測用)

X264_GPU_PROF=1 x264 --preset slow --crf 23 --gpu-me 2 input.yuv ... -o out.264
# x264_gpu_analyze*() の実呼び出し時間と、先出し解析の待ち時間を表示します

X264_GPU_ASYNC=0 x264 ... # 非同期プリフェッチ (先出し解析) を無効化して同期実行
```

品質ゲート (全クリップで GPU モードの BD-rate がしきい値内かを判定):

```powershell
# クリップ × モード × CRF で SSIM/PSNR/VMAF を計測 (再実行時は済んだ組み合わせをスキップ)
powershell -NoProfile -ExecutionPolicy Bypass -File bench-quality.ps1 `
           -Ffmpeg "C:\path\to\ffmpeg.exe" -Configs esa,tesa -Modes 0,1,2,3
# BD-rate 判定 (per-comparison <= +0.5%, 平均 <= +0.2% が既定しきい値)
py -3 tools/quality-gate.py bench-results/quality/quality_summary.csv
```

---

## 10. ライセンス

x264 本体と同じ **GPL v2 以降**です (`x264/COPYING` を参照)。
本フォークの追加部分 (`common/gpu.h`, `common/gpu.cu` および各差分) も
GPL v2 以降で配布されます。
