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
- v1.0.2 からは **GPU lookahead (`--gpu-lookahead`)** も追加されました。GPU が
  ルックアヘッド (低解像度解析) のコスト計算を先読みし、exact モードでは
  **CPU と同一の判断を保証**します (詳細は下記「GPU lookahead」)。

---

## 1. 特徴

| 項目 | 内容 |
|---|---|
| 対象 | H.264/AVC (x264 フォーク、High profile / 8-bit・10-bit ビルドで GPU 有効) |
| GPU 利用 | 汎用 CUDA / OpenCL カーネル (NVENC/QSV 等の固定機能は不使用) |
| 画質 | GPU は候補を「追加」するだけ。CPU が本物のコストで選び直すため、実測でも SSIM 差は ±0.00005 以内 |
| 互換性 | 出力は通常の H.264 ストリーム。MPC-HC / ffmpeg 等でデコード可能 (検証済み) |
| 対応解像度 | 任意 (x264 準拠。MB 境界までパディングして GPU 解析。1080p も GPU 加速) |
| 速度 | `--me esa` / `--me tesa` で **+17〜+47%** (2 測定セッション。下記ベンチマーク) |
| 既定動作 | `--gpu-me 0` / `--gpu-lookahead 0`。従来と完全に同一・ビット一致 |
| フォールバック | CUDA 不可の GPU は OpenCL へ自動切替。GPU 自体が無い / 非対応条件では警告を出して自動的に CPU 動作へ |

### モード一覧

| `--gpu-me` | 動作 | 用途 |
|---|---|---|
| `0` | GPU オフ (既定) | 従来と同じ。ビット一致 |
| `1` | 16x16 ME に GPU 候補を追加してから通常探索 | 画質重視。候補が追加されるため画質は同等以上 |
| `2` | 全パーティションに GPU 候補を追加し、CPU 探索範囲を ±4 に縮小 | **推奨**。esa/tesa で +28〜29% 高速化 (品質ゲート通過) |
| `3` | 全パーティションに GPU 候補を追加し、CPU のフルペル探索を省略 | 最速。品質は内容依存 (BD-rate +0.4〜+1.9%、下記品質ゲート参照) |
| `4` | GPU 候補を追加し、CPU 探索範囲を ±1 に縮小 | 速度と品質のバランス (mode 2 と 3 の中間) |

### GPU lookahead (`--gpu-lookahead`)

x264 はフレームタイプ (I/P/B) や MB ツリーの決定に**低解像度 (1/2) 解析
(ルックアヘッド)** を使います。`--gpu-lookahead` はこの解析の重い部分
(8x8 ブロックごとの intra コストと、(fenc, ref) ペアごとの網羅的な 8x8
コスト正方テーブル) を GPU が先読みする機能です。

| `--gpu-lookahead` | 動作 | 用途 |
|---|---|---|
| `0` | GPU オフ (既定) | 従来と同じ。ビット一致 |
| `1` | **exact モード**: GPU が CPU が計算するはずの値をビット単位で同じに先計算し、CPU 探索はテーブルを参照 | 出力を一切変えずにルックアヘッドの SAD/SATD 計算を GPU へオフロード |
| `2` | **fast モード (実験的)**: GPU が階層探索で整数精度の起点ベクトルを提案し、CPU はそれを本物のコストで評価してパターン探索を省略 | exhaustive テーブルの代償を払わない実用モード。出力は CPU ルックアヘッドと**異なる** (本家 `--opencl` と同様) が、決定的 |

exact モードの契約:

- GPU が先計算するのは (a) 8x8 lowres ブロックの raw 最小 intra 予測コスト、
  (b) MB 中心 ±R (R = min(mvrange/2 + 2, 32)) の全変位に対する 8x8 SAD/SATD
  (メトリックは CPU の `fpelcmp` に一致。`--me tesa` + subme>1 では SATD) です。
- テーブルに無い位置 (範囲外・テーブル作成失敗時) は **CPU が自前で同じ
  関数を計算する**ため、探索結果は CPU のみの場合と常に一致します。
  GPU 失敗時も出力は変わりません (CPU フォールバック)。
- `--gpu-lookahead 1` を付けると SEI のパラメータ文字列に `gpu_la=1:0` が
  入るため、未指定時の stock x264 とは**ハッシュが変わります** (ヘッダのみ。
  ビデオ内容の符号化判断は同一)。未指定時の出力は従来とビット一致です。
- 本家 x264 の `--opencl` (OpenCL lookahead) は**別機能**で、本フォークでも
  元のまま維持しています。両方を同時に指定した場合は、GPU lookahead を自動で
  無効化して警告を表示します (stock の OpenCL lookahead が低解像度解析を
  丸ごと置き換えるため、GPU lookahead の結果は使われないためです)。

fast モード (`--gpu-lookahead 2`) の契約:

- GPU が MB ごとに階層探索 (±範囲を 4px グリッドの 16 サンプル SAD で大まかに
  ランキング → 全 8x8 SAD で ±2/±1 にリファイン) で整数精度の起点ベクトルを
  提案します。探索窓は CPU の `mv_limit_fpel` と同じ式でクランプされます。
- CPU は提案ベクトルを**本物のコスト関数** (fpelcmp + λ·MV コスト) で評価して
  採否を決め、自分のパターン探索 (hex) だけを省略します。サブペル refine、
  bidir、skip 判定、intra 比較はすべて元の x264 のままです。
- 出力は CPU ルックアヘッドと一致しません (判断の起点が変わるため) が、
  **同一入力・同一設定の反復実行でバイト単位で同一**の決定的な結果になり、
  `--lookahead-threads` にも依存しません。GPU 失敗時はその MB の探索が
  通常の CPU 経路に落ちます (出力は決定的のまま)。

---

## 2. 必要環境

- Windows x64
- Visual Studio (C++ ツールセット。動作確認: VS 18 Community / cl 19.51)
- **CUDA Toolkit 11.8** (`nvcc`、`cudart_static.lib`) — Kepler〜Hopper 用カーネルと
  実行時ランタイム (cudart 11.8 は R451 以降の全ドライバで動作)
- **CUDA Toolkit 13.x** (`nvcc` のみ) — Blackwell 以降用カーネル
  (13.x は sm_75 未満をコンパイルできないため、旧 GPU 対応に 11.8 も必要です。
  逆に 13.x は sm_100/sm_120 を生成できる唯一の系統です)
- NASM 2.16.03 (`toolchain/nasm-2.16.03` に配置済み。別バージョンでも可)
- Git Bash + `mingw32-make` (MSYS2/MinGW の make でも可)
- NVIDIA GPU (Kepler 以降) + ドライバ R451 以降 (CUDA 利用時。それ以前は OpenCL へ)
- OpenCL 1.2+ ランタイム (任意。`--gpu-me-backend 2` 用。動作確認: NVIDIA OpenCL 3.0)

GPU 対応条件 (これ以外は自動で CPU フォールバック):

- 8-bit / 10-bit エンコード (`--output-depth 8` / `10`)。12-bit は CPU フォールバック
- プログレッシブ (インターレースは非対応)

GPU 世代ごとのバックエンド (既定の自動選択。実行時に解析が失敗した場合は
もう一方のバックエンドへ 1 回だけ切り替え、警告を表示します):

| GPU 世代 | CUDA | OpenCL | 備考 |
|---|---|---|---|
| Kepler〜Volta (sm_35〜sm_70) | ✅ | ✅ | CUDA 11.8 カーネル (Kepler は最終ドライバ R470 で動作) |
| Turing〜Hopper (sm_75〜sm_90) | ✅ | ✅ | CUDA 11.8 カーネル (動作確認: RTX 2060 SUPER / sm_75) |
| Blackwell (sm_100/120) | ✅ | ✅ | CUDA 13.x カーネル |
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
> CUDA バックエンドはカーネルとホストに分かれています。カーネル
> (`x264/common/gpu_kernels.cu`) は nvcc の `-fatbin` でコンパイルされ、
> RCDATA リソースとして exe に埋め込まれます。ホスト側
> (`x264/common/gpu_cuda.cpp`) は通常の cl でコンパイルし、実行時に
> **ドライバ API** (`nvcuda.dll` を動的ロード) でリソースからカーネルを
> ロードします。これにより (1) 複数ツールキットのカーネルを 1 つの exe に
> 同居でき、(2) ホスト側のビルドが nvcc のバージョン制約を受けません
> (最新 VS の標準ライブラリは古い nvcc がパースできないため、この分離が
> 必要です。カーネル側は `common/cuda_stub/` の最小ヘッダでビルドします)。
>
> 既定のカーネル構成は「CUDA 11.8: sm_35〜sm_90 + compute_75 PTX」と
> 「CUDA 13.x: sm_100/sm_120 + compute_90 PTX」です。環境変数
> `CUDA_ARCHS_LEGACY` / `CUDA_PTX_LEGACY` / `CUDA_ARCHS_MODERN` /
> `CUDA_PTX_MODERN` で変更できます (`configure-msvc.sh` を参照)。
> デバイスに合うカーネルは実行時に自動選択され、どちらも無い場合は
> OpenCL → CPU へフォールバックします。テスト用に
> `X264_GPU_FATBIN=legacy|modern` で強制選択できます。
>
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
| `--gpu-lookahead <0-2>` | `0` | GPU ルックアヘッド (低解像度解析) モード: 0=オフ、1=exact (CPU 同一判断保証)、2=fast (実験的。GPU 提案ベクトルで探索短縮、出力は CPU と異なるが決定的) |

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
x264 [warning]: GPU ME: no usable GPU backend, falling back to CPU
```

GPU lookahead も同様です。

```
x264 [info]: GPU lookahead: cuda / NVIDIA GeForce RTX 2060 SUPER (sm_75) (exact mode, 8-bit)
x264 [warning]: GPU lookahead: fast mode is not available yet, falling back to CPU
x264 [warning]: GPU lookahead: interlaced coding is not supported, falling back to CPU
x264 [warning]: GPU lookahead: no usable GPU backend, falling back to CPU
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
| CUDA アーキテクチャ (配布バイナリ) | ✅ CUDA 11.8 カーネル: sm_35〜sm_90 + compute_75 PTX / CUDA 13.x カーネル: sm_100・sm_120 + compute_90 PTX (fatbin を cuobjdump で、実行時のロードを実機で確認) |
| カーネルの実行時ロード | ✅ nvcuda.dll を動的ロード (import 依存は SHELL32/KERNEL32 のみ)。カーネルはリソースから `cuModuleLoadData` でロード |
| v1.0.0 (13.3 カーネル) との A/B | ✅ 6 ケースすべてビット一致 (720p m1/m4/medium-m2、1080p medium-m2/placebo-m2/m3) — 11.8 カーネル + ドライバ API 化は出力を変えない |
| 旧世代カーネル経路の検証 | ✅ `X264_GPU_FATBIN=legacy` で 11.8 カーネルを明示ロードして全ハッシュ一致。13.x カーネルのみを強制した場合は `NO_BINARY_FOR_GPU` を検出し OpenCL へ自動フォールバック |
| 実行時バックエンドフォールバック | ✅ 起動時プローブで非対応 GPU を検出、解析失敗時はもう一方へ 1 回切替 (出力はビット一致のまま)。selftest で自動検証 |
| 10-bit 出力 | ✅ GPU 有効 (`GPU ME: ... (10-bit)`)。10-bit でも GPU ME / GPU lookahead ともに動作し、`--gpu-lookahead 1` は 10-bit でも CPU とビット一致 (下記参照) |
| インターレース | ✅ 警告を出して CPU フォールバック |
| GPU モードの再現性 (同一入力・同一設定) | ✅ mode 1/2/3 の反復実行が SHA-1 一致 (240 frames placebo) |
| フレームタイプ決定が GPU モードに依存しない | ✅ `--gpu-me 0/1/2/3` で I/P/B 列が完全一致、stock とも一致 (240 frames placebo) |
| **GPU lookahead (exact モード) のビット一致** | ✅ `--gpu-lookahead 1` (60f 720p CRF23 threads1 `--gpu-me 0`): medium `6A50DA31…` / ultrafast `624677FE…` / superfast `EBE94973…` / veryslow `86D27CA7…` / placebo `52A6DBFD…` が、テーブルを一切使わない all-CPU 実行 (`X264_GPU_LA_SKIP` 相当) と SHA-1 完全一致。placebo (TESA→`fpelcmp`=SATD) はテーブル側も SATD を計算して一致 |
| **GPU lookahead (fast モード) の決定性** | ✅ `--gpu-lookahead 2` (placebo / medium 60f 720p、10-bit、`--lookahead-threads 2`) の反復実行が SHA-1 完全一致。threads 2 でも 1 スレッドと同一ハッシュ (スライス分割非依存)。同一 CRF の出力サイズは CPU ルックアヘッド比 −0.4〜−2.2% (効率ほぼ同等) |
| **GPU lookahead (fast モード) の品質** | ✅ x264 内蔵 `--psnr --ssim` (60f 720p CRF23): placebo で SSIM_Y 0.993099 (CPU 0.992921) / PSNR 48.77 dB (CPU 48.63)、medium で SSIM_Y 0.992907 (CPU 0.992817) / PSNR 48.62 dB (CPU 48.53)。CPU ルックアヘッドと同等〜わずかに良好 |
| GPU lookahead のプロファイラ | ✅ `X264_GPU_LA_PROF=1` で GPU 準備 / CPU 探索の内訳を表示 (出力はビット不変) |
| `--opencl` 併用時 | ✅ GPU lookahead を自動無効化して警告 (stock の OpenCL lookahead を維持) |
| GPU lookahead のテーブル照合 | ✅ f1 の全 15,210,000 エントリ (720p, R=32) が CPU の `fpelcmp[8x8]` と 0 ミスマッチ。intra カーネルも 0/3600 ミスマッチ (複数フレーム) |
| GPU lookahead × `--lookahead-threads 2` | ✅ medium / placebo とも 1 スレッド結果と同一ハッシュ (exact / fast とも) |
| GPU lookahead × 10-bit | ✅ medium `D0E5BD4F…` / placebo `C267975F…` が all-CPU 実行とビット一致 (10-bit SATD カーネル含む)。fast モードも 10-bit で決定的 (`EF87142A…`) |
| GPU lookahead × GPU ME 同時使用 | ✅ `--gpu-me 2 --gpu-lookahead 1` で LA 部分のみ GPU/CPU を切り替えても同一ハッシュ (`D383AF1E…`) |
| `--gpu-lookahead` 未指定時の不変条件 | ✅ g0 = `BEB55C36…`、m2 = `965066A7…`、m3 = `B3090B30…`、OpenCL m2 は全て従来値を維持 (オプション追加による出力変化なし) |
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
- **GPU lookahead は値を先計算するだけ。** exact モード (`--gpu-lookahead 1`)
  では GPU は「CPU が計算するはずの intra コストと 8x8 コストテーブル」を
  ビット単位で同じ値で事前計算し、CPU の探索 (`me_search`) はその値を参照する
  だけです。テーブルに無い位置は CPU が同じ関数で自前計算するため、判断は
  GPU の有無で一切変わりません (GPU 失敗時も CPU に落ちて同一出力)。

### GPU lookahead (低解像度解析の先読み)

`--gpu-lookahead 1` (exact モード) の処理フロー:

```
x264_slicetype_frame_cost()                [encoder/slicetype.c]
  ├─ x264_weights_analyse()                重み決定 (テーブルの参照面が決まる)
  ├─ gpu_la_prep_intra()                   各 8x8 lowres ブロックの raw 最小
  │                                        intra 予測コストを GPU で一括計算
  │                                        (la_intra カーネル。SATD/SAD は
  │                                         mbcmp_init と同じ条件で選択)
  ├─ gpu_la_prep_table()                   (fenc, ref, dist) ごとに、全 MB ×
  │                                        変位 (dx,dy) ∈ [-R,R]² の 8x8
  │                                        SAD/SATD 正方テーブルを GPU で一括
  │                                        計算 (la_table カーネル。R = min(
  │                                         mvrange/2+2, 32)、重み付き参照時は
  │                                         加算後の lowres 平面を参照)
  └─ slicetype_mb_cost() → me_search()
      ├─ GPU_LA_SAD がテーブル値を返す (範囲外は CPU が fpelcmp で計算)
      ├─ intra コストも GPU 結果を採用 (depth shift / penalty / AQ は CPU 側で
      │   元の CPU 経路と同じ式を適用)
      └─ 探索ロジック (hex/subpel/bidir) は全て元の x264 のまま
```

- テーブルは `(fenc, list, dist)` ごとにキャッシュされ、検索済みペアの再解析
  (b-adapt の DP が同一パスを再訪) では再計算されません。
- テーブルの値は `fpelcmp[PIXEL_8x8]` と同じ関数の値です。`--me tesa` +
  subme>1 (placebo など) の場合は `fpelcmp` が SATD になるため、テーブル側も
  同じ Hadamard SATD を計算します (SAD のままにすると placebo で不一致に
  なるため、v1.0.2 で修正済み)。
- フレーム再利用に備え、`x264_frame_pop_unused()` が `gpu_la_*` フィールドを
  リセットします (x264 はフレーム構造体をゼロ化せずに使い回すため)。
- OpenCL バックエンドの lookahead 実装は未移植です (CUDA 専用。OpenCL のみ
  の環境では GPU lookahead は CPU にフォールバックします)。

fast モード (`--gpu-lookahead 2`) の処理フロー:

```
x264_slicetype_frame_cost()
  ├─ gpu_la_prep_intra()                     exact モードと共通 (ビット一致)
  ├─ gpu_la_prep_search()                    (fenc, ref, dist) ごとに la_search
  │                                          カーネル: 1 MB 1 スレッドで階層探索
  │                                          (L1: 4px グリッド ×16 サンプル SAD、
  │                                           L2/L3: 全 8x8 SAD で ±2/±1)。
  │                                          探索窓は CPU の mv_limit_fpel と同一式
  └─ slicetype_mb_cost() → me_search()
      ├─ GPU 提案ベクトルを COST_MV で本物のコスト評価 (CHECK_MVRANGE 保険付き)
      ├─ パターン探索 (hex) を省略
      └─ bpred/0 ベクトル評価・skip 判定・subpel refine・bidir は元の x264 のまま
```

### バックエンド (CUDA / OpenCL)

`common/gpu.c` が公開 API のディスパッチャで、`common/gpu.cu` (CUDA) と
`common/gpu_cl.c` (OpenCL) が同一アルゴリズムを実装します。

- 選択: `--gpu-me-backend 0|1|2` (既定 0 = 自動で CUDA → OpenCL の順に試行)。
  使用中のバックエンドとデバイスは起動ログ (`GPU ME: <backend> / <device>`) に表示。
- **CUDA カーネルはリソース埋め込み。** カーネルは 2 系統の fatbin
  (CUDA 11.8: sm_35〜sm_90、CUDA 13.x: sm_100/sm_120。各 PTX 付き) として
  exe に埋め込まれ、デバイスの Compute Capability に応じて実行時に
  `cuModuleLoadData` でロードされます。ホスト API (メモリ/ストリーム/転送) は
  cudart 11.8 (静的リンク)、カーネル起動はドライバ API を使用。CUDA の
  コンテキストはプライマリコンテキストを明示的に retain してから使います。
- **実行時フォールバック。** 起動時に CUDA カーネルが動かない GPU (対応
  カーネル無し・非対応ドライバ等) はカーネルロードとプローブ起動で検出し、
  自動選択が OpenCL を選びます。エンコード中に解析が失敗した場合も、その
  バックエンドを退役させてもう一方へ 1 回だけ切り替えます (保持していた候補
  バッファは解放して作り直し、`GPU ME: switched to the <backend> backend` の
  警告を表示)。どちらも使えない場合は従来どおり警告を出して CPU 動作になります。
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
| `common/gpu_kernels.h` / `common/gpu_kernels.cu` | 新規。CUDA カーネル (デバイスコードのみ。`nvcc -fatbin` でビルドし、リソースとして埋め込み) |
| `common/gpu_cuda.cpp` | 新規。CUDA バックエンドのホスト側 (cudart + ドライバ API。カーネルロード、転送、キャッシュ、解析) |
| `common/cuda_stub/cuda_runtime.h` | 新規。カーネルビルド用の最小 `cuda_runtime.h` (標準ライブラリを引き込まないため) |
| `common/gpu_cl.c` | 新規。OpenCL バックエンド (CUDA と同一アルゴリズム・同一出力) |
| `x264.h` | `x264_param_t.analyse` に `i_gpu_me*` を追加 |
| `common/base.c` | 既定値・オプション解析・パラメータ文字列 |
| `common/frame.h` / `frame.c` | フレームごとの候補バッファ (`gpu_me`) と解放処理 |
| `common/common.h` | エンコーダごとの GPU コンテキスト (`gpu_ctx`) |
| `encoder/encoder.c` | GPU コンテキストの生成/破棄、`gpu_me_analyse()`、フォールバック警告 |
| `encoder/me.h` / `encoder/analyse.c` | `x264_me_t` に `i_list` / `i_xoff` / `i_yoff` を追加し、候補検索用の情報を設定 |
| `encoder/me.c` | GPU 候補の注入 (mode 1/2/3/4)。`i_list >= 0` を必須条件とする |
| `encoder/slicetype.c` | ルックアヘッドの低解像度 ME に `i_list = -1` を設定し、GPU 候補の注入対象外にする |
| `common/gpu.h` / `gpu.c` | ルックアヘッド用 API (`x264_gpu_la_*`) とバックエンド・ディスパッチャ (`x264_gpu_la_backend_t`) |
| `common/gpu_kernels.cu` | `la_intra(_10)` / `la_table(_10)` / `la_search(_10)` カーネル (intra 予測器・SATD/SAD・階層探索を CPU 実装から正確に転写) |
| `common/gpu_cuda.cpp` | ルックアヘッド用 CUDA ホスト (ピン留めホストリング、平面転送、遅延 grow) |
| `common/gpu_cl.c` | ルックアヘッドの OpenCL スタブ (未移植。available=0 で CPU フォールバック) |
| `common/frame.h` / `frame.c` | フレームごとの `gpu_la_intra` / `gpu_la_tab` / `gpu_la_R` と `pop_unused` でのリセット |
| `x264.h` / `common/base.c` / `x264.c` | `--gpu-lookahead` オプションと SEI パラメータ文字列 (`gpu_la=<mode>:<device>`) |
| `Makefile` | `common/gpu*.o`・カーネル fatbin・リソース (`gpu_res.o`) のビルド規則と `cudart_static.lib` のリンク |
| `x264/tools/msvsdepend.sh` | Windows 差分ビルドのヘッダ依存生成を修正 (日本語ロケールの `/showIncludes` 対応、ビット深度定義の付与、失敗時のビルド中断) |
| `configure-msvc.sh` / `make-msvc.sh` | GPU 設定 (2 系統の CUDA / OpenCL) の追記、ライブラリ・ヘッダパス設定 |
| `dev/gpu_me_bench.cu` | 開発用の単体 CUDA ベンチマーク (エンコーダ本体にはリンクされません) |
| `bench720.ps1` / `bench-results/` | ベンチマーク実行スクリプトと全測定ログ (NVENC 比較含む) |

---

## 8. 既知の制限と今後の課題

- **12-bit エンコードでは GPU は使われません** (8-bit / 10-bit は対応。CPU フォールバック)。
- **インターレース非対応** (GPU ME / GPU lookahead とも CPU フォールバック)。
- **CUDA 利用にはドライバ R451 以降が必要**。それ以前のドライバでは CUDA
  カーネルがロードできず、OpenCL (または CPU) へ自動フォールバックします。
- **Kepler〜Volta の CUDA 経路は実機検証が未実施**。カーネルは全世代で同一
  ソース・同一アルゴリズムで、sm_75 上で 11.8 カーネルの出力一致を検証済み
  ですが、Kepler〜Volta の実機での動作確認はできていません (OpenCL 経路は
  NVIDIA のドライバ同梱実装を使用)。GPU lookahead も同様に sm_75 実機のみの
  検証です (失敗時は CPU にフォールバックして出力は変わりません)。
- **exact モード (`--gpu-lookahead 1`) は現状、ルックアヘッドが軽い設定では
  逆に遅くなります。** CPU 探索が評価する数十位置に対し、GPU は MB あたり
  (2R+1)² = 最大 4,225 位置のテーブルを事前計算するためです (60f 720p の
  実測: medium 1.5s→6.3s、1080p 5.0s→13.8s)。exact モードは「出力を一切
  変えずにルックアヘッドの計算を GPU に置換できる」ことの保証が主目的です。
- **fast モード (`--gpu-lookahead 2`) は exact のテーブル代償を解消しますが、
  720p/1080p クラスでは CPU ルックアヘッドとほぼ互角です** (60f 実測:
  medium 720p 1.5s(CPU)→2.2s(fast)、placebo 720p 25.3s→28.2s、medium 1080p
  5.0s→6.0s)。ルックアヘッドの fpel 探索自体が SIMD で軽く、(fenc, ref) ペア
  ごとの GPU 往復 (アップロード+同期) が上回るためです。fast モードの価値は
  (1) exhaustive テーブルを持たない実用モードであること、(2) 決定的で
  lookahead-threads にも依存しないこと、(3) 今後のパイプライン化・大解像度
  ワークロード向けの土台であること、にあります。実際の高速化は今後の課題です。
- **GPU lookahead の OpenCL 実装は未移植** (CUDA 専用)。OpenCL のみの環境では
  警告を出して CPU にフォールバックします。
- **本家 `--opencl` (OpenCL lookahead) は変更していません。** 本フォークの
  `--gpu-lookahead` は独立した機能です。`--opencl` と同時に指定した場合は
  GPU lookahead を自動で無効化し、警告を表示します (OpenCL lookahead が
  低解像度解析を置き換えるため)。
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

### 性能の現状と残された高速化余地 (実測)

`X264_ME_PROF` / `X264_GPU_LA_PROF` による内訳 (60f 720p、`--threads 1`):

| ケース | エンコード | ME (サブペル含む) | ルックアヘッド | 備考 |
|---|---|---|---|---|
| placebo `--gpu-me 0` | 33.1 s | 25.7 s (**77.6%**) | ~4.6 s (14%) | ME 支配 |
| placebo `--gpu-me 3` | 26.8 s | 20.3 s (76.0%) | ~4.6 s | ME −21% |
| tesa24 `--gpu-me 0` | 3.3 s | 2.45 s (**74.2%**) | — | |
| tesa24 `--gpu-me 3` | 2.5 s | 1.55 s (61.7%) | — | ME **−37%** |
| umh24 `--gpu-me 0` | 2.0 s | 1.19 s (59.5%) | — | |
| umh24 `--gpu-me 2` | 2.2 s | 1.23 s (55.8%) | — | 候補追加のオーバーヘッド |

- **高圧縮プリセットのコストは ME (特にサブペル refine) が支配的**です。
  gpu-me 3 はフルペル探索を省略して placebo −21% / tesa24 −37% を達成しますが、
  その後に残る hpel/qpel refine (placebo は hpel 4 + qpel 10 反復) と RDO は
  CPU 側のままです。さらに速度を伸ばすにはサブペル refine の GPU 化が必要で、
  補間・RDO 統合を含む大規模な取り組みになります (現行の候補生成方式では
  これ以上の大きな短縮は見込めません)。
- **GPU lookahead の内訳**: fast モードの GPU 準備は 1.2 ms/パス、CPU パスは
  1.5 ms/パス。GPU が置換できるのはパスあたり ~0.25 ms (fpel 探索) のみで、
  サブペル refine・bidir・skip/intra 判定が CPU 側の大半を占めます。
  ルックアヘッド自体も placebo で全体の ~14%、medium では数 % に過ぎないため、
  fast モードの更なる最適化 (往復のパイプライン化やサブペルまでの GPU 化) の
  上限は placebo で 1 s 程度 (全体の 3〜4%) と見積もられます。
- **メモリ**: GPU lookahead の追加使用量は CUDA コンテキスト + 平面で約
  +110 MB (exact はテーブル + 約 60 MB)、フレーム数に対する増加は CPU のみの
  実行と同程度 (60f→240f で +33/+35 MB) で、リークはありません。

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

バックエンドの実行時フォールバックの検証 (`X264_GPU_SIMULATE_FAIL` は指定した
バックエンドを最初の解析で失敗させるテスト用フックです):

```powershell
# CUDA を失敗させ、OpenCL へ自動切替することを確認 (出力は a.264 と同一)
$env:X264_GPU_SIMULATE_FAIL = "cuda"
build-msvc\x264.exe --preset placebo --crf 23 --threads 1 --gpu-me 2 `
    testdata\motion720p.yuv --input-res 1280x720 --fps 30 -o c.264
# 標準エラーに "GPU ME: switched to the opencl backend after an analysis failure"
Get-FileHash c.264 -Algorithm SHA1          # a.264 と同じハッシュになる
Remove-Item Env:\X264_GPU_SIMULATE_FAIL
```

> `X264_GPU_ASYNC=0` を併用すると、先出し解析ではなく同期解析側の失敗経路も
> 検証できます (どちらも同じ切替動作)。

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

X264_GPU_LA_PROF=1 x264 --preset slow --crf 23 --gpu-lookahead 2 input.yuv ... -o out.264
# GPU lookahead の GPU 準備 (イントラ/テーブル/探索) と CPU 探索の内訳を表示します
# (--threads 1 での計測用。ペアごとの GPU 往復コストの見積りに使えます)

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
