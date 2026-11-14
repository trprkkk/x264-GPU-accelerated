CPU/GPU ハイブリッド H.264 エンコーダ (x264 フォーク) の Windows x64 バイナリです。

## 特徴

- GPU は固定機能 (NVENC/QSV) ではなく **汎用 CUDA / OpenCL カーネル**で整数精度の
  動き探索候補を生成し、最終判断は CPU 側の x264 コードが行います
- **CUDA と OpenCL の両バックエンドを同梱** (既定は自動選択、`--gpu-me-backend` で指定可)。
  両バックエンドは**完全に同一のビットストリームを生成**します (ビット単位で検証済み)
- `--gpu-me 0` (既定) は stock x264 とビット一致 — 従来のワークフローにそのまま使えます

## GPU 対応 (このバイナリ)

| GPU | バックエンド |
|---|---|
| Kepler〜Volta (sm_35〜sm_70) | OpenCL (CUDA Toolkit 13 が非対応のため) |
| Turing (sm_75) | CUDA (SASS) / OpenCL |
| Ampere (sm_80/86/87) | CUDA (SASS) / OpenCL |
| Ada (sm_89) | CUDA (SASS) / OpenCL |
| Hopper (sm_90) | CUDA (SASS) / OpenCL |
| Blackwell (sm_100/120) | CUDA (SASS) / OpenCL |
| それ以降の世代 | CUDA (compute_75 PTX を JIT) / OpenCL |

- 既定の自動選択は CUDA → OpenCL の順に試行します。非対応 GPU は起動時の
  プローブで検出し、実行中に解析が失敗した場合ももう一方へ **1 回だけ自動切替**
  します (切替後も出力は同一のビットストリーム)。どちらも使えない環境では
  警告を出して CPU エンコードへフォールバックします

## 動作要件

- Windows x64 — 単体で動作します (同梱 DLL 不要。依存は Windows 標準の SHELL32 / KERNEL32 のみ)
- NVIDIA ドライバ (CUDA バックエンド) と/または OpenCL ドライバ (OpenCL バックエンド) を使用
- CUDA Toolkit / VC++ ランタイムのインストールは不要 (cudart は静的リンク、OpenCL.dll は動的ロード)

## 使い方 (例)

```
x264.exe --me tesa --merange 24 --crf 23 --gpu-me 2 input.yuv --input-res 1920x1080 --fps 30 -o out.264
```

- 8-bit・プログレッシブで GPU が有効になります (10-bit / インターレースは CPU フォールバック)
- `--me esa` / `--me tesa` / `--preset placebo` で **CPU のみ比 +17〜47%** (品質同等)
- mode 2 = 推奨 (速度+品質バランス) / mode 3 = 最速 / mode 4 = mode 2 と 3 の中間

## このバイナリの検証 (実測値)

- `--gpu-me 0`: 本家 x264 とビット一致 (720p / 1080p)
- 旧リリース (sm_75 のみ) との A/B: 6 ケースすべてビット一致 (720p m1/m4/medium m2、1080p medium m2 / placebo m2/m3)
- CUDA と OpenCL: 同一設定でビット一致 (mode 2/3、720p/1080p)
- 実行時フォールバック: CUDA 失敗→OpenCL / OpenCL 失敗→CUDA の各方向で
  切替ログを確認し、出力は切替なしとビット一致
- 品質ゲート PASS (3クリップ × BD-rate: 平均 PSNR +0.23% / SSIM +0.28%、mode 2 は最悪 +0.37%)
- NVENC (p7 最高品質設定) 比: 5 レート帯すべてで少ないビット・高 SSIM/PSNR
  (BD-rate −33.6% (SSIM) / −40.7% (PSNR))
- selftest 27/27 合格 (ビット一致・決定性・フレームタイプ不変・1080p・2パス/qpfile/zones/スレッド・OpenCL 一致・フォールバック)

## ファイル

| ファイル | サイズ | SHA256 |
|---|---|---|
| `x264.exe` | 2,828,288 bytes | `A737F91D5CE9F43CB9C3F7BF0612E41C578299294E89D54281829D67079EBB67` |

## ライセンス

GPL v2 以降 (x264 本体に準拠)。詳細はリポジトリの `COPYING` / `x264/COPYING` を参照してください。
