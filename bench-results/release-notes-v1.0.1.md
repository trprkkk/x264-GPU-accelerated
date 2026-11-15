CPU/GPU ハイブリッド H.264 エンコーダ (x264 フォーク) の Windows x64 バイナリです。

**v1.0.1 の主な変更: Kepler〜Volta (旧 GeForce) でも CUDA が使えるようになりました。**

## 特徴

- GPU は固定機能 (NVENC/QSV) ではなく **汎用 CUDA / OpenCL カーネル**で整数精度の
  動き探索候補を生成し、最終判断は CPU 側の x264 コードが行います
- **CUDA と OpenCL の両バックエンドを同梱** (既定は自動選択、`--gpu-me-backend` で指定可)。
  両バックエンドは**完全に同一のビットストリームを生成**します (ビット単位で検証済み)
- `--gpu-me 0` (既定) は stock x264 とビット一致 — 従来のワークフローにそのまま使えます

## GPU 対応 (このバイナリ)

| GPU 世代 | CUDA | OpenCL | 備考 |
|---|---|---|---|
| **Kepler〜Volta (sm_35〜sm_70)** | ✅ | ✅ | **CUDA 11.8 カーネル (新)**。Kepler は最終ドライバ R470 で動作 |
| Turing〜Hopper (sm_75〜sm_90) | ✅ | ✅ | CUDA 11.8 カーネル |
| Blackwell (sm_100/120) | ✅ | ✅ | CUDA 13.x カーネル |
| それ以降の世代 | ✅ | ✅ | compute_75 PTX を JIT |

- CUDA 利用にはドライバ **R451 以降**が必要です (それ以前は OpenCL へ自動フォールバック)
- カーネルは 2 系統 (CUDA 11.8 / 13.x) とも exe に埋め込まれ、実行時に
  ドライバ API で自動選択されます。対応カーネルが無い場合は OpenCL →
  CPU へフォールバックし、エンコード中の解析失敗時ももう一方のバックエンドへ
  1 回だけ自動切替します (切替後も出力は同一のビットストリーム)

## 動作要件

- Windows x64 — 単体で動作します (同梱 DLL 不要。依存は Windows 標準の SHELL32 / KERNEL32 のみ)
- NVIDIA ドライバ (CUDA バックエンド) と/または OpenCL ドライバ (OpenCL バックエンド) を使用
- CUDA Toolkit / VC++ ランタイムのインストールは不要 (cudart は静的リンク、nvcuda.dll は動的ロード)

## 使い方 (例)

```
x264.exe --me tesa --merange 24 --crf 23 --gpu-me 2 input.yuv --input-res 1920x1080 --fps 30 -o out.264
```

- 8-bit・プログレッシブで GPU が有効になります (10-bit / インターレースは CPU フォールバック)
- `--me esa` / `--me tesa` / `--preset placebo` で **CPU のみ比 +17〜47%** (品質同等)
- mode 2 = 推奨 (速度+品質バランス) / mode 3 = 最速 / mode 4 = mode 2 と 3 の中間

## このバイナリの検証 (実測値)

- `--gpu-me 0`: 本家 x264 とビット一致 (720p / 1080p)
- **v1.0.0 (CUDA 13.3 カーネル) との A/B: 6 ケースすべてビット一致** (720p m1/m4/medium m2、
  1080p medium m2 / placebo m2/m3) — CUDA 11.8 カーネル + ドライバ API 化による出力変化なし
- CUDA 11.8 カーネル / CUDA 13.x カーネル / OpenCL の三者が同一ハッシュ (720p mode 2/3)
- カーネル非対応を検出した場合の OpenCL フォールバックを実機で検証
- バックエンド切替 (4 方向、CUDA⇔OpenCL) の出力が切替なしとビット一致
- 品質ゲート PASS (3クリップ × BD-rate: 平均 PSNR +0.23% / SSIM +0.28%、mode 2 は最悪 +0.37%)
- NVENC (p7 最高品質設定) 比: 5 レート帯すべてで少ないビット・高 SSIM/PSNR
  (BD-rate −33.6% (SSIM) / −40.7% (PSNR))
- selftest 27/27 合格 / 決定性バッテリー合格 / 速度は v1.0.0 と同等 (誤差内)

## ファイル

| ファイル | サイズ | SHA256 |
|---|---|---|
| `x264.exe` | 3,167,744 bytes | `3DF63A7B8851C7D826585D1BAF104A894500701296752A3F8FCDC49631C340BC` |

## ライセンス

GPL v2 以降 (x264 本体に準拠)。詳細はリポジトリの `COPYING` / `x264/COPYING` を参照してください。
