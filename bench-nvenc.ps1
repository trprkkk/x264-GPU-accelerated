# NVENC quality comparison for hybrid-x264-gpu.
# Encodes the hybrid encoder (placebo + --gpu-me 2) at the given CRF points and
# NVENC (h264_nvenc, best-quality settings) at matched bitrates, then measures
# SSIM/PSNR with ffmpeg's filters.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File bench-nvenc.ps1 [-Ffmpeg <path>] [-Frames 240] [-CrfList "23,20,16"]
param(
    [string]$Ffmpeg = "ffmpeg",
    [int]$Frames = 240,
    [string]$CrfList = "23,20,16"
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$exe = "build-msvc\x264.exe"
$yuv = "testdata\motion720p.yuv"
$out = "bench-results\nvenc-comparison"
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Get-Metrics($file) {
    $dec = "$out\_tmp.dec.yuv"
    & $Ffmpeg -hide_banner -loglevel error -i $file -pix_fmt yuv420p -f rawvideo -y $dec
    # Limit the comparison to the encoded frame count: without this, ffmpeg
    # would repeat the last decoded frame to match the longer source input.
    $s = (& $Ffmpeg -hide_banner -f rawvideo -pix_fmt yuv420p -s 1280x720 -r 30 -i $yuv `
          -f rawvideo -pix_fmt yuv420p -s 1280x720 -r 30 -i $dec -lavfi ssim -frames:v $Frames -f null NUL 2>&1 | Out-String)
    $p = (& $Ffmpeg -hide_banner -f rawvideo -pix_fmt yuv420p -s 1280x720 -r 30 -i $yuv `
          -f rawvideo -pix_fmt yuv420p -s 1280x720 -r 30 -i $dec -lavfi psnr -frames:v $Frames -f null NUL 2>&1 | Out-String)
    Remove-Item $dec -ErrorAction SilentlyContinue
    [pscustomobject]@{
        ssim_y   = [regex]::Match($s,"SSIM Y:([\d\.]+)").Groups[1].Value
        psnr_y   = [regex]::Match($p,"PSNR y:([\d\.]+)").Groups[1].Value
        psnr_min = [regex]::Match($p,"min:([\d\.]+)").Groups[1].Value
    }
}

$rows = @()
foreach( $crf in ($CrfList -split ',' | ForEach-Object { [int]$_.Trim() }) )
{
    # hybrid: placebo + GPU candidates
    $hf = "$out\repro_hybrid_c$crf.264"
    & $exe "--preset" "placebo" "--gpu-me" "2" "--crf" $crf "--threads" "1" "--frames" $Frames `
          $yuv "--input-res" "1280x720" "--fps" "30" "-o" $hf 2> "$out\repro_hybrid_c$crf.log"
    if( $LASTEXITCODE -ne 0 ) { Write-Host "hybrid encode failed (crf $crf)" -ForegroundColor Red; exit 1 }
    $hkbs = [math]::Round((Get-Item $hf).Length * 8 / ($Frames / 30.0) / 1000, 1)
    $hm = Get-Metrics $hf

    # NVENC at the same bitrate with its best quality settings
    $nf = "$out\repro_nvenc_c$crf.264"
    & $Ffmpeg -hide_banner -loglevel error -f rawvideo -pix_fmt yuv420p -s 1280x720 -r 30 -i $yuv `
        -frames:v $Frames -c:v h264_nvenc -preset p7 -tune hq -rc cbr `
        -b:v ("$($hkbs)k") -maxrate ("$($hkbs)k") -bufsize ("$([int]($hkbs*2))k") `
        -multipass fullres -spatial-aq 1 -temporal-aq 1 -aq-strength 8 -bf 3 `
        -b_ref_mode middle -rc-lookahead 32 -profile:v high -g 250 -f h264 -y $nf
    if( $LASTEXITCODE -ne 0 ) { Write-Host "NVENC unavailable, stopping comparison" -ForegroundColor Yellow; break }
    $nkbs = [math]::Round((Get-Item $nf).Length * 8 / ($Frames / 30.0) / 1000, 1)
    $nm = Get-Metrics $nf

    $rows += [pscustomobject]@{
        crf      = $crf
        h_kbps   = $hkbs; h_ssim = $hm.ssim_y; h_psnr = $hm.psnr_y; h_min = $hm.psnr_min
        n_kbps   = $nkbs; n_ssim = $nm.ssim_y; n_psnr = $nm.psnr_y; n_min = $nm.psnr_min
    }
}

$rows | Format-Table -AutoSize
$rows | Export-Csv "$out\repro_summary.csv" -NoTypeInformation -Encoding UTF8
