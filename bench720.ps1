# 720p motion search benchmark for the hybrid-x264-gpu fork.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File bench720.ps1
# Results are written to testdata\*.264 + testdata\*.log and bench-results\*.log.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$exe = "build-msvc\x264.exe"
$yuv = "testdata\motion720p.yuv"
New-Item -ItemType Directory -Force -Path "bench-results" | Out-Null

function Run-One($tag, $opts) {
    $out = "testdata\$tag.264"
    $log = "testdata\$tag.log"
    cmd /c "`"$exe`" $opts --input-res 1280x720 --fps 30 --psnr --ssim --threads 1 `"$yuv`" -o `"$out`" 1> `"$log`" 2>&1" | Out-Null
    $text = Get-Content $log -Raw
    $enc  = [regex]::Match($text, "encoded (\d+) frames, ([\d\.]+) fps, ([\d\.]+) kb/s")
    $ssim = [regex]::Match($text, "SSIM Mean Y:([\d\.]+)")
    $psnr = [regex]::Match($text, "PSNR Mean Y:([\d\.]+).*Global:([\d\.]+)")
    $line = "{0,-14} fps={1,7}  kb/s={2,8}  ssim={3}  psnr={4}  size={5}" -f `
        $tag, $enc.Groups[2].Value, $enc.Groups[3].Value, $ssim.Groups[1].Value,
        $psnr.Groups[2].Value, (Get-Item $out).Length
    $line
    Copy-Item $log "bench-results\$tag.log" -Force
}

"=== medium (subme 7, hex) ==="
Run-One "f_med_g0" "--preset medium --crf 23 --gpu-me 0"
Run-One "f_med_g1" "--preset medium --crf 23 --gpu-me 1"
Run-One "f_med_g2" "--preset medium --crf 23 --gpu-me 2"
Run-One "f_med_g3" "--preset medium --crf 23 --gpu-me 3"
"=== slow (subme 8, umh) ==="
Run-One "f_slow_g0" "--preset slow --crf 23 --gpu-me 0"
Run-One "f_slow_g1" "--preset slow --crf 23 --gpu-me 1"
Run-One "f_slow_g2" "--preset slow --crf 23 --gpu-me 2"
Run-One "f_slow_g3" "--preset slow --crf 23 --gpu-me 3"
"=== esa 24 ==="
Run-One "f_esa_g0" "--me esa --merange 24 --crf 23 --gpu-me 0"
Run-One "f_esa_g2" "--me esa --merange 24 --crf 23 --gpu-me 2"
Run-One "f_esa_g3" "--me esa --merange 24 --crf 23 --gpu-me 3"
"=== tesa 24 ==="
Run-One "f_tesa_g0" "--me tesa --merange 24 --crf 23 --gpu-me 0"
Run-One "f_tesa_g2" "--me tesa --merange 24 --crf 23 --gpu-me 2"
Run-One "f_tesa_g3" "--me tesa --merange 24 --crf 23 --gpu-me 3"
