# Acceptance self-test for hybrid-x264-gpu.
# Verifies that:
#   1) --gpu-me 0 output is bit-identical to the stock x264 baseline
#   2) GPU modes run and exit cleanly
#   3) GPU mode output is decodable (if ffmpeg is available)
#   4) 10-bit encoding is GPU-accelerated (CPU fallback without a usable GPU)
#      and --gpu-lookahead (exact mode / fast-mode warning) works
#   5) repeated GPU-mode runs are deterministic (identical SHA-1)
#   6) GPU modes do not change frame-type decisions vs mode 0
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File selftest.ps1 [-Ffmpeg <path>]
#   ffmpeg/ffprobe are located on PATH (or passed with -Ffmpeg); the decode and
#   frame-type checks are skipped when they are unavailable.
param( [string]$Ffmpeg = "" )
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
New-Item -ItemType Directory -Force -Path testdata | Out-Null

$exe  = Join-Path $root "build-msvc\x264.exe"
$base = Join-Path $root "x264-baseline.exe"
$yuv  = Join-Path $root "testdata\motion720p.yuv"

$ff = $null
if( $Ffmpeg )
{
    if( Test-Path $Ffmpeg ) { $ff = $Ffmpeg }
}
else
{
    $c = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if( $c ) { $ff = $c.Source }
}

if( !(Test-Path $exe) ) { Write-Host "FAIL: $exe not found (build first)" -ForegroundColor Red; exit 1 }
if( !(Test-Path $yuv) ) { Write-Host "SKIP: $yuv not found (benchmark clip unavailable)" -ForegroundColor Yellow; exit 0 }

$common = @("--crf","23","--threads","1","--frames","30","--input-res","1280x720","--fps","30")
$fail = 0

# 1) bit-exactness of mode 0 against stock x264
if( Test-Path $base )
{
    & $base "--preset" "medium" @common $yuv "-o" "testdata\st_base.264" 2> $null
    & $exe  "--preset" "medium" "--gpu-me" "0" @common $yuv "-o" "testdata\st_g0.264" 2> $null
    $h1 = (Get-FileHash "testdata\st_base.264" -Algorithm SHA1).Hash
    $h2 = (Get-FileHash "testdata\st_g0.264"   -Algorithm SHA1).Hash
    if( $h1 -eq $h2 ) { Write-Host "PASS: --gpu-me 0 is bit-identical to stock x264" -ForegroundColor Green }
    else              { Write-Host "FAIL: mode 0 mismatch ($h1 vs $h2)" -ForegroundColor Red; $fail = 1 }
}
else { Write-Host "SKIP: x264-baseline.exe not found, cannot verify bit-exactness" -ForegroundColor Yellow }

# 2) GPU modes run and exit cleanly
foreach( $m in 1,2,3,4 )
{
    & $exe "--preset" "slow" "--gpu-me" $m @common $yuv "-o" "testdata\st_g$m.264" 2> $null
    if( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: --gpu-me $m encode exited cleanly" -ForegroundColor Green }
    else                      { Write-Host "FAIL: --gpu-me $m exit code $LASTEXITCODE" -ForegroundColor Red; $fail = 1 }

    if( $ff )
    {
        & $ff "-v" "error" "-i" "testdata\st_g$m.264" "-f" "null" "NUL" 2> $null
        if( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: --gpu-me $m output decodes" -ForegroundColor Green }
        else                      { Write-Host "FAIL: --gpu-me $m output failed to decode" -ForegroundColor Red; $fail = 1 }
    }
}

# 2b) CUDA analysis is actually engaged (informational: CPU fallback is valid
#     on machines without a compatible GPU/driver)
& $exe "--preset" "slow" "--gpu-me" "2" @common $yuv "-o" "NUL" 2> "testdata\st_gpu.log"
if( (Get-Content "testdata\st_gpu.log" -Raw) -match "GPU ME: " ) { Write-Host "PASS: GPU analysis active" -ForegroundColor Green }
else { Write-Host "WARN: CUDA not active (CPU fallback)" -ForegroundColor Yellow }

# 3) 10-bit encoding is GPU-accelerated (bit depth shown in the log); on
#    machines without a usable GPU it degrades to the CPU fallback warning
& $exe "--preset" "medium" "--gpu-me" "2" "--output-depth" "10" @common $yuv "-o" "NUL" 2> "testdata\st_10bit.log"
$log = Get-Content "testdata\st_10bit.log" -Raw
if( $log -match "GPU ME: \S+ / .+\(10-bit\)" ) { Write-Host "PASS: 10-bit encoding is GPU-accelerated" -ForegroundColor Green }
elseif( $log -match "GPU ME: no usable GPU backend" ) { Write-Host "WARN: no usable GPU backend, 10-bit ran on the CPU" -ForegroundColor Yellow }
else { Write-Host "FAIL: 10-bit GPU acceleration missing" -ForegroundColor Red; $fail = 1 }

# 3b) GPU lookahead (exact mode): runs cleanly, reports itself, and the
#     reserved fast mode falls back to the CPU with a warning
& $exe "--preset" "medium" "--gpu-me" "0" "--gpu-lookahead" "1" @common $yuv "-o" "testdata\st_la.264" 2> "testdata\st_la.log"
$laLog = Get-Content "testdata\st_la.log" -Raw
if( $LASTEXITCODE -eq 0 -and $laLog -match "GPU lookahead: \S+ / .+exact mode" )
    { Write-Host "PASS: --gpu-lookahead 1 (exact mode) runs" -ForegroundColor Green }
else { Write-Host "FAIL: --gpu-lookahead 1 failed (exit $LASTEXITCODE)" -ForegroundColor Red; $fail = 1 }
if( $ff )
{
    & $ff "-v" "error" "-i" "testdata\st_la.264" "-f" "null" "NUL" 2> $null
    if( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: --gpu-lookahead 1 output decodes" -ForegroundColor Green }
    else { Write-Host "FAIL: --gpu-lookahead 1 output failed to decode" -ForegroundColor Red; $fail = 1 }
}
& $exe "--preset" "medium" "--gpu-me" "0" "--gpu-lookahead" "2" @common $yuv "-o" "NUL" 2> "testdata\st_la2.log"
$la2Log = Get-Content "testdata\st_la2.log" -Raw
if( $la2Log -match "GPU lookahead: fast mode is not available yet" )
    { Write-Host "PASS: --gpu-lookahead 2 warns and falls back to the CPU" -ForegroundColor Green }
else { Write-Host "FAIL: fast-mode fallback warning missing" -ForegroundColor Red; $fail = 1 }

# 4) determinism: repeated runs of the same GPU mode must be bit-identical
$detCommon = @("--preset","slow","--crf","23","--threads","1","--frames","80","--input-res","1280x720","--fps","30")
foreach( $m in 1,2,3,4 )
{
    & $exe "--gpu-me" $m @detCommon $yuv "-o" "testdata\st_d${m}a.264" 2> $null
    & $exe "--gpu-me" $m @detCommon $yuv "-o" "testdata\st_d${m}b.264" 2> $null
    $ha = (Get-FileHash "testdata\st_d${m}a.264" -Algorithm SHA1).Hash
    $hb = (Get-FileHash "testdata\st_d${m}b.264" -Algorithm SHA1).Hash
    if( $ha -eq $hb ) { Write-Host "PASS: --gpu-me $m is deterministic across runs" -ForegroundColor Green }
    else              { Write-Host "FAIL: --gpu-me $m differs between runs ($ha vs $hb)" -ForegroundColor Red; $fail = 1 }
}

# 5) GPU candidate injection must not change the lookahead's frame-type
#    decisions: the frame type sequence has to match mode 0 exactly
$ffprobe = $null
if( $ff )
{
    $cand = Join-Path (Split-Path $ff) "ffprobe.exe"
    if( Test-Path $cand ) { $ffprobe = $cand }
}
if( $ffprobe )
{
    & $exe "--gpu-me" "0" @detCommon $yuv "-o" "testdata\st_ft0.264" 2> $null
    $types0 = (& $ffprobe "-v" "error" "-select_streams" "v" "-show_entries" "frame=pict_type" "-of" "csv=p=0" "testdata\st_ft0.264") -join ''
    foreach( $m in 1,2,3,4 )
    {
        $types = (& $ffprobe "-v" "error" "-select_streams" "v" "-show_entries" "frame=pict_type" "-of" "csv=p=0" "testdata\st_d${m}a.264") -join ''
        if( $types -eq $types0 ) { Write-Host "PASS: --gpu-me $m frame types match mode 0" -ForegroundColor Green }
        else                     { Write-Host "FAIL: --gpu-me $m frame types differ from mode 0" -ForegroundColor Red; $fail = 1 }
    }
}
else { Write-Host "SKIP: ffprobe not found, cannot verify frame-type decisions" -ForegroundColor Yellow }

# 6) non-16-multiple resolution (1080p) uses the GPU when the clip is present
$yuv1080 = Join-Path $root "testdata\test1080p.yuv"
if( Test-Path $yuv1080 )
{
    $common1080 = @("--crf","23","--threads","1","--frames","30","--input-res","1920x1080","--fps","30")
    & $exe "--preset" "medium" "--gpu-me" "2" @common1080 $yuv1080 "-o" "testdata\st_1080.264" 2> "testdata\st_1080.log"
    $log1080 = Get-Content "testdata\st_1080.log" -Raw
    if( ($LASTEXITCODE -eq 0) -and ($log1080 -match "GPU ME: ") )
        { Write-Host "PASS: 1080p (non-16-multiple) uses the GPU" -ForegroundColor Green }
    else
        { Write-Host "FAIL: 1080p GPU run failed (exit $LASTEXITCODE)" -ForegroundColor Red; $fail = 1 }

    if( $ff )
    {
        & $ff "-v" "error" "-i" "testdata\st_1080.264" "-f" "null" "NUL" 2> $null
        if( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: 1080p output decodes" -ForegroundColor Green }
        else { Write-Host "FAIL: 1080p output failed to decode" -ForegroundColor Red; $fail = 1 }
    }
}
else { Write-Host "SKIP: test1080p.yuv not found, cannot verify 1080p GPU path" -ForegroundColor Yellow }

# 7) compatibility smoke: 2-pass stats, qpfile, zones, multi-threaded encode
$smoke = @("--preset","medium","--frames","30","--input-res","1280x720","--fps","30")
$compatFail = @()
# 2-pass encode with a stats file
& $exe "--pass" "1" "--stats" "testdata\st_stats.log" "--bitrate" "600" @smoke "--gpu-me" "2" $yuv "-o" "NUL" 2> $null
$p1 = $LASTEXITCODE
& $exe "--pass" "2" "--stats" "testdata\st_stats.log" "--bitrate" "600" @smoke "--gpu-me" "2" $yuv "-o" "testdata\st_2pass.264" 2> $null
$p2 = $LASTEXITCODE
if( $p1 -eq 0 -and $p2 -eq 0 ) { Write-Host "PASS: 2-pass encode with stats file" -ForegroundColor Green }
else { Write-Host "FAIL: 2-pass encode ($p1/$p2)" -ForegroundColor Red; $fail = 1 }
# qpfile: force an I frame at display index 25
Set-Content -Path "testdata\st.qpfile" -Value "25 I"
& $exe "--qpfile" "testdata\st.qpfile" @smoke "--gpu-me" "2" $yuv "-o" "testdata\st_qp.264" 2> $null
if( $LASTEXITCODE -eq 0 -and $ffprobe )
{
    $types = @(& $ffprobe "-v" "error" "-select_streams" "v" "-show_entries" "frame=pict_type" "-of" "csv=p=0" "testdata\st_qp.264")
    if( $types.Count -gt 25 -and $types[25] -eq "I" ) { Write-Host "PASS: qpfile forced an I frame at index 25" -ForegroundColor Green }
    else { Write-Host "FAIL: qpfile frame type at index 25 is $($types[25])" -ForegroundColor Red; $fail = 1 }
}
elseif( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: qpfile encode exited cleanly (no ffprobe to verify)" -ForegroundColor Green }
else { Write-Host "FAIL: qpfile encode exit $LASTEXITCODE" -ForegroundColor Red; $fail = 1 }
# zones
& $exe "--zones" "10,20,b=1.5" @smoke "--gpu-me" "2" $yuv "-o" "testdata\st_zone.264" 2> $null
if( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: zones encode exited cleanly" -ForegroundColor Green }
else { Write-Host "FAIL: zones encode exit $LASTEXITCODE" -ForegroundColor Red; $fail = 1 }
# frame threads + sliced threads
& $exe "--threads" "4" "--sliced-threads" @smoke "--gpu-me" "2" $yuv "-o" "testdata\st_t4s.264" 2> $null
if( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: --threads 4 --sliced-threads with GPU mode" -ForegroundColor Green }
else { Write-Host "FAIL: threaded encode exit $LASTEXITCODE" -ForegroundColor Red; $fail = 1 }

# 8) OpenCL backend (when available) produces the same stream as CUDA
& $exe "--preset" "slow" "--gpu-me" "2" @common $yuv "-o" "testdata\st_ocl_a.264" 2> "testdata\st_ocl_a.log"
& $exe "--preset" "slow" "--gpu-me" "2" "--gpu-me-backend" "2" @common $yuv "-o" "testdata\st_ocl_b.264" 2> "testdata\st_ocl_b.log"
if( (Get-Content "testdata\st_ocl_b.log" -Raw) -match "GPU ME: opencl" )
{
    $ha = (Get-FileHash "testdata\st_ocl_a.264" -Algorithm SHA1).Hash
    $hb = (Get-FileHash "testdata\st_ocl_b.264" -Algorithm SHA1).Hash
    if( $ha -eq $hb ) { Write-Host "PASS: OpenCL backend output is bit-identical to CUDA" -ForegroundColor Green }
    else { Write-Host "FAIL: OpenCL output differs from CUDA ($ha vs $hb)" -ForegroundColor Red; $fail = 1 }
}
else { Write-Host "SKIP: OpenCL backend not available" -ForegroundColor Yellow }

# 9) a failing backend switches to the other one with bit-identical output
$env:X264_GPU_SIMULATE_FAIL = "cuda"
& $exe "--preset" "slow" "--gpu-me" "2" @common $yuv "-o" "testdata\st_fb.264" 2> "testdata\st_fb.log"
$fbLog = Get-Content "testdata\st_fb.log" -Raw
Remove-Item Env:\X264_GPU_SIMULATE_FAIL
if( $fbLog -match "GPU ME: cuda" )
{
    if( $fbLog -match "GPU ME: switched to the opencl backend after an analysis failure" )
    {
        $hf = (Get-FileHash "testdata\st_fb.264"  -Algorithm SHA1).Hash
        $hc = (Get-FileHash "testdata\st_ocl_a.264" -Algorithm SHA1).Hash
        if( $hf -eq $hc ) { Write-Host "PASS: CUDA failure falls back to OpenCL (bit-identical)" -ForegroundColor Green }
        else { Write-Host "FAIL: fallback output differs ($hf vs $hc)" -ForegroundColor Red; $fail = 1 }
    }
    else { Write-Host "FAIL: CUDA failure did not switch to the OpenCL backend" -ForegroundColor Red; $fail = 1 }
}
else { Write-Host "SKIP: CUDA backend is not the initial backend, cannot verify fallback" -ForegroundColor Yellow }

Remove-Item testdata\st_*.264,testdata\st_10bit.log,testdata\st_gpu.log,testdata\st_la.log,testdata\st_la2.log,testdata\st_1080.log,testdata\st_stats.log,testdata\st_stats.log.mbtree,testdata\st.qpfile,testdata\st_ocl_a.log,testdata\st_ocl_b.log,testdata\st_fb.log -ErrorAction SilentlyContinue

if( $fail ) { Write-Host "SELFTEST FAILED" -ForegroundColor Red; exit 1 }
Write-Host "ALL TESTS PASSED" -ForegroundColor Green
exit 0
