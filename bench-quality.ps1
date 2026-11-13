# Quality-gate benchmark for hybrid-x264-gpu.
# Encodes every (clip, config, GPU mode, CRF) combination, decodes the stream
# and measures SSIM / PSNR / VMAF (ffmpeg's libvmaf) against the source, then
# writes one CSV row per run for tools/quality-gate.py.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File bench-quality.ps1
#       [-Ffmpeg <path>] [-NoVmaf] [-ClipFilter <substring>]
#       [-Configs "esa,tesa"] [-Modes "0,2,3"] [-CrfList "20,23,26"]
#       [-RadiusList "2"] [-OutDir <dir>]
#   -RadiusList adds `--gpu-me-radius N` variants (N != 2); they are tagged
#   `_rN` and grouped separately by tools/quality-gate.py.
#
# Example (fast smoke run):
#   ... -File bench-quality.ps1 -Configs esa -Modes 0,2 -CrfList 23 -ClipFilter motion720p
param(
    [string]$Ffmpeg = "ffmpeg",
    [switch]$NoVmaf,
    [string]$ClipFilter = "",
    [string]$Configs = "esa,tesa",
    [string]$Modes = "0,2,3",
    [string]$CrfList = "20,23,26",
    [string]$RadiusList = "2",
    [string]$OutDir = "bench-results\quality"
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$exe = "build-msvc\x264.exe"
$outDir = $OutDir
$tmpDir = Join-Path $outDir "tmp"
New-Item -ItemType Directory -Force -Path $outDir, $tmpDir | Out-Null
$csv = Join-Path $outDir "quality_summary.csv"

$clipTable = @(
    @{ id = "motion720p"; file = "testdata\motion720p.yuv"; w = 1280; h = 720;  n = 240 },
    @{ id = "test720p";   file = "testdata\test720p.yuv";   w = 1280; h = 720;  n = 120 },
    @{ id = "test1080p";  file = "testdata\test1080p.yuv";  w = 1920; h = 1080; n = 60 }
)
$cfgTable = @(
    @{ id = "esa";     opts = @("--me", "esa", "--merange", "24") },
    @{ id = "tesa";    opts = @("--me", "tesa", "--merange", "24") },
    @{ id = "placebo"; opts = @("--preset", "placebo") }
)

$selClips = @($clipTable | Where-Object { $ClipFilter -eq "" -or $_.id -like "*$ClipFilter*" })
$selConfigs = @($cfgTable | Where-Object { ($Configs -split ",") -contains $_.id })
if( $selClips.Count -eq 0 -or $selConfigs.Count -eq 0 )
{
    Write-Host "no clips/configs selected" -ForegroundColor Red
    exit 1
}

function Get-Metrics($src, $dec, $w, $h, $n, $bVmaf)
{
    # NOTE: no -loglevel here: the ssim/psnr/libvmaf summaries are printed at
    # info level and would be hidden.
    $size = "{0}x{1}" -f $w, $h
    $s = (& $Ffmpeg -hide_banner -f rawvideo -pix_fmt yuv420p -s $size -r 30 -i $src `
          -f rawvideo -pix_fmt yuv420p -s $size -r 30 -i $dec -lavfi ssim -frames:v $n -f null NUL 2>&1 | Out-String)
    $p = (& $Ffmpeg -hide_banner -f rawvideo -pix_fmt yuv420p -s $size -r 30 -i $src `
          -f rawvideo -pix_fmt yuv420p -s $size -r 30 -i $dec -lavfi psnr -frames:v $n -f null NUL 2>&1 | Out-String)
    $v = ""
    if( $bVmaf )
    {
        $vm = (& $Ffmpeg -hide_banner -f rawvideo -pix_fmt yuv420p -s $size -r 30 -i $src `
               -f rawvideo -pix_fmt yuv420p -s $size -r 30 -i $dec -lavfi libvmaf -frames:v $n -f null NUL 2>&1 | Out-String)
        $v = [regex]::Match($vm, "VMAF score:\s*([\d\.]+)").Groups[1].Value
    }
    [pscustomobject]@{
        ssim    = [regex]::Match($s, "SSIM Y:([\d\.]+)").Groups[1].Value
        psnr    = [regex]::Match($p, "PSNR y:([\d\.]+)").Groups[1].Value
        psnrmin = [regex]::Match($p, "min:([\d\.]+)").Groups[1].Value
        vmaf    = $v
    }
}

$rows = @()
if( Test-Path $csv )
{
    # Normalise the schema so rows written before the radius column existed
    # (and rows from other invocations) export with a uniform column set.
    $rows = @(Import-Csv $csv | ForEach-Object {
        [pscustomobject]@{
            clip = $_.clip; config = $_.config; mode = [int]$_.mode
            radius = $(if( [int]$_.radius -gt 0 ) { [int]$_.radius } else { 2 })
            crf = [int]$_.crf; fps = $_.fps; kbps = $_.kbps
            ssim_y = $_.ssim_y; psnr_y = $_.psnr_y; psnr_min = $_.psnr_min; vmaf = $_.vmaf
            bytes = $_.bytes; exit = [int]$_.exit
        } })
    "resuming: {0} rows already in {1}" -f $rows.Count, $csv
}
$total = $selClips.Count * $selConfigs.Count * ($Modes -split ",").Count * ($RadiusList -split ",").Count * ($CrfList -split ",").Count
$done = 0
foreach( $c in $selClips )
{
    foreach( $cfg in $selConfigs )
    {
        foreach( $mode in ($Modes -split ",") )
        {
            foreach( $radius in ($RadiusList -split ",") )
            {
                # the radius only matters for the GPU modes; skip mode-0 rows
                if( [int]$radius -ne 2 -and [int]$mode -eq 0 ) { continue }
                foreach( $crf in ($CrfList -split ",") )
                {
                    $done++
                    $rsuffix = if( [int]$radius -ne 2 ) { "_r$radius" } else { "" }
                    $tag = "{0}_{1}_m{2}{3}_c{4}" -f $c.id, $cfg.id, $mode, $rsuffix, $crf
                    $have = @($rows | Where-Object {
                        $_.clip -eq $c.id -and $_.config -eq $cfg.id -and
                        [int]$_.mode -eq [int]$mode -and [int]$_.radius -eq [int]$radius -and
                        [int]$_.crf -eq [int]$crf })
                    if( $have.Count -gt 0 )
                    {
                        "[{0}/{1}] {2,-30} skip (already measured)" -f $done, $total, $tag
                        continue
                    }
                    $out = Join-Path $outDir "$tag.264"
                    $log = Join-Path $outDir "$tag.log"
                    $encArgs = $cfg.opts + @("--crf", $crf, "--threads", "1", "--gpu-me", $mode)
                    if( [int]$radius -ne 2 ) { $encArgs += @("--gpu-me-radius", $radius) }
                    $encArgs += @("--frames", $c.n, $c.file, "--input-res", ("{0}x{1}" -f $c.w, $c.h),
                                  "--fps", "30", "-o", $out)
                    & $exe @encArgs 2> $log
                    $code = $LASTEXITCODE
                    $text = Get-Content $log -Raw
                    $enc = [regex]::Match($text, "encoded (\d+) frames, ([\d\.]+) fps, ([\d\.]+) kb/s")
                    $bytes = (Get-Item $out).Length
                    $kbps = [math]::Round($bytes * 8 / ($c.n / 30.0) / 1000, 1)
                    $dec = Join-Path $tmpDir "$tag.dec.yuv"
                    & $Ffmpeg -hide_banner -loglevel error -i $out -pix_fmt yuv420p -f rawvideo -y $dec 2> $null
                    $m = Get-Metrics $c.file $dec $c.w $c.h $c.n (-not $NoVmaf)
                    Remove-Item $dec -ErrorAction SilentlyContinue
                    $rows += [pscustomobject]@{
                        clip = $c.id; config = $cfg.id; mode = [int]$mode; radius = [int]$radius
                        crf = [int]$crf; fps = $enc.Groups[2].Value; kbps = $kbps
                        ssim_y = $m.ssim; psnr_y = $m.psnr; psnr_min = $m.psnrmin; vmaf = $m.vmaf
                        bytes = $bytes; exit = $code
                    }
                    $r = $rows[-1]
                    "[{0}/{1}] {2,-30} fps={3,7} kbps={4,7} ssim={5} psnr={6} vmaf={7}" -f `
                        $done, $total, $tag, $r.fps, $r.kbps, $r.ssim_y, $r.psnr_y, $r.vmaf
                    $rows | Export-Csv $csv -NoTypeInformation -Encoding UTF8
                }
            }
        }
    }
}
"== done: $csv =="
