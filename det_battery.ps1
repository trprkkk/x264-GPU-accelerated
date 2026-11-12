# Determinism battery for hybrid-x264-gpu.
# Verifies:
#   - mode 1/2/3 produce identical output across repeated runs
#   - mode 0 matches stock x264 bit-exactly
#   - frame type decisions (I/P/B) are identical across all modes and runs
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File det_battery.ps1 [-Ffprobe <path>]
#   ffprobe is located on PATH (or passed with -Ffprobe); the frame-type
#   checks are skipped when it is unavailable.
param( [string]$Ffprobe = "" )
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$res = Join-Path $root "bench-results\det_battery_results.txt"
$ffprobe = $Ffprobe
if( !$ffprobe )
{
    $c = Get-Command ffprobe -ErrorAction SilentlyContinue
    if( $c ) { $ffprobe = $c.Source }
}
$lines = @()
$lines += "start $(Get-Date -Format o)"

$exe  = "$root\build-msvc\x264.exe"
$base = "$root\x264-baseline.exe"

$runs = @(
    @{ tag = "det_e1"; exe = $exe;  extra = @("--gpu-me", "2") },
    @{ tag = "det_e2"; exe = $exe;  extra = @("--gpu-me", "2") },
    @{ tag = "det_e3"; exe = $exe;  extra = @("--gpu-me", "2") },
    @{ tag = "det_f1"; exe = $exe;  extra = @("--gpu-me", "3") },
    @{ tag = "det_f2"; exe = $exe;  extra = @("--gpu-me", "3") },
    @{ tag = "det_m1"; exe = $exe;  extra = @("--gpu-me", "1") },
    @{ tag = "det_m2"; exe = $exe;  extra = @("--gpu-me", "1") },
    @{ tag = "det_z1"; exe = $exe;  extra = @("--gpu-me", "0") },
    @{ tag = "det_s1"; exe = $base; extra = @() }
)

$results = @()
foreach( $r in $runs )
{
    $a = @("--preset", "placebo", "--crf", "23", "--threads", "1") + $r.extra
    $a += @("testdata\motion720p.yuv", "--input-res", "1280x720", "--fps", "30", "-o", "testdata\$($r.tag).264")
    & $r.exe @a 2> "testdata\$($r.tag).log"
    $code = $LASTEXITCODE
    $sz = (Get-Item "testdata\$($r.tag).264").Length
    $h = (Get-FileHash "testdata\$($r.tag).264" -Algorithm SHA1).Hash
    $types = ""
    if( $ffprobe -and (Test-Path $ffprobe) )
    {
        $types = (& $ffprobe -v error -select_streams v -show_entries frame=pict_type -of csv=p=0 "testdata\$($r.tag).264") -join ''
    }
    $lines += "  $($r.tag) exit=$code size=$sz sha1=$h"
    $results += [pscustomobject]@{ tag = $r.tag; code = $code; size = $sz; sha1 = $h; types = $types }
}

function Get-Hashes($tags) { @($results | Where-Object { $tags -contains $_.tag } | ForEach-Object { $_.sha1 }) }

$lines += "== summary =="
$lines += "mode1 identical : " + ((Get-Hashes @("det_m1","det_m2") | Select-Object -Unique).Count -eq 1)
$lines += "mode2 identical : " + ((Get-Hashes @("det_e1","det_e2","det_e3") | Select-Object -Unique).Count -eq 1)
$lines += "mode3 identical : " + ((Get-Hashes @("det_f1","det_f2") | Select-Object -Unique).Count -eq 1)
$lines += "mode0 == stock  : " + ((Get-Hashes @("det_z1","det_s1") | Select-Object -Unique).Count -eq 1)

$types = @($results | ForEach-Object { $_.types } | Select-Object -Unique)
$typesOk = ($types.Count -eq 1) -and ($types[0].Length -gt 0)
$lines += "frame types identical across all runs : $typesOk"
if( -not $typesOk )
{
    foreach( $t in $types ) { $lines += "  types variant ($($t.Length) frames)" }
}

# cross-mode check: every mode must encode the same frame type sequence as stock
$refTypes = ($results | Where-Object { $_.tag -eq "det_s1" }).types
if( $ffprobe -and (Test-Path $ffprobe) )
{
    $lines += "== per-mode frame type diff vs stock (must be -1) =="
    foreach( $rr in $results )
    {
        if( $rr.tag -eq "det_s1" ) { continue }
        $d = -1
        for( $i = 0; $i -lt [Math]::Min($refTypes.Length, $rr.types.Length); $i++ )
        {
            if( $refTypes[$i] -ne $rr.types[$i] ) { $d = $i; break }
        }
        $lines += "  $($rr.tag): $d"
    }
}

$lines += "done $(Get-Date -Format o)"
$lines | Out-File $res
Get-Content $res
