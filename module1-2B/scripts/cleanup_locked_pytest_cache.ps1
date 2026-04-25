param(
    [string]$Root = "C:\Users\SAMSUNG\Downloads\kcc_2",
    [switch]$ForceAcl
)

$ErrorActionPreference = "Continue"

if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    Write-Error "Root path not found: $Root"
    exit 2
}

$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
$targets = Get-ChildItem -LiteralPath $resolvedRoot -Directory -Filter "pytest-cache-files-*" -ErrorAction SilentlyContinue

Write-Host "Root: $resolvedRoot"
Write-Host "Found targets: $($targets.Count)"

if ($targets.Count -eq 0) {
    Write-Host "Nothing to clean."
    exit 0
}

$removed = New-Object System.Collections.Generic.List[string]
$failed = New-Object System.Collections.Generic.List[string]

foreach ($dir in $targets) {
    $path = $dir.FullName

    try {
        Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
        $removed.Add($path)
        continue
    } catch {
        if (-not $ForceAcl) {
            $failed.Add($path)
            continue
        }
    }

    try {
        # Try ownership/ACL repair only when -ForceAcl is requested.
        takeown /f $path /r /d y | Out-Null
        icacls $path /grant "$env:USERNAME:(OI)(CI)F" /t /c | Out-Null
        Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
        $removed.Add($path)
    } catch {
        $failed.Add($path)
    }
}

$remaining = (Get-ChildItem -LiteralPath $resolvedRoot -Directory -Filter "pytest-cache-files-*" -ErrorAction SilentlyContinue).Count

Write-Host ""
Write-Host "Removed:   $($removed.Count)"
Write-Host "Failed:    $($failed.Count)"
Write-Host "Remaining: $remaining"

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed paths:"
    $failed | ForEach-Object { Write-Host " - $_" }
    exit 1
}

exit 0
