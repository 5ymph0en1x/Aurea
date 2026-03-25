#Requires -RunAsAdministrator
# AUREA v12 codec uninstallation for Windows
# Also removes legacy codecs (XTS, Echo, old AUREA)

$ErrorActionPreference = "SilentlyContinue"

$WicDecoderCatid = "{7ED96837-96F0-4812-B211-F13C24117ED3}"

Write-Host "=== AUREA v12 Uninstallation ===" -ForegroundColor Yellow
Write-Host ""

# 1. Stop Explorer to release DLLs
Write-Host "[1/4] Stopping Explorer..."
Stop-Process -Name explorer -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
Write-Host "  OK" -ForegroundColor Green

# 2. Remove all codec registry entries and binaries
Write-Host "[2/4] Removing registry entries and binaries..."

$cleaned = @()

# --- XTS ---
$XtsDir = "C:\Program Files\x267"
$XtsClsidThumb = "{267F1100-C0DE-4ABC-9DEF-000000267001}"
$XtsClsidWic = "{267F1100-C0DE-4ABC-9DEF-000000267002}"

if ((Test-Path $XtsDir) -or (Test-Path "HKLM:\SOFTWARE\Classes\.xts")) {
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$XtsClsidThumb" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$XtsClsidWic" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$WicDecoderCatid\Instance\$XtsClsidWic" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\.xts" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\x267.image" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\x267.PhotoViewer" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\SystemFileAssociations\image\shell\ConvertToXTS" -Recurse -Force
    Remove-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\KindMap" -Name ".xts" -Force
    Remove-Item -Path $XtsDir -Recurse -Force
    $cleaned += "XTS"
}

# --- Echo ---
$EchoDir = "C:\Program Files\echolot"
$EchoClsidThumb = "{EC401017-C0DE-4ABC-9DEF-000000EC4001}"
$EchoClsidWic = "{EC401017-C0DE-4ABC-9DEF-000000EC4002}"

if ((Test-Path $EchoDir) -or (Test-Path "HKLM:\SOFTWARE\Classes\.echo")) {
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$EchoClsidThumb" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$EchoClsidWic" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$WicDecoderCatid\Instance\$EchoClsidWic" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\.echo" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\echolot.image" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\echolot.PhotoViewer" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\SystemFileAssociations\image\shell\ConvertToECHO" -Recurse -Force
    Remove-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\KindMap" -Name ".echo" -Force
    Remove-Item -Path $EchoDir -Recurse -Force
    $cleaned += "Echo"
}

# --- AUREA (v12 + old) ---
$AureaDir = "C:\Program Files\aurea"
$AureaClsidThumb = "{267A0E00-C0DE-4ABC-9DEF-000000267011}"
$AureaClsidWic = "{267A0E00-C0DE-4ABC-9DEF-000000267012}"

if ((Test-Path $AureaDir) -or (Test-Path "HKLM:\SOFTWARE\Classes\.aur")) {
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$AureaClsidThumb" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$AureaClsidWic" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$WicDecoderCatid\Instance\$AureaClsidWic" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\.aur" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\aurea.PhotoViewer" -Recurse -Force
    # Old single-entry context menu
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\SystemFileAssociations\image\shell\ConvertToAUREA" -Recurse -Force
    # v12 cascading submenu (parent)
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\SystemFileAssociations\image\shell\AUREAConvert" -Recurse -Force
    # Stale entries from buggy install (wrong registry location)
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\AUREAConvert.Low" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\AUREAConvert.Medium" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\AUREAConvert.High" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\AUREAConvert.Ultra" -Recurse -Force
    # Correct CommandStore location
    $CmdStore = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell"
    Remove-Item -Path "$CmdStore\AUREAConvert.Low" -Recurse -Force
    Remove-Item -Path "$CmdStore\AUREAConvert.Medium" -Recurse -Force
    Remove-Item -Path "$CmdStore\AUREAConvert.High" -Recurse -Force
    Remove-Item -Path "$CmdStore\AUREAConvert.Ultra" -Recurse -Force
    Remove-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\KindMap" -Name ".aur" -Force
    Remove-Item -Path $AureaDir -Recurse -Force
    $cleaned += "AUREA"
}

if ($cleaned.Count -gt 0) {
    Write-Host "  Removed: $($cleaned -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host "  Nothing to remove" -ForegroundColor Green
}

# 3. Clear thumbnail cache
Write-Host "[3/4] Clearing thumbnail cache..."
$ThumbCacheDir = "$env:LOCALAPPDATA\Microsoft\Windows\Explorer"
Get-ChildItem "$ThumbCacheDir\thumbcache_*.db" -ErrorAction SilentlyContinue | Remove-Item -Force
Write-Host "  OK" -ForegroundColor Green

# 4. Restart Explorer
Write-Host "[4/4] Restarting Explorer..."
Start-Process explorer
Write-Host "  OK" -ForegroundColor Green

Write-Host ""
Write-Host "Uninstallation complete." -ForegroundColor Green
