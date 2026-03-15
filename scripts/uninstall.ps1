#Requires -RunAsAdministrator
# AUREA codec uninstallation for Windows

$ErrorActionPreference = "SilentlyContinue"

$InstallDir = "C:\Program Files\aurea"
$ClsidThumb = "{267A0E00-C0DE-4ABC-9DEF-000000267011}"
$ClsidWic = "{267A0E00-C0DE-4ABC-9DEF-000000267012}"
$ContainerGuid = "{267A0E00-C0DE-4ABC-9DEF-000000267013}"
$ThumbHandlerCatid = "{E357FCCD-A995-4576-B01F-234630154E96}"
$WicDecoderCatid = "{7ED96837-96F0-4812-B211-F13C24117ED3}"

Write-Host "=== AUREA Uninstallation ===" -ForegroundColor Yellow
Write-Host ""

# 1. Stop Explorer to release the DLL
Write-Host "[1/5] Stopping Explorer..."
Stop-Process -Name explorer -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
Write-Host "  OK" -ForegroundColor Green

# 2. Remove all registry keys
Write-Host "[2/5] Removing registry entries..."

# COM CLSIDs
Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$ClsidThumb" -Recurse -Force
Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$ClsidWic" -Recurse -Force

# WIC decoder category instance
Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$WicDecoderCatid\Instance\$ClsidWic" -Recurse -Force

# File extension and type
Remove-Item -Path "HKLM:\SOFTWARE\Classes\.aur" -Recurse -Force
Remove-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image" -Recurse -Force
Remove-Item -Path "HKLM:\SOFTWARE\Classes\aurea.PhotoViewer" -Recurse -Force

# KindMap entry
Remove-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\KindMap" -Name ".aur" -Force

# Context menu "Convert to AUREA"
Remove-Item -Path "HKLM:\SOFTWARE\Classes\SystemFileAssociations\image\shell\ConvertToAUREA" -Recurse -Force

Write-Host "  OK" -ForegroundColor Green

# 3. Delete installed binaries
Write-Host "[3/5] Deleting $InstallDir..."
Remove-Item -Path $InstallDir -Recurse -Force
Write-Host "  OK" -ForegroundColor Green

# 4. Clear thumbnail cache
Write-Host "[4/5] Clearing thumbnail cache..."
$ThumbCacheDir = "$env:LOCALAPPDATA\Microsoft\Windows\Explorer"
Get-ChildItem "$ThumbCacheDir\thumbcache_*.db" -ErrorAction SilentlyContinue | Remove-Item -Force
Write-Host "  OK" -ForegroundColor Green

# 5. Restart Explorer
Write-Host "[5/5] Restarting Explorer..."
Start-Process explorer
Write-Host "  OK" -ForegroundColor Green

Write-Host ""
Write-Host "AUREA uninstallation complete." -ForegroundColor Green
