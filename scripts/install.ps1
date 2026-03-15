#Requires -RunAsAdministrator
# AUREA codec installation for Windows
# - Native thumbnails in Explorer
# - File association (double-click)
# - "image" type in Explorer
# - WIC codec (Photos, Paint, any WIC app)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallDir = "C:\Program Files\aurea"
$DllPath = "$InstallDir\aurea_shell.dll"
$CliPath = "$InstallDir\aurea.exe"
$ViewerPath = "$InstallDir\aurea-viewer.exe"
$ClsidThumb = "{267A0E00-C0DE-4ABC-9DEF-000000267011}"
$ClsidWic = "{267A0E00-C0DE-4ABC-9DEF-000000267012}"
$ContainerGuid = "{267A0E00-C0DE-4ABC-9DEF-000000267013}"
$ThumbHandlerCatid = "{E357FCCD-A995-4576-B01F-234630154E96}"
$WicDecoderCatid = "{7ED96837-96F0-4812-B211-F13C24117ED3}"

Write-Host "=== AUREA Installation for Windows ===" -ForegroundColor Cyan
Write-Host ""

# 1. Stop Explorer BEFORE copying (the DLL is locked otherwise)
Write-Host "[1/7] Stopping Explorer to unlock the DLL..."
Stop-Process -Name explorer -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "  OK" -ForegroundColor Green

# 2. Copy binaries
Write-Host "[2/7] Copying binaries to $InstallDir..."
New-Item -Path $InstallDir -ItemType Directory -Force | Out-Null

# Check for binaries in the same directory as the script first (GitHub Release zip),
# then fall back to the build output directory (building from source).
$SrcDll = "$ScriptDir\aurea_shell.dll"
$SrcCli = "$ScriptDir\aurea.exe"
$SrcViewer = "$ScriptDir\aurea-viewer.exe"

if (-not (Test-Path $SrcDll)) {
    Write-Host "  Binaries not found next to script, checking build output..." -ForegroundColor Yellow
    $SrcDll = "$ScriptDir\..\target\release\aurea_shell.dll"
    $SrcCli = "$ScriptDir\..\target\release\aurea.exe"
    $SrcViewer = "$ScriptDir\..\target\release\aurea-viewer.exe"
}

if (-not (Test-Path $SrcDll)) {
    Write-Host "ERROR: $SrcDll not found. Run 'cargo build --release' first, or place binaries next to this script." -ForegroundColor Red
    Start-Process explorer
    exit 1
}

Copy-Item $SrcDll $DllPath -Force
Copy-Item $SrcCli $CliPath -Force
Copy-Item $SrcViewer $ViewerPath -Force

Write-Host "  OK" -ForegroundColor Green

# 3. Register the COM CLSID (thumbnail provider)
Write-Host "[3/7] Registering the thumbnail provider COM..."

$ClsidKey = "HKLM:\SOFTWARE\Classes\CLSID\$ClsidThumb"
New-Item -Path $ClsidKey -Value "AUREA Thumbnail Provider" -Force | Out-Null
New-Item -Path "$ClsidKey\InProcServer32" -Value $DllPath -Force | Out-Null
Set-ItemProperty -Path "$ClsidKey\InProcServer32" -Name "ThreadingModel" -Value "Both"

# Disable process isolation (required for IInitializeWithItem)
Set-ItemProperty -Path $ClsidKey -Name "DisableProcessIsolation" -Value 1 -Type DWord

Write-Host "  OK" -ForegroundColor Green

# 4. File association
Write-Host "[4/7] Associating the .aur extension..."

# File type
New-Item -Path "HKLM:\SOFTWARE\Classes\.aur" -Value "aurea.image" -Force | Out-Null
Set-ItemProperty -Path "HKLM:\SOFTWARE\Classes\.aur" -Name "Content Type" -Value "image/x-aurea"
Set-ItemProperty -Path "HKLM:\SOFTWARE\Classes\.aur" -Name "PerceivedType" -Value "image"

# Description and icon
New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image" -Value "AUREA Compressed Image" -Force | Out-Null
New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\DefaultIcon" -Value "imageres.dll,67" -Force | Out-Null

# Default open command (aurea-viewer)
New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell" -Force | Out-Null
Set-ItemProperty -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell" -Name "(Default)" -Value "open"

New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\open" -Value "Open with AUREA Viewer" -Force | Out-Null
New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\open\command" -Force | Out-Null
Set-ItemProperty -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\open\command" -Name "(Default)" -Value "`"$ViewerPath`" `"%1`""

# Open with Windows Photo Viewer (uses WIC = our codec)
$PhotoViewerDll = "$env:ProgramFiles\Windows Photo Viewer\PhotoViewer.dll"
if (Test-Path $PhotoViewerDll) {
    # Register PhotoViewer association for .aur
    $PVAssocKey = "HKLM:\SOFTWARE\Classes\aurea.PhotoViewer"
    New-Item -Path $PVAssocKey -Value "AUREA Image (Photo Viewer)" -Force | Out-Null
    New-Item -Path "$PVAssocKey\DefaultIcon" -Value "imageres.dll,67" -Force | Out-Null
    New-Item -Path "$PVAssocKey\shell\open\command" -Force | Out-Null
    Set-ItemProperty -Path "$PVAssocKey\shell\open\command" -Name "(Default)" -Value "`"%SystemRoot%\System32\rundll32.exe`" `"$PhotoViewerDll`", ImageView_Fullscreen %1"
    New-Item -Path "$PVAssocKey\shell\open\DropTarget" -Force | Out-Null
    Set-ItemProperty -Path "$PVAssocKey\shell\open\DropTarget" -Name "CLSID" -Value "{FFE2A43C-56B9-4bf5-9A79-CC6D4285608A}"

    # Add to OpenWithProgids for the "Open with" menu
    $OpenWithKey = "HKLM:\SOFTWARE\Classes\.aur\OpenWithProgids"
    New-Item -Path $OpenWithKey -Force | Out-Null
    Set-ItemProperty -Path $OpenWithKey -Name "aurea.PhotoViewer" -Value ([byte[]]@()) -Type Binary

    # Context menu option
    New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\photoviewer" -Value "Open with Windows Photo Viewer" -Force | Out-Null
    New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\photoviewer\command" -Force | Out-Null
    Set-ItemProperty -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\photoviewer\command" -Name "(Default)" -Value "`"%SystemRoot%\System32\rundll32.exe`" `"$PhotoViewerDll`", ImageView_Fullscreen %1"

    Write-Host "  OK (+ Windows Photo Viewer)" -ForegroundColor Green
} else {
    Write-Host "  OK (Windows Photo Viewer not found)" -ForegroundColor Yellow
}


# 4b. Register the thumbnail handler
Write-Host "[4b/7] Registering the thumbnail handler..."

$ShellExKey = "HKLM:\SOFTWARE\Classes\.aur\ShellEx\$ThumbHandlerCatid"
New-Item -Path $ShellExKey -Value $ClsidThumb -Force | Out-Null

# KindMap: treat .aur as an image
$KindMapKey = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\KindMap"
if (-not (Test-Path $KindMapKey)) {
    New-Item -Path $KindMapKey -Force | Out-Null
}
Set-ItemProperty -Path $KindMapKey -Name ".aur" -Value "picture"

Write-Host "  OK" -ForegroundColor Green

# 5. Context menu "Convert to AUREA" on all images
Write-Host "[5/7] Adding 'Convert to AUREA' context menu on images..."

$ConvertKey = "HKLM:\SOFTWARE\Classes\SystemFileAssociations\image\shell\ConvertToAUREA"
New-Item -Path $ConvertKey -Value "Convert to AUREA" -Force | Out-Null
Set-ItemProperty -Path $ConvertKey -Name "Icon" -Value "`"$CliPath`",0"
New-Item -Path "$ConvertKey\command" -Force | Out-Null
Set-ItemProperty -Path "$ConvertKey\command" -Name "(Default)" -Value "cmd /c `"`"$CliPath`" encode --geometric `"%1`" -q 88 && echo. && echo Conversion complete! && timeout /t 3`""

Write-Host "  OK" -ForegroundColor Green

# 6. Register the WIC codec
Write-Host "[6/7] Registering the WIC codec (Photo Viewer, WIC apps)..."

# WIC decoder CLSID
$WicClsidKey = "HKLM:\SOFTWARE\Classes\CLSID\$ClsidWic"
New-Item -Path $WicClsidKey -Value "AUREA WIC Decoder" -Force | Out-Null
New-Item -Path "$WicClsidKey\InProcServer32" -Value $DllPath -Force | Out-Null
Set-ItemProperty -Path "$WicClsidKey\InProcServer32" -Name "ThreadingModel" -Value "Both"

# Decoder information
Set-ItemProperty -Path $WicClsidKey -Name "Author" -Value "aurea"
Set-ItemProperty -Path $WicClsidKey -Name "FriendlyName" -Value "AUREA Image Decoder"
Set-ItemProperty -Path $WicClsidKey -Name "ContainerFormat" -Value $ContainerGuid
Set-ItemProperty -Path $WicClsidKey -Name "FileExtensions" -Value ".aur"
Set-ItemProperty -Path $WicClsidKey -Name "MimeTypes" -Value "image/x-aurea"
Set-ItemProperty -Path $WicClsidKey -Name "Version" -Value "1.0.0"
Set-ItemProperty -Path $WicClsidKey -Name "VendorGUID" -Value $ContainerGuid

# Supported formats
$FormatsKey = "$WicClsidKey\Formats"
New-Item -Path $FormatsKey -Force | Out-Null
# GUID_WICPixelFormat32bppBGRA
New-Item -Path "$FormatsKey\{6fddc324-4e03-4bfe-b185-3d77768dc90f}" -Force | Out-Null

# Detection pattern (magic "AURA" at the first 4 bytes)
$PatternsKey = "$WicClsidKey\Patterns"
New-Item -Path $PatternsKey -Force | Out-Null
$Pat0 = "$PatternsKey\0"
New-Item -Path $Pat0 -Force | Out-Null
Set-ItemProperty -Path $Pat0 -Name "Position" -Value 0 -Type DWord
Set-ItemProperty -Path $Pat0 -Name "Length" -Value 4 -Type DWord
Set-ItemProperty -Path $Pat0 -Name "Pattern" -Value ([byte[]]@(0x41,0x55,0x52,0x41)) -Type Binary
Set-ItemProperty -Path $Pat0 -Name "Mask" -Value ([byte[]]@(0xFF,0xFF,0xFF,0xFF)) -Type Binary

# Register in the WIC Decoder category
$WicCatKey = "HKLM:\SOFTWARE\Classes\CLSID\$WicDecoderCatid\Instance\$ClsidWic"
New-Item -Path $WicCatKey -Force | Out-Null
Set-ItemProperty -Path $WicCatKey -Name "CLSID" -Value $ClsidWic
Set-ItemProperty -Path $WicCatKey -Name "FriendlyName" -Value "AUREA Image Decoder"

Write-Host "  OK" -ForegroundColor Green

# 7. Clear thumbnail cache and restart Explorer
Write-Host "[7/7] Clearing cache and restarting Explorer..."

$ThumbCacheDir = "$env:LOCALAPPDATA\Microsoft\Windows\Explorer"
Get-ChildItem "$ThumbCacheDir\thumbcache_*.db" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Start-Process explorer

Write-Host "  OK" -ForegroundColor Green

Write-Host ""
Write-Host "=== Installation complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host ".aur files now have:"
Write-Host "  - Native thumbnails in Windows Explorer"
Write-Host "  - Double-click to open with AUREA Viewer"
Write-Host "  - 'image' type in file properties"
Write-Host "  - Right-click on any image: 'Convert to AUREA'"
Write-Host "  - Right-click on .aur: 'Open with Windows Photo Viewer'"
Write-Host "  - WIC support: Windows Photo Viewer, WIC apps"
Write-Host ""
Write-Host "Binaries installed to: $InstallDir"
