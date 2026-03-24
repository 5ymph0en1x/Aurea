#Requires -RunAsAdministrator
# AUREA v12 codec installation for Windows
# - Removes any previous codec (XTS, Echo, old AUREA)
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

Write-Host "=== AUREA v10 Installation for Windows ===" -ForegroundColor Cyan
Write-Host ""

# ======================================================================
# 0. Build if needed
# ======================================================================
$SrcDll = "$ScriptDir\aurea_shell.dll"
$BuildDir = "$ScriptDir\..\target\release"
if (-not (Test-Path $SrcDll) -and -not (Test-Path "$BuildDir\aurea_shell.dll")) {
    Write-Host "[0] Binaries not found - building from source..." -ForegroundColor Yellow
    $WorkspaceRoot = (Resolve-Path "$ScriptDir\..").Path
    Push-Location $WorkspaceRoot
    try {
        & cargo build --release --workspace
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: cargo build failed." -ForegroundColor Red
            exit 1
        }
        Write-Host "  Build OK" -ForegroundColor Green
    } finally {
        Pop-Location
    }
}

# ======================================================================
# 1. Stop Explorer (DLL is locked otherwise)
# ======================================================================
Write-Host "[1/7] Stopping Explorer..."
Stop-Process -Name explorer -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "  OK" -ForegroundColor Green

# ======================================================================
# 2. Remove previous installations (XTS, Echo, old AUREA)
# ======================================================================
Write-Host "[2/7] Removing previous codec installations..."

$SavedErrorPref = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"

$cleaned = @()

# --- XTS (C:\Program Files\x267) ---
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

# --- Echo (C:\Program Files\echolot) ---
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

# --- Old AUREA - same CLSIDs, same dir - wipe registry + stale binaries ---
if ((Test-Path $InstallDir) -or (Test-Path "HKLM:\SOFTWARE\Classes\.aur")) {
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$ClsidThumb" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$ClsidWic" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\CLSID\$WicDecoderCatid\Instance\$ClsidWic" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\.aur" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\aurea.PhotoViewer" -Recurse -Force
    Remove-Item -Path "HKLM:\SOFTWARE\Classes\SystemFileAssociations\image\shell\ConvertToAUREA" -Recurse -Force
    Remove-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\KindMap" -Name ".aur" -Force
    Remove-Item -Path $InstallDir -Recurse -Force
    $cleaned += "AUREA"
}

$ErrorActionPreference = $SavedErrorPref

if ($cleaned.Count -gt 0) {
    Write-Host "  Removed: $($cleaned -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host "  Nothing to remove" -ForegroundColor Green
}

# ======================================================================
# 3. Copy v10 binaries
# ======================================================================
Write-Host "[3/7] Copying binaries to $InstallDir..."
New-Item -Path $InstallDir -ItemType Directory -Force | Out-Null

# Check next to script first (GitHub Release zip), then build output.
$SrcDll = "$ScriptDir\aurea_shell.dll"
$SrcCli = "$ScriptDir\aurea.exe"
$SrcViewer = "$ScriptDir\aurea-viewer.exe"

if (-not (Test-Path $SrcDll)) {
    $SrcDll = "$ScriptDir\..\target\release\aurea_shell.dll"
    $SrcCli = "$ScriptDir\..\target\release\aurea.exe"
    $SrcViewer = "$ScriptDir\..\target\release\aurea-viewer.exe"
}

if (-not (Test-Path $SrcDll)) {
    Write-Host "ERROR: aurea_shell.dll not found. Run 'cargo build --release --workspace' first." -ForegroundColor Red
    Start-Process explorer
    exit 1
}

Copy-Item $SrcDll $DllPath -Force
Copy-Item $SrcCli $CliPath -Force
Copy-Item $SrcViewer $ViewerPath -Force

# Generate silent convert helper (VBS, zero window flash)
$VbsPath = "$InstallDir\convert.vbs"
@"
Dim exe, src
exe = "$InstallDir\aurea.exe"
src = WScript.Arguments(0)
CreateObject("WScript.Shell").Run """" & exe & """ encode """ & src & """ -q 50", 0, True
"@ | Set-Content -Path $VbsPath -Encoding ASCII

Write-Host "  OK" -ForegroundColor Green

# ======================================================================
# 4. Register the COM thumbnail provider
# ======================================================================
Write-Host "[4/7] Registering the thumbnail provider..."

$ClsidKey = "HKLM:\SOFTWARE\Classes\CLSID\$ClsidThumb"
New-Item -Path $ClsidKey -Value "AUREA Thumbnail Provider" -Force | Out-Null
New-Item -Path "$ClsidKey\InProcServer32" -Value $DllPath -Force | Out-Null
Set-ItemProperty -Path "$ClsidKey\InProcServer32" -Name "ThreadingModel" -Value "Both"
Set-ItemProperty -Path $ClsidKey -Name "DisableProcessIsolation" -Value 1 -Type DWord

Write-Host "  OK" -ForegroundColor Green

# ======================================================================
# 5. File association + thumbnail handler
# ======================================================================
Write-Host "[5/7] Associating the .aur extension..."

# Extension
New-Item -Path "HKLM:\SOFTWARE\Classes\.aur" -Value "aurea.image" -Force | Out-Null
Set-ItemProperty -Path "HKLM:\SOFTWARE\Classes\.aur" -Name "Content Type" -Value "image/x-aurea"
Set-ItemProperty -Path "HKLM:\SOFTWARE\Classes\.aur" -Name "PerceivedType" -Value "image"

# File type description
New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image" -Value "AUREA v10 Image" -Force | Out-Null
New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\DefaultIcon" -Value "imageres.dll,67" -Force | Out-Null

# Default open (aurea-viewer)
New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell" -Force | Out-Null
Set-ItemProperty -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell" -Name "(Default)" -Value "open"
New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\open" -Value "Open with AUREA Viewer" -Force | Out-Null
New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\open\command" -Force | Out-Null
Set-ItemProperty -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\open\command" -Name "(Default)" -Value "`"$ViewerPath`" `"%1`""

# Windows Photo Viewer (WIC)
$PhotoViewerDll = "$env:ProgramFiles\Windows Photo Viewer\PhotoViewer.dll"
if (Test-Path $PhotoViewerDll) {
    $PVAssocKey = "HKLM:\SOFTWARE\Classes\aurea.PhotoViewer"
    New-Item -Path $PVAssocKey -Value "AUREA Image (Photo Viewer)" -Force | Out-Null
    New-Item -Path "$PVAssocKey\DefaultIcon" -Value "imageres.dll,67" -Force | Out-Null
    New-Item -Path "$PVAssocKey\shell\open\command" -Force | Out-Null
    Set-ItemProperty -Path "$PVAssocKey\shell\open\command" -Name "(Default)" -Value "`"%SystemRoot%\System32\rundll32.exe`" `"$PhotoViewerDll`", ImageView_Fullscreen %1"
    New-Item -Path "$PVAssocKey\shell\open\DropTarget" -Force | Out-Null
    Set-ItemProperty -Path "$PVAssocKey\shell\open\DropTarget" -Name "CLSID" -Value "{FFE2A43C-56B9-4bf5-9A79-CC6D4285608A}"

    $OpenWithKey = "HKLM:\SOFTWARE\Classes\.aur\OpenWithProgids"
    New-Item -Path $OpenWithKey -Force | Out-Null
    Set-ItemProperty -Path $OpenWithKey -Name "aurea.PhotoViewer" -Value ([byte[]]@()) -Type Binary

    New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\photoviewer" -Value "Open with Windows Photo Viewer" -Force | Out-Null
    New-Item -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\photoviewer\command" -Force | Out-Null
    Set-ItemProperty -Path "HKLM:\SOFTWARE\Classes\aurea.image\shell\photoviewer\command" -Name "(Default)" -Value "`"%SystemRoot%\System32\rundll32.exe`" `"$PhotoViewerDll`", ImageView_Fullscreen %1"
    Write-Host "  OK (+ Windows Photo Viewer)" -ForegroundColor Green
} else {
    Write-Host "  OK" -ForegroundColor Green
}

# Thumbnail handler on .aur
$ShellExKey = "HKLM:\SOFTWARE\Classes\.aur\ShellEx\$ThumbHandlerCatid"
New-Item -Path $ShellExKey -Value $ClsidThumb -Force | Out-Null

# KindMap: treat .aur as an image
$KindMapKey = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\KindMap"
if (-not (Test-Path $KindMapKey)) {
    New-Item -Path $KindMapKey -Force | Out-Null
}
Set-ItemProperty -Path $KindMapKey -Name ".aur" -Value "picture"

# ======================================================================
# 6. Context menu "Convert to AUREA" on all images
# ======================================================================
Write-Host "[6/7] Adding 'Convert to AUREA' context menu..."

$ConvertKey = "HKLM:\SOFTWARE\Classes\SystemFileAssociations\image\shell\ConvertToAUREA"
New-Item -Path $ConvertKey -Value "Convert to AUREA" -Force | Out-Null
Set-ItemProperty -Path $ConvertKey -Name "Icon" -Value "`"$CliPath`",0"
New-Item -Path "$ConvertKey\command" -Force | Out-Null
Set-ItemProperty -Path "$ConvertKey\command" -Name "(Default)" -Value "wscript.exe `"$InstallDir\convert.vbs`" `"%1`""

Write-Host "  OK" -ForegroundColor Green

# ======================================================================
# 7. Register the WIC codec
# ======================================================================
Write-Host "[7/7] Registering the WIC codec..."

$WicClsidKey = "HKLM:\SOFTWARE\Classes\CLSID\$ClsidWic"
New-Item -Path $WicClsidKey -Value "AUREA WIC Decoder" -Force | Out-Null
New-Item -Path "$WicClsidKey\InProcServer32" -Value $DllPath -Force | Out-Null
Set-ItemProperty -Path "$WicClsidKey\InProcServer32" -Name "ThreadingModel" -Value "Both"

Set-ItemProperty -Path $WicClsidKey -Name "Author" -Value "aurea"
Set-ItemProperty -Path $WicClsidKey -Name "FriendlyName" -Value "AUREA v10 Image Decoder"
Set-ItemProperty -Path $WicClsidKey -Name "ContainerFormat" -Value $ContainerGuid
Set-ItemProperty -Path $WicClsidKey -Name "FileExtensions" -Value ".aur"
Set-ItemProperty -Path $WicClsidKey -Name "MimeTypes" -Value "image/x-aurea"
Set-ItemProperty -Path $WicClsidKey -Name "Version" -Value "10.0.0"
Set-ItemProperty -Path $WicClsidKey -Name "VendorGUID" -Value $ContainerGuid

$FormatsKey = "$WicClsidKey\Formats"
New-Item -Path $FormatsKey -Force | Out-Null
New-Item -Path "$FormatsKey\{6fddc324-4e03-4bfe-b185-3d77768dc90f}" -Force | Out-Null

# Detection pattern: "AUR2" at offset 0
$PatternsKey = "$WicClsidKey\Patterns"
New-Item -Path $PatternsKey -Force | Out-Null
$Pat0 = "$PatternsKey\0"
New-Item -Path $Pat0 -Force | Out-Null
Set-ItemProperty -Path $Pat0 -Name "Position" -Value 0 -Type DWord
Set-ItemProperty -Path $Pat0 -Name "Length" -Value 4 -Type DWord
Set-ItemProperty -Path $Pat0 -Name "Pattern" -Value ([byte[]]@(0x41,0x55,0x52,0x32)) -Type Binary
Set-ItemProperty -Path $Pat0 -Name "Mask" -Value ([byte[]]@(0xFF,0xFF,0xFF,0xFF)) -Type Binary

$WicCatKey = "HKLM:\SOFTWARE\Classes\CLSID\$WicDecoderCatid\Instance\$ClsidWic"
New-Item -Path $WicCatKey -Force | Out-Null
Set-ItemProperty -Path $WicCatKey -Name "CLSID" -Value $ClsidWic
Set-ItemProperty -Path $WicCatKey -Name "FriendlyName" -Value "AUREA v10 Image Decoder"

Write-Host "  OK" -ForegroundColor Green

# ======================================================================
# Done - clear cache and restart Explorer
# ======================================================================
Write-Host ""
Write-Host "Clearing thumbnail cache and restarting Explorer..."
$ThumbCacheDir = "$env:LOCALAPPDATA\Microsoft\Windows\Explorer"
Get-ChildItem "$ThumbCacheDir\thumbcache_*.db" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Start-Process explorer
Write-Host "  OK" -ForegroundColor Green

Write-Host ""
Write-Host "=== AUREA v10 installation complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host ".aur files (AUR2 format) now have:"
Write-Host "  - Native thumbnails in Windows Explorer"
Write-Host "  - Double-click to open with AUREA Viewer"
Write-Host "  - 'image' type in file properties"
Write-Host "  - Right-click on any image: 'Convert to AUREA'"
Write-Host "  - WIC codec: Photos, Paint, any WIC application"
Write-Host ""
Write-Host "Binaries installed to: $InstallDir"
