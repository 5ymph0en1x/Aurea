//! WIC decoder for the AUREA (.aur) image format.
//!
//! A COM in-process server exposing `IWICBitmapDecoder` +
//! `IWICBitmapFrameDecode`, so every WIC consumer — Windows Photo Viewer,
//! Explorer thumbnails, Paint, WPF — reads `.aur` natively. Decoding is
//! delegated to `aurea-core`; the frame is served as 24bpp BGR.
#![allow(non_snake_case)]

use std::sync::Mutex;

use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Imaging::*;
use windows::Win32::System::Com::*;

/// {A08EA2A4-32F0-4D42-9E3B-1A76A2A7E132}
pub const CLSID_AUREA_DECODER: GUID = GUID::from_u128(0xA08EA2A4_32F0_4D42_9E3B_1A76A2A7E132);
/// {B54DDAB2-8C51-4B2A-BFCA-4A2A21F3D4C2}
pub const GUID_CONTAINER_AUREA: GUID = GUID::from_u128(0xB54DDAB2_8C51_4B2A_BFCA_4A2A21F3D4C2);
/// GUID_WICPixelFormat24bppBGR
const PF_24BPP_BGR: GUID = GUID::from_u128(0x6fddc324_4e03_4bfe_b185_3d77768dc90c);

const WINCODEC_ERR_NOTINITIALIZED: HRESULT = HRESULT(0x88982F0Cu32 as i32);
const WINCODEC_ERR_PALETTEUNAVAILABLE: HRESULT = HRESULT(0x88982F45u32 as i32);
const WINCODEC_ERR_UNSUPPORTEDOPERATION: HRESULT = HRESULT(0x88982F81u32 as i32);
const WINCODEC_ERR_CODECNOTHUMBNAIL: HRESULT = HRESULT(0x88982F44u32 as i32);
const WINCODEC_ERR_FRAMEMISSING: HRESULT = HRESULT(0x88982F62u32 as i32);
const WINCODEC_ERR_BADIMAGE: HRESULT = HRESULT(0x88982F60u32 as i32);

/// Debug trace to %TEMP%\aurea_wic.log, enabled by AUREA_WIC_LOG=1.
fn dbg_log(msg: &str) {
    if std::env::var_os("AUREA_WIC_LOG").is_none() {
        return;
    }
    if let Some(tmp) = std::env::var_os("TEMP") {
        let path = std::path::Path::new(&tmp).join("aurea_wic.log");
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{msg}");
        }
    }
}

struct FrameData {
    width: u32,
    height: u32,
    /// Tightly packed BGR, stride = width * 3.
    bgr: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

#[implement(IWICBitmapFrameDecode, IWICBitmapSource)]
struct AureaFrame {
    data: std::sync::Arc<FrameData>,
}

impl IWICBitmapSource_Impl for AureaFrame_Impl {
    fn GetSize(&self, width: *mut u32, height: *mut u32) -> Result<()> {
        unsafe {
            *width = self.data.width;
            *height = self.data.height;
        }
        Ok(())
    }

    fn GetPixelFormat(&self) -> Result<GUID> {
        Ok(PF_24BPP_BGR)
    }

    fn GetResolution(&self, dpix: *mut f64, dpiy: *mut f64) -> Result<()> {
        unsafe {
            *dpix = 96.0;
            *dpiy = 96.0;
        }
        Ok(())
    }

    fn CopyPalette(&self, _palette: Ref<IWICPalette>) -> Result<()> {
        Err(WINCODEC_ERR_PALETTEUNAVAILABLE.into())
    }

    fn CopyPixels(
        &self,
        prc: *const WICRect,
        stride: u32,
        buffer_size: u32,
        buffer: *mut u8,
    ) -> Result<()> {
        let data = &self.data;
        let (x, y, w, h) = unsafe {
            if prc.is_null() {
                (0i32, 0i32, data.width as i32, data.height as i32)
            } else {
                ((*prc).X, (*prc).Y, (*prc).Width, (*prc).Height)
            }
        };
        if x < 0
            || y < 0
            || w < 0
            || h < 0
            || (x + w) as u32 > data.width
            || (y + h) as u32 > data.height
        {
            return Err(E_INVALIDARG.into());
        }
        let row_bytes = w as usize * 3;
        if (stride as usize) < row_bytes {
            return Err(E_INVALIDARG.into());
        }
        if (buffer_size as usize) < (h as usize - 1) * stride as usize + row_bytes && h > 0 {
            return Err(E_INVALIDARG.into());
        }
        let src_stride = data.width as usize * 3;
        unsafe {
            for row in 0..h as usize {
                let src_off = (y as usize + row) * src_stride + x as usize * 3;
                let dst = buffer.add(row * stride as usize);
                std::ptr::copy_nonoverlapping(data.bgr.as_ptr().add(src_off), dst, row_bytes);
            }
        }
        Ok(())
    }
}

impl IWICBitmapFrameDecode_Impl for AureaFrame_Impl {
    fn GetMetadataQueryReader(&self) -> Result<IWICMetadataQueryReader> {
        Err(WINCODEC_ERR_UNSUPPORTEDOPERATION.into())
    }

    fn GetColorContexts(
        &self,
        _count: u32,
        _contexts: *mut Option<IWICColorContext>,
        actual: *mut u32,
    ) -> Result<()> {
        unsafe {
            if !actual.is_null() {
                *actual = 0;
            }
        }
        Ok(())
    }

    fn GetThumbnail(&self) -> Result<IWICBitmapSource> {
        Err(WINCODEC_ERR_CODECNOTHUMBNAIL.into())
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

#[implement(IWICBitmapDecoder)]
struct AureaDecoder {
    frame: Mutex<Option<std::sync::Arc<FrameData>>>,
}

impl AureaDecoder {
    fn new() -> Self {
        Self {
            frame: Mutex::new(None),
        }
    }
}

fn read_all(stream: &IStream) -> Result<Vec<u8>> {
    // WIC's pattern check may leave the stream mid-file; rewind first.
    unsafe { stream.Seek(0, STREAM_SEEK_SET, None)? };
    let mut bytes = Vec::<u8>::new();
    let mut chunk = [0u8; 65536];
    loop {
        let mut read = 0u32;
        let hr = unsafe {
            stream.Read(
                chunk.as_mut_ptr() as *mut core::ffi::c_void,
                chunk.len() as u32,
                Some(&mut read),
            )
        };
        if read > 0 {
            bytes.extend_from_slice(&chunk[..read as usize]);
        }
        if hr == S_FALSE || read == 0 {
            break;
        }
        hr.ok()?;
    }
    Ok(bytes)
}

impl IWICBitmapDecoder_Impl for AureaDecoder_Impl {
    fn QueryCapability(&self, stream: Ref<IStream>) -> Result<u32> {
        let stream = stream.ok()?;
        let mut magic = [0u8; 4];
        let mut read = 0u32;
        unsafe {
            stream.Seek(0, STREAM_SEEK_SET, None)?;
            let _ = stream.Read(
                magic.as_mut_ptr() as *mut core::ffi::c_void,
                4,
                Some(&mut read),
            );
        }
        if read == 4 && &magic == b"AUR2" {
            Ok((WICBitmapDecoderCapabilityCanDecodeAllImages.0
                | WICBitmapDecoderCapabilityCanDecodeSomeImages.0) as u32)
        } else {
            Ok(0)
        }
    }

    fn Initialize(&self, stream: Ref<IStream>, _options: WICDecodeOptions) -> Result<()> {
        let stream = stream.ok()?;
        let bytes = read_all(stream)?;
        dbg_log(&format!(
            "Initialize: {} bytes, head={:02X?}",
            bytes.len(),
            &bytes[..bytes.len().min(8)]
        ));
        let decoded = aurea_core::decode_aurea(&bytes).map_err(|e| {
            dbg_log(&format!("decode_aurea error: {e:?}"));
            Error::from(WINCODEC_ERR_BADIMAGE)
        })?;
        let (w, h) = (decoded.width as u32, decoded.height as u32);
        let mut bgr = decoded.rgb;
        // RGB -> BGR in place.
        for px in bgr.chunks_exact_mut(3) {
            px.swap(0, 2);
        }
        *self.frame.lock().unwrap() = Some(std::sync::Arc::new(FrameData {
            width: w,
            height: h,
            bgr,
        }));
        Ok(())
    }

    fn GetContainerFormat(&self) -> Result<GUID> {
        Ok(GUID_CONTAINER_AUREA)
    }

    fn GetDecoderInfo(&self) -> Result<IWICBitmapDecoderInfo> {
        unsafe {
            let factory: IWICImagingFactory =
                CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER)?;
            let info = factory.CreateComponentInfo(&CLSID_AUREA_DECODER)?;
            info.cast()
        }
    }

    fn CopyPalette(&self, _palette: Ref<IWICPalette>) -> Result<()> {
        Err(WINCODEC_ERR_PALETTEUNAVAILABLE.into())
    }

    fn GetMetadataQueryReader(&self) -> Result<IWICMetadataQueryReader> {
        Err(WINCODEC_ERR_UNSUPPORTEDOPERATION.into())
    }

    fn GetPreview(&self) -> Result<IWICBitmapSource> {
        Err(WINCODEC_ERR_UNSUPPORTEDOPERATION.into())
    }

    fn GetColorContexts(
        &self,
        _count: u32,
        _contexts: *mut Option<IWICColorContext>,
        actual: *mut u32,
    ) -> Result<()> {
        unsafe {
            if !actual.is_null() {
                *actual = 0;
            }
        }
        Ok(())
    }

    fn GetThumbnail(&self) -> Result<IWICBitmapSource> {
        Err(WINCODEC_ERR_CODECNOTHUMBNAIL.into())
    }

    fn GetFrameCount(&self) -> Result<u32> {
        Ok(1)
    }

    fn GetFrame(&self, index: u32) -> Result<IWICBitmapFrameDecode> {
        if index != 0 {
            return Err(WINCODEC_ERR_FRAMEMISSING.into());
        }
        let guard = self.frame.lock().unwrap();
        let data = guard
            .as_ref()
            .ok_or_else(|| Error::from(WINCODEC_ERR_NOTINITIALIZED))?
            .clone();
        Ok(AureaFrame { data }.into())
    }
}

// ---------------------------------------------------------------------------
// Class factory + exports
// ---------------------------------------------------------------------------

#[implement(IClassFactory)]
struct Factory;

impl IClassFactory_Impl for Factory_Impl {
    fn CreateInstance(
        &self,
        outer: Ref<IUnknown>,
        iid: *const GUID,
        object: *mut *mut core::ffi::c_void,
    ) -> Result<()> {
        if outer.is_some() {
            return Err(CLASS_E_NOAGGREGATION.into());
        }
        let unknown: IUnknown = AureaDecoder::new().into();
        unsafe { unknown.query(iid, object).ok() }
    }

    fn LockServer(&self, _lock: BOOL) -> Result<()> {
        Ok(())
    }
}

#[no_mangle]
extern "system" fn DllGetClassObject(
    rclsid: *const GUID,
    riid: *const GUID,
    ppv: *mut *mut core::ffi::c_void,
) -> HRESULT {
    unsafe {
        if rclsid.is_null() || *rclsid != CLSID_AUREA_DECODER {
            return CLASS_E_CLASSNOTAVAILABLE;
        }
        let factory: IClassFactory = Factory.into();
        factory.query(riid, ppv)
    }
}

#[no_mangle]
extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}
