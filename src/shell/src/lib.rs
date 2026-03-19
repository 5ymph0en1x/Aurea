/// Windows shell extension for .aur
/// - IThumbnailProvider : native previews in Explorer
/// - IWICBitmapDecoder : WIC support (Photos, Paint, etc.)

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::Graphics::Imaging::*;
use windows::Win32::System::Com::*;
use windows::Win32::UI::Shell::*;

// CLSIDs
const CLSID_AUREA_THUMBNAIL: GUID =
    GUID::from_u128(0x267A0E00_C0DE_4ABC_9DEF_000000267011);
const CLSID_AUREA_WIC_DECODER: GUID =
    GUID::from_u128(0x267A0E00_C0DE_4ABC_9DEF_000000267012);
const GUID_AUREA_CONTAINER: GUID =
    GUID::from_u128(0x267A0E00_C0DE_4ABC_9DEF_000000267013);

const CLASS_E_CLASSNOTAVAILABLE: HRESULT = HRESULT(0x80040111_u32 as i32);
const CLASS_E_NOAGGREGATION: HRESULT = HRESULT(0x80040110_u32 as i32);
const WINCODEC_ERR_PALETTEUNAVAILABLE: HRESULT = HRESULT(0x88982F45_u32 as i32);
const WINCODEC_ERR_CODECNOTHUMBNAIL: HRESULT = HRESULT(0x88982F44_u32 as i32);
const WINCODEC_ERR_UNSUPPORTEDOPERATION: HRESULT = HRESULT(0x88982F81_u32 as i32);

// CLSID WICImagingFactory (cacaf262-9370-4615-a13b-9f5539da4c0a)
const CLSID_WIC_IMAGING_FACTORY: GUID =
    GUID::from_u128(0xcacaf262_9370_4615_a13b_9f5539da4c0a);

// Magic bytes AURA: 0x41 0x55 0x52 0x41
const AUREA_MAGIC: &[u8; 4] = b"AURA";
// Magic bytes AUR2: 0x41 0x55 0x52 0x32
const AUR2_MAGIC: &[u8; 4] = b"AUR2";

// Raw vtable layout of IUnknown to call QueryInterface via COM.
#[repr(C)]
struct IUnknownVtblRaw {
    query_interface: unsafe extern "system" fn(
        *mut core::ffi::c_void,
        *const GUID,
        *mut *mut core::ffi::c_void,
    ) -> HRESULT,
    _add_ref: unsafe extern "system" fn(*mut core::ffi::c_void) -> u32,
    _release: unsafe extern "system" fn(*mut core::ffi::c_void) -> u32,
}

/// Calls QueryInterface on an IUnknown via the raw COM vtable.
unsafe fn raw_query_interface(
    unknown: &IUnknown,
    iid: *const GUID,
    ppv: *mut *mut core::ffi::c_void,
) -> HRESULT {
    let raw = unknown.as_raw();
    unsafe {
        let vtbl = &**(raw as *const *const IUnknownVtblRaw);
        (vtbl.query_interface)(raw, iid, ppv)
    }
}

/// Reads all data from an IStream into a Vec.
unsafe fn read_istream(stream: &IStream) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    let mut buf = [0u8; 65536];
    loop {
        let mut read = 0u32;
        let _ = unsafe {
            stream.Read(
                buf.as_mut_ptr() as *mut core::ffi::c_void,
                buf.len() as u32,
                Some(&mut read),
            )
        };
        if read == 0 {
            break;
        }
        data.extend_from_slice(&buf[..read as usize]);
    }
    if data.is_empty() {
        Err(Error::from(E_FAIL))
    } else {
        Ok(data)
    }
}

// ======================================================================
// Thumbnail Provider (via IInitializeWithItem to receive the path)
// ======================================================================

#[implement(IThumbnailProvider, IInitializeWithItem)]
struct AureaThumbnailProvider {
    data: Mutex<Vec<u8>>,
}

impl AureaThumbnailProvider {
    fn new() -> Self {
        Self {
            data: Mutex::new(Vec::new()),
        }
    }
}

impl IInitializeWithItem_Impl for AureaThumbnailProvider_Impl {
    fn Initialize(&self, psi: Option<&IShellItem>, _grfmode: u32) -> Result<()> {
        let item = psi.ok_or(Error::from(E_INVALIDARG))?;

        // Get the file path from the shell item
        let path_pwstr = unsafe { item.GetDisplayName(SIGDN_FILESYSPATH)? };
        let path = unsafe { path_pwstr.to_string().map_err(|_| Error::from(E_FAIL))? };
        unsafe { CoTaskMemFree(Some(path_pwstr.0 as *const _)) };

        // Read the file
        let file_data = std::fs::read(&path).map_err(|_| Error::from(E_FAIL))?;
        *self.data.lock().unwrap() = file_data;
        Ok(())
    }
}

impl IThumbnailProvider_Impl for AureaThumbnailProvider_Impl {
    fn GetThumbnail(
        &self,
        cx: u32,
        phbmp: *mut HBITMAP,
        pdwalpha: *mut WTS_ALPHATYPE,
    ) -> Result<()> {
        // catch_unwind to prevent a Rust panic from crashing explorer.exe
        catch_unwind(AssertUnwindSafe(|| {
            self.get_thumbnail_inner(cx, phbmp, pdwalpha)
        }))
        .unwrap_or(Err(Error::from(E_FAIL)))
    }
}

impl AureaThumbnailProvider_Impl {
    fn get_thumbnail_inner(
        &self,
        cx: u32,
        phbmp: *mut HBITMAP,
        pdwalpha: *mut WTS_ALPHATYPE,
    ) -> Result<()> {
        let data = self.data.lock().unwrap();
        if data.is_empty() {
            return Err(Error::from(E_FAIL));
        }

        // Decode AUREA -> RGB
        let decoded = aurea_core::decode_aurea(&data).map_err(|_| Error::from(E_FAIL))?;

        let img = image::RgbImage::from_raw(
            decoded.width as u32,
            decoded.height as u32,
            decoded.rgb,
        )
        .ok_or(Error::from(E_FAIL))?;

        // Proportional resize (largest dimension = cx)
        let max_dim = decoded.width.max(decoded.height) as f64;
        let scale = cx as f64 / max_dim;
        let tw = ((decoded.width as f64 * scale) as u32).max(1);
        let th = ((decoded.height as f64 * scale) as u32).max(1);

        let thumb =
            image::imageops::resize(&img, tw, th, image::imageops::FilterType::Triangle);

        let tw_i = thumb.width() as i32;
        let th_i = thumb.height() as i32;

        // Create a 32-bit BGRA DIB section
        let bmi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: tw_i,
                biHeight: -th_i, // top-down
                biPlanes: 1,
                biBitCount: 32,
                biCompression: 0, // BI_RGB
                ..Default::default()
            },
            ..Default::default()
        };

        let mut bits: *mut core::ffi::c_void = core::ptr::null_mut();
        let hbmp = unsafe {
            CreateDIBSection(
                HDC::default(),
                &bmi,
                DIB_RGB_COLORS,
                &mut bits,
                HANDLE::default(),
                0,
            )
        }
        .map_err(|_| Error::from(E_FAIL))?;

        // Copy RGB -> BGRA
        let pixel_data = unsafe {
            std::slice::from_raw_parts_mut(bits as *mut u8, (tw_i * th_i * 4) as usize)
        };
        for (i, pixel) in thumb.pixels().enumerate() {
            let o = i * 4;
            pixel_data[o] = pixel[2];     // B
            pixel_data[o + 1] = pixel[1]; // G
            pixel_data[o + 2] = pixel[0]; // R
            pixel_data[o + 3] = 255;      // A
        }

        unsafe {
            *phbmp = hbmp;
            *pdwalpha = WTSAT_RGB;
        }

        Ok(())
    }
}

// ======================================================================
// WIC Decoder (IWICBitmapDecoder)
// ======================================================================

struct WicDecoderState {
    width: u32,
    height: u32,
    bgra_data: Arc<Vec<u8>>,
    initialized: bool,
}

#[implement(IWICBitmapDecoder)]
struct AureaWicDecoder {
    state: Mutex<WicDecoderState>,
}

impl AureaWicDecoder {
    fn new() -> Self {
        Self {
            state: Mutex::new(WicDecoderState {
                width: 0,
                height: 0,
                bgra_data: Arc::new(Vec::new()),
                initialized: false,
            }),
        }
    }
}

impl IWICBitmapDecoder_Impl for AureaWicDecoder_Impl {
    fn QueryCapability(&self, pistream: Option<&IStream>) -> Result<u32> {
        let stream = pistream.ok_or(Error::from(E_INVALIDARG))?;
        let mut magic = [0u8; 4];
        let mut read = 0u32;
        unsafe {
            let _ = stream.Read(
                magic.as_mut_ptr() as *mut core::ffi::c_void,
                4,
                Some(&mut read),
            );
            // Rewind the stream
            let _ = stream.Seek(0, STREAM_SEEK_SET, None);
        }
        if read < 4 || (&magic != AUREA_MAGIC && &magic != AUR2_MAGIC) {
            return Ok(0);
        }
        // WICBitmapDecoderCapabilityCanDecodeAllImages
        Ok(2)
    }

    fn Initialize(
        &self,
        pistream: Option<&IStream>,
        _cacheoptions: WICDecodeOptions,
    ) -> Result<()> {
        // catch_unwind to prevent a Rust panic from crashing explorer.exe
        catch_unwind(AssertUnwindSafe(|| {
            let stream = pistream.ok_or(Error::from(E_INVALIDARG))?;
            let raw_data = unsafe { read_istream(stream)? };

            let decoded = aurea_core::decode_aurea(&raw_data).map_err(|_| Error::from(E_FAIL))?;

            // RGB -> BGRA
            let npix = decoded.width * decoded.height;
            let mut bgra = vec![0u8; npix * 4];
            for i in 0..npix {
                bgra[i * 4] = decoded.rgb[i * 3 + 2];     // B
                bgra[i * 4 + 1] = decoded.rgb[i * 3 + 1]; // G
                bgra[i * 4 + 2] = decoded.rgb[i * 3];     // R
                bgra[i * 4 + 3] = 255;                     // A
            }

            let mut state = self.state.lock().unwrap();
            state.width = decoded.width as u32;
            state.height = decoded.height as u32;
            state.bgra_data = Arc::new(bgra);
            state.initialized = true;
            Ok(())
        }))
        .unwrap_or(Err(Error::from(E_FAIL)))
    }

    fn GetContainerFormat(&self) -> Result<GUID> {
        Ok(GUID_AUREA_CONTAINER)
    }

    fn GetDecoderInfo(&self) -> Result<IWICBitmapDecoderInfo> {
        unsafe {
            let factory: IWICImagingFactory = CoCreateInstance(
                &CLSID_WIC_IMAGING_FACTORY,
                None,
                CLSCTX_INPROC_SERVER,
            )?;
            let info = factory.CreateComponentInfo(&CLSID_AUREA_WIC_DECODER)?;
            info.cast()
        }
    }

    fn CopyPalette(&self, _pipalette: Option<&IWICPalette>) -> Result<()> {
        Err(Error::from(WINCODEC_ERR_PALETTEUNAVAILABLE))
    }

    fn GetMetadataQueryReader(&self) -> Result<IWICMetadataQueryReader> {
        Err(Error::from(WINCODEC_ERR_UNSUPPORTEDOPERATION))
    }

    fn GetPreview(&self) -> Result<IWICBitmapSource> {
        Err(Error::from(WINCODEC_ERR_UNSUPPORTEDOPERATION))
    }

    fn GetColorContexts(
        &self,
        _ccount: u32,
        _ppicolorcontexts: *mut Option<IWICColorContext>,
        pcactualcount: *mut u32,
    ) -> Result<()> {
        if !pcactualcount.is_null() {
            unsafe { *pcactualcount = 0 };
        }
        Ok(())
    }

    fn GetThumbnail(&self) -> Result<IWICBitmapSource> {
        Err(Error::from(WINCODEC_ERR_CODECNOTHUMBNAIL))
    }

    fn GetFrameCount(&self) -> Result<u32> {
        Ok(1)
    }

    fn GetFrame(&self, index: u32) -> Result<IWICBitmapFrameDecode> {
        if index != 0 {
            return Err(Error::from(E_INVALIDARG));
        }
        let state = self.state.lock().unwrap();
        if !state.initialized {
            return Err(Error::from(E_FAIL));
        }
        let frame = AureaWicFrame {
            width: state.width,
            height: state.height,
            bgra_data: Arc::clone(&state.bgra_data),
        };
        Ok(frame.into())
    }
}

// ======================================================================
// WIC Frame (IWICBitmapFrameDecode extends IWICBitmapSource)
// ======================================================================

#[implement(IWICBitmapFrameDecode)]
struct AureaWicFrame {
    width: u32,
    height: u32,
    bgra_data: Arc<Vec<u8>>,
}

impl IWICBitmapSource_Impl for AureaWicFrame_Impl {
    fn GetSize(&self, puiwidth: *mut u32, puiheight: *mut u32) -> Result<()> {
        unsafe {
            *puiwidth = self.width;
            *puiheight = self.height;
        }
        Ok(())
    }

    fn GetPixelFormat(&self) -> Result<GUID> {
        Ok(GUID_WICPixelFormat32bppBGRA)
    }

    fn GetResolution(&self, pdpix: *mut f64, pdpiy: *mut f64) -> Result<()> {
        unsafe {
            *pdpix = 96.0;
            *pdpiy = 96.0;
        }
        Ok(())
    }

    fn CopyPalette(&self, _pipalette: Option<&IWICPalette>) -> Result<()> {
        Err(Error::from(WINCODEC_ERR_PALETTEUNAVAILABLE))
    }

    fn CopyPixels(
        &self,
        prc: *const WICRect,
        cbstride: u32,
        cbbuffersize: u32,
        pbbuffer: *mut u8,
    ) -> Result<()> {
        let (x, y, w, h) = if prc.is_null() {
            (0i32, 0i32, self.width as i32, self.height as i32)
        } else {
            let rc = unsafe { &*prc };
            (rc.X, rc.Y, rc.Width, rc.Height)
        };

        if x < 0
            || y < 0
            || w <= 0
            || h <= 0
            || (x + w) as u32 > self.width
            || (y + h) as u32 > self.height
        {
            return Err(Error::from(E_INVALIDARG));
        }

        let src_stride = self.width as usize * 4;
        let dst_stride = cbstride as usize;
        let row_bytes = w as usize * 4;

        let dst =
            unsafe { std::slice::from_raw_parts_mut(pbbuffer, cbbuffersize as usize) };

        for row in 0..h as usize {
            let src_off = (y as usize + row) * src_stride + x as usize * 4;
            let dst_off = row * dst_stride;
            dst[dst_off..dst_off + row_bytes]
                .copy_from_slice(&self.bgra_data[src_off..src_off + row_bytes]);
        }

        Ok(())
    }
}

impl IWICBitmapFrameDecode_Impl for AureaWicFrame_Impl {
    fn GetMetadataQueryReader(&self) -> Result<IWICMetadataQueryReader> {
        Err(Error::from(WINCODEC_ERR_UNSUPPORTEDOPERATION))
    }

    fn GetColorContexts(
        &self,
        _ccount: u32,
        _ppicolorcontexts: *mut Option<IWICColorContext>,
        pcactualcount: *mut u32,
    ) -> Result<()> {
        if !pcactualcount.is_null() {
            unsafe { *pcactualcount = 0 };
        }
        Ok(())
    }

    fn GetThumbnail(&self) -> Result<IWICBitmapSource> {
        Err(Error::from(WINCODEC_ERR_CODECNOTHUMBNAIL))
    }
}

// ======================================================================
// Class Factory
// ======================================================================

#[implement(IClassFactory)]
struct AureaClassFactory {
    clsid: GUID,
}

impl IClassFactory_Impl for AureaClassFactory_Impl {
    fn CreateInstance(
        &self,
        outer: Option<&IUnknown>,
        iid: *const GUID,
        obj: *mut *mut core::ffi::c_void,
    ) -> Result<()> {
        if outer.is_some() {
            return Err(Error::from(CLASS_E_NOAGGREGATION));
        }

        let unknown: IUnknown = if self.clsid == CLSID_AUREA_THUMBNAIL {
            AureaThumbnailProvider::new().into()
        } else if self.clsid == CLSID_AUREA_WIC_DECODER {
            AureaWicDecoder::new().into()
        } else {
            return Err(Error::from(CLASS_E_CLASSNOTAVAILABLE));
        };

        unsafe { raw_query_interface(&unknown, iid, obj).ok() }
    }

    fn LockServer(&self, _lock: BOOL) -> Result<()> {
        Ok(())
    }
}

// ======================================================================
// DLL Exports
// ======================================================================

#[unsafe(no_mangle)]
unsafe extern "system" fn DllGetClassObject(
    clsid: *const GUID,
    iid: *const GUID,
    result: *mut *mut core::ffi::c_void,
) -> HRESULT {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        if clsid.is_null() || iid.is_null() || result.is_null() {
            return E_POINTER;
        }
        *result = core::ptr::null_mut();

        let cls = *clsid;
        if cls != CLSID_AUREA_THUMBNAIL && cls != CLSID_AUREA_WIC_DECODER {
            return CLASS_E_CLASSNOTAVAILABLE;
        }

        let factory: IUnknown = AureaClassFactory { clsid: cls }.into();
        raw_query_interface(&factory, iid, result)
    }))
    .unwrap_or(E_FAIL)
}

#[unsafe(no_mangle)]
extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}
