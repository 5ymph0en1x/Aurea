/// Native AUREA viewer (.aur)
/// Scroll wheel zoom, drag pan, Escape to close, 0/Home to fit.

use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};


fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: aurea-viewer <file.aur>");
        std::process::exit(1);
    }
    let path = &args[1];

    // Decode aurea
    let data = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Read error {}: {}", path, e);
        std::process::exit(1);
    });
    let decoded = aurea_core::decode_aurea(&data).unwrap_or_else(|e| {
        eprintln!("Decode error: {}", e);
        std::process::exit(1);
    });

    let iw = decoded.width;
    let ih = decoded.height;

    // RGB -> u32 (0x00RRGGBB for minifb)
    let image: Vec<u32> = (0..iw * ih)
        .map(|i| {
            let r = decoded.rgb[i * 3] as u32;
            let g = decoded.rgb[i * 3 + 1] as u32;
            let b = decoded.rgb[i * 3 + 2] as u32;
            (r << 16) | (g << 8) | b
        })
        .collect();

    // Initial window size (max 1600x900 or image size)
    let fit_ratio = (1600.0 / iw as f64).min(900.0 / ih as f64).min(1.0);
    let win_w = (iw as f64 * fit_ratio) as usize;
    let win_h = (ih as f64 * fit_ratio) as usize;

    let filename = std::path::Path::new(path)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());
    let title = format!("AUREA Viewer \u{2014} {} ({}x{})", filename, iw, ih);

    let mut window = Window::new(
        &title,
        win_w,
        win_h,
        WindowOptions {
            resize: true,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("Window error: {}", e);
        std::process::exit(1);
    });

    window.set_target_fps(60);

    // State: zoom and center (in image coordinates)
    let mut zoom = 1.0f64;
    let mut cx = iw as f64 / 2.0;
    let mut cy = ih as f64 / 2.0;
    let mut dragging = false;
    let mut last_mx = 0.0f32;
    let mut last_my = 0.0f32;

    let mut dw = win_w;
    let mut dh = win_h;
    let mut display = vec![0u32; dw * dh];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Resize
        let (new_w, new_h) = window.get_size();
        if new_w != dw || new_h != dh {
            dw = new_w.max(1);
            dh = new_h.max(1);
            display.resize(dw * dh, 0);
        }

        // Scroll = zoom (centered on mouse)
        if let Some((_, scroll_y)) = window.get_scroll_wheel() {
            if scroll_y.abs() > 0.01 {
                let factor = if scroll_y > 0.0 { 1.15 } else { 1.0 / 1.15 };
                let fit_scale = (dw as f64 / iw as f64).min(dh as f64 / ih as f64);
                let old_scale = fit_scale * zoom;

                if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard) {
                    let ix = (mx as f64 - dw as f64 / 2.0) / old_scale + cx;
                    let iy = (my as f64 - dh as f64 / 2.0) / old_scale + cy;
                    zoom = (zoom * factor).clamp(0.1, 50.0);
                    let new_scale = fit_scale * zoom;
                    cx = ix - (mx as f64 - dw as f64 / 2.0) / new_scale;
                    cy = iy - (my as f64 - dh as f64 / 2.0) / new_scale;
                } else {
                    zoom = (zoom * factor).clamp(0.1, 50.0);
                }
            }
        }

        // Drag = pan
        if window.get_mouse_down(MouseButton::Left) {
            if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard) {
                if dragging {
                    let fit_scale = (dw as f64 / iw as f64).min(dh as f64 / ih as f64);
                    let scale = fit_scale * zoom;
                    cx -= (mx - last_mx) as f64 / scale;
                    cy -= (my - last_my) as f64 / scale;
                }
                last_mx = mx;
                last_my = my;
                dragging = true;
            }
        } else {
            dragging = false;
        }

        // Reset: 0, Home or F
        if window.is_key_down(Key::Key0)
            || window.is_key_down(Key::Home)
            || window.is_key_down(Key::F)
        {
            zoom = 1.0;
            cx = iw as f64 / 2.0;
            cy = ih as f64 / 2.0;
        }

        render(&mut display, dw, dh, &image, iw, ih, zoom, cx, cy);
        window.update_with_buffer(&display, dw, dh).unwrap();
    }
}

fn render(
    display: &mut [u32],
    dw: usize,
    dh: usize,
    image: &[u32],
    iw: usize,
    ih: usize,
    zoom: f64,
    cx: f64,
    cy: f64,
) {
    let fit_scale = (dw as f64 / iw as f64).min(dh as f64 / ih as f64);
    let scale = fit_scale * zoom;
    let half_dw = dw as f64 / 2.0;
    let half_dh = dh as f64 / 2.0;
    let inv_scale = 1.0 / scale;

    let bg = 0x00202020u32;

    for dy in 0..dh {
        let iy_f = (dy as f64 - half_dh) * inv_scale + cy;
        let iy = iy_f.floor() as i64;

        if iy < 0 || iy >= ih as i64 {
            let start = dy * dw;
            display[start..start + dw].fill(bg);
            continue;
        }

        let img_row = iy as usize * iw;

        for dx in 0..dw {
            let ix_f = (dx as f64 - half_dw) * inv_scale + cx;
            let ix = ix_f.floor() as i64;

            display[dy * dw + dx] = if ix >= 0 && ix < iw as i64 {
                image[img_row + ix as usize]
            } else {
                bg
            };
        }
    }
}
