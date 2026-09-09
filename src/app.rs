//! The optimizer UI: pick two slices on the projection view, run a test
//! MBIRJAX reconstruction of those two slices, tune the parameters, repeat —
//! then save the parameters into the checkpoint HDF5.

use crate::recon::{
    BAND, MbirjaxParams, ReconJob, TEST_REBIN_FACTORS, center_from_offset, checkpoint_geometry,
    max_test_rebin, offset_from_center, save_params, test_band_fits, tilt_tool_records,
};
use ct_reconstruction::combine::{LoadJob, LoadedStack};
use ct_reconstruction::rebin::rebinned_size;
use egui::{Color32, RichText};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

/// The standalone tilt & center-of-rotation tool (same binary the main
/// application launches from its pre-processing screen).
pub const TILT_COR_BIN: &str =
    "/SNS/VENUS/shared/software/git/rust_tilt_center_of_rotation/target/release/tilt_center_of_rotation";

/// The checkpoint's center of rotation, or the detector center when it
/// carries none (what the seeding rule falls back to).
fn checkpoint_center(stack: &LoadedStack, width: usize) -> (f64, bool) {
    match stack.center_of_rotation {
        Some(c) => (c, true),
        None => (center_from_offset(0.0, width), false),
    }
}

/// The tilt corrections recorded in the checkpoint, one line each: the
/// in-pipeline step of the main application (`tilt_correction`) and the
/// standalone tool's JSON records (`tilt_center_of_rotation`).
fn tilt_summary_lines(stack: &LoadedStack) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some((_, desc)) = stack
        .metadata
        .iter()
        .find(|(name, _)| name == "tilt_correction")
    {
        // "tilt -2.0511 deg, axis shift 191 px, edge-padded (…)"
        let mut words = desc.split_whitespace();
        let deg = words.nth(1).and_then(|v| v.parse::<f64>().ok());
        let shift = desc
            .split("shift")
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|v| v.parse::<i64>().ok());
        lines.push(match (deg, shift) {
            (Some(deg), Some(shift)) => format!(
                "pre-processing step: tilt {deg:+.4}° corrected, axis shifted {shift} px"
            ),
            _ => format!("pre-processing step: {desc}"),
        });
    }
    for record in tilt_tool_records(&stack.metadata) {
        let doc: serde_json::Value = match serde_json::from_str(&record) {
            Ok(doc) => doc,
            Err(_) => continue,
        };
        let get = |key: &str| doc.get(key).and_then(|v| v.as_f64());
        let text = |key: &str| doc.get(key).and_then(|v| v.as_str()).unwrap_or("");
        lines.push(format!(
            "tilt & center-of-rotation tool: tilt {:+.4}° corrected, center of rotation              {:.2} px ({}{}{})",
            get("corrected_tilt_deg").unwrap_or(0.0),
            get("center_of_rotation").unwrap_or(0.0),
            text("method"),
            if text("date").is_empty() { "" } else { ", " },
            text("date"),
        ));
    }
    lines
}

/// SHA-256 of the advanced-parameters password (same gate as the marimo
/// notebook and the main application's admin section).
const ADVANCED_PASSWORD_SHA256: &str =
    "b8b22aedc372aa891df895be9a7626e6d9ddc6d39ba85d202ca68de8c52ad782";

/// Imaging team logo (same asset and placement as the other rust
/// applications) and the official MBIRJAX logo, both embedded in the binary
/// and shown at the right end of the toolbar.
const IMAGING_LOGO_BYTES: &[u8] = include_bytes!("../logos/ImagingLogo.png");
const MBIRJAX_LOGO_BYTES: &[u8] = include_bytes!("../logos/mbirjax_logo.png");
const LOGO_MAX_HEIGHT: f32 = 36.0;

fn load_logo(ctx: &egui::Context, name: &str, bytes: &[u8]) -> Option<egui::TextureHandle> {
    let img = image::load_from_memory(bytes).ok()?;
    let rgba = img.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let pixels = rgba.into_raw();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    Some(ctx.load_texture(name, color_image, egui::TextureOptions::LINEAR))
}

/// Drag speed for a value field: holding Shift while dragging (or using
/// the arrow keys) moves the value 10× FASTER, as in Adobe's tools. egui's
/// built-in shift behavior divides the speed by 10, so ×100 nets ×10 —
/// the same convention as the tilt & center-of-rotation tool.
fn drag_speed(ui: &egui::Ui, base: f64) -> f64 {
    if ui.input(|i| i.modifiers.shift_only()) {
        base * 100.0
    } else {
        base
    }
}

fn password_matches(input: &str) -> bool {
    let digest = Sha256::digest(input.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    hex == ADVANCED_PASSWORD_SHA256
}

/// One entry of the run history.
struct HistoryEntry {
    params: MbirjaxParams,
    top_slice: usize,
    bottom_slice: usize,
    /// n×n rebin the test data was reconstructed at (1 = the checkpoint's
    /// own resolution).
    test_rebin: usize,
    seconds: f64,
    /// Downsampled copies of the two reconstructed slices, kept so past runs
    /// can be previewed side by side when choosing the parameters.
    thumb_size: (usize, usize),
    top_thumb: Vec<f32>,
    bottom_thumb: Vec<f32>,
    tex: Option<(egui::TextureHandle, egui::TextureHandle)>,
}

pub struct OptimizerApp {
    called_from_app: bool,

    stack: Option<Arc<LoadedStack>>,
    load_job: Option<LoadJob>,
    load_error: Option<String>,

    // Slice selection on the projection view.
    top_slice: usize,
    bottom_slice: usize,
    preview_frame: usize,
    preview_tex: Option<((usize, usize), egui::TextureHandle)>,

    // Parameters.
    params: MbirjaxParams,
    advanced_unlocked: bool,
    advanced_password: String,
    advanced_error: Option<String>,

    // Test reconstruction.
    /// n×n rebin of the test bands before reconstructing them (1 = none):
    /// a shortcut for the test only — the saved parameters keep the
    /// checkpoint's pixel units.
    test_rebin: usize,
    recon_job: Option<ReconJob>,
    /// Last result: (height, width, top slice, bottom slice, seconds).
    result: Option<(usize, usize, Vec<f32>, Vec<f32>, f64)>,
    /// The rebin factor of the last result (for its caption).
    result_rebin: usize,
    result_tex: Option<(egui::TextureHandle, egui::TextureHandle)>,
    recon_error: Option<String>,
    history: Vec<HistoryEntry>,

    // Saving into the HDF5.
    save_status: Option<Result<String, String>>,

    // The standalone tilt & center-of-rotation tool, run on the checkpoint
    // itself; when it applied a correction the file is reloaded.
    tilt_tool_job: Option<Receiver<Result<(), String>>>,
    tilt_tool_note: Option<Result<String, String>>,
    /// Set while reloading after the tool changed the file: seed the
    /// detector-channel offset from the new center of rotation instead of
    /// the (now stale) saved parameters.
    reseed_offset_on_load: bool,

    /// Imaging team + tool logos, loaded into textures on the first frame.
    logo_tex: Option<Vec<egui::TextureHandle>>,

    status: String,
}

impl OptimizerApp {
    pub fn new(input: Option<PathBuf>, called_from_app: bool) -> Self {
        let mut app = Self {
            called_from_app,
            stack: None,
            load_job: None,
            load_error: None,
            top_slice: 0,
            bottom_slice: 0,
            preview_frame: 0,
            preview_tex: None,
            params: MbirjaxParams::default(),
            advanced_unlocked: false,
            advanced_password: String::new(),
            advanced_error: None,
            test_rebin: 1,
            recon_job: None,
            result: None,
            result_rebin: 1,
            result_tex: None,
            recon_error: None,
            history: Vec::new(),
            save_status: None,
            tilt_tool_job: None,
            tilt_tool_note: None,
            reseed_offset_on_load: false,
            logo_tex: None,
            status: "Open a pre-processed checkpoint HDF5 to begin.".to_owned(),
        };
        if let Some(path) = input {
            app.start_load(path);
        }
        app
    }

    fn start_load(&mut self, path: PathBuf) {
        self.status = format!("Loading {}…", path.display());
        self.load_error = None;
        self.load_job = Some(LoadJob::start(path));
    }

    fn adopt_stack(&mut self, stack: LoadedStack) {
        let h = stack.sample.first().map(|p| p.height).unwrap_or(1);
        self.top_slice = h / 3;
        self.bottom_slice = (2 * h) / 3;
        self.preview_frame = 0;
        self.preview_tex = None;
        self.result = None;
        self.result_tex = None;
        self.history.clear();
        self.save_status = None;
        self.params = MbirjaxParams::from_stack(&stack);
        let restored = stack
            .metadata
            .iter()
            .any(|(name, _)| name == "mbirjax_config");
        let (w, views) = stack
            .sample
            .first()
            .map(|p| (p.width, stack.sample.len()))
            .unwrap_or((0, 0));
        self.test_rebin = crate::recon::default_test_rebin(w, h, views);
        let mut note = if restored {
            " — saved MBIRJAX parameters restored"
        } else {
            ""
        }
        .to_owned();
        if std::mem::take(&mut self.reseed_offset_on_load)
            && let Some(cor) = stack.center_of_rotation
        {
            self.params.det_channel_offset = offset_from_center(cor, w);
            note = format!(
                " — center of rotation offset re-seeded from the corrected file ({cor:.2} px); save the parameters to keep it"
            );
        }
        self.status = format!(
            "{} — {} projections{note}",
            stack.path.display(),
            stack.sample.len(),
        );
        self.stack = Some(Arc::new(stack));
    }

    /// Stride-downsample a w×h image so its longest side is at most `max`.
    fn downsample(values: &[f32], w: usize, h: usize, max: usize) -> (Vec<f32>, usize, usize) {
        let stride = (w.max(h) / max).max(1);
        let (sw, sh) = (w.div_ceil(stride), h.div_ceil(stride));
        let mut small = Vec::with_capacity(sw * sh);
        for y in (0..h).step_by(stride) {
            for x in (0..w).step_by(stride) {
                small.push(values[y * w + x]);
            }
        }
        (small, sw, sh)
    }

    fn grayscale_texture(
        ctx: &egui::Context,
        name: &str,
        values: &[f32],
        w: usize,
        h: usize,
    ) -> egui::TextureHandle {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for v in values {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let span = (hi - lo).max(1e-6);
        let pixels: Vec<Color32> = values
            .iter()
            .map(|v| Color32::from_gray((((v - lo) / span) * 255.0) as u8))
            .collect();
        ctx.load_texture(
            name.to_owned(),
            egui::ColorImage {
                size: [w, h],
                source_size: egui::vec2(w as f32, h as f32),
                pixels,
            },
            egui::TextureOptions::LINEAR,
        )
    }

    fn projection_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(stack) = self.stack.clone() else {
            return;
        };
        let Some(first) = stack.sample.first() else {
            return;
        };
        let (w, h, n) = (first.width, first.height, stack.sample.len());
        ui.label(RichText::new("Slice selection").strong());
        self.preview_frame = self.preview_frame.min(n - 1);
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.preview_frame, 0..=n - 1).text("projection"));
            let p = &stack.sample[self.preview_frame];
            ui.label(
                RichText::new(match p.angle_deg {
                    Some(a) => format!("{a:.2}°"),
                    None => String::new(),
                })
                .weak()
                .size(11.0),
            );
        });
        ui.add(egui::Slider::new(&mut self.top_slice, 0..=h - 1).text("top slice (red)"));
        ui.add(
            egui::Slider::new(&mut self.bottom_slice, 0..=h - 1).text("bottom slice (cyan)"),
        );

        let key = (Arc::as_ptr(&stack) as usize, self.preview_frame);
        if self.preview_tex.as_ref().map(|(k, _)| *k) != Some(key) {
            let p = &stack.sample[self.preview_frame];
            let (small, sw, sh) = Self::downsample(&p.mean, p.width, p.height, 512);
            let tex = Self::grayscale_texture(ctx, "projection", &small, sw, sh);
            self.preview_tex = Some((key, tex));
        }
        if let Some((_, tex)) = &self.preview_tex {
            let size = tex.size_vec2();
            let scale = (420.0 / size.x.max(size.y)).min(2.0);
            // Allocate exactly the drawn size: inside `ui.columns` (a
            // justified layout) an Image widget's response rect spans the
            // whole column, which would put the overlay lines off the image.
            let (rect, _response) =
                ui.allocate_exact_size(size * scale, egui::Sense::hover());
            egui::Image::from_texture(tex).paint_at(ui, rect);
            let painter = ui.painter_at(rect);
            let y_of =
                |row: usize| rect.top() + (row as f32 / h as f32) * rect.height();
            for (row, color) in [
                (self.top_slice, Color32::from_rgb(255, 90, 80)),
                (self.bottom_slice, Color32::from_rgb(110, 230, 230)),
            ] {
                painter.line_segment(
                    [
                        egui::pos2(rect.left(), y_of(row)),
                        egui::pos2(rect.right(), y_of(row)),
                    ],
                    egui::Stroke::new(1.5, color),
                );
            }
            // The center of rotation: the checkpoint's (dashed orange) and,
            // when the offset was changed here, the current one (solid green).
            let x_of = |col: f64| rect.left() + ((col + 0.5) / w as f64) as f32 * rect.width();
            let (file_cor, _) = checkpoint_center(&stack, w);
            let current_cor = center_from_offset(self.params.det_channel_offset, w);
            let moved = (current_cor - file_cor).abs() > 1e-6;
            painter.add(egui::Shape::dashed_line(
                &[
                    egui::pos2(x_of(file_cor), rect.top()),
                    egui::pos2(x_of(file_cor), rect.bottom()),
                ],
                egui::Stroke::new(1.5, Color32::from_rgb(255, 170, 40)),
                6.0,
                4.0,
            ));
            if moved {
                painter.line_segment(
                    [
                        egui::pos2(x_of(current_cor), rect.top()),
                        egui::pos2(x_of(current_cor), rect.bottom()),
                    ],
                    egui::Stroke::new(1.5, Color32::from_rgb(120, 230, 120)),
                );
            }
        }
        self.center_readout(ui, &stack, w);
        ui.add_space(4.0);

        // The resolution the test runs at.
        let max_rebin = max_test_rebin(h);
        self.test_rebin = self.test_rebin.clamp(1, max_rebin);
        let rebin_label = |n: usize| -> String {
            let (rw, rh) = rebinned_size(w, h, n);
            if n <= 1 {
                format!("full resolution ({w} px wide)")
            } else {
                format!("{n}x{n} rebinned ({rw}x{rh} px)")
            }
        };
        ui.horizontal(|ui| {
            ui.label(RichText::new("Test data:").strong());
            egui::ComboBox::from_id_salt("test_rebin")
                .selected_text(rebin_label(self.test_rebin))
                .show_ui(ui, |ui| {
                    for f in TEST_REBIN_FACTORS.iter().copied().filter(|f| *f <= max_rebin) {
                        ui.selectable_value(&mut self.test_rebin, f, rebin_label(f));
                    }
                })
                .response
                .on_hover_text(
                    "reconstruct the test slices from n×n block-averaged projections: \
                     smaller sinograms reconstruct much faster (and fit the GPU when \
                     the full width does not), at a coarser resolution. A shortcut \
                     for the test only — the parameters are saved in the checkpoint's \
                     pixel units and the full reconstruction runs on the checkpoint \
                     as is.",
                );
            let (rw, _) = rebinned_size(w, h, self.test_rebin);
            match test_band_fits(rw, n) {
                Some(true) => {
                    ui.label(RichText::new("✔ fits one GPU").weak().size(11.0));
                }
                Some(false) => {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!(
                            "⚠ a {BAND}-slice band {rw} px wide does not fit one GPU — \
                             choose a larger rebin"
                        ),
                    );
                }
                None => {}
            }
        });
        ui.label(
            RichText::new(format!(
                "the test reconstruction runs on the two marked slices only (a {BAND}-slice \
                 band around each{})",
                if self.test_rebin > 1 {
                    format!(", {0}x{0} rebinned first", self.test_rebin)
                } else {
                    String::new()
                }
            ))
            .weak()
            .size(11.0),
        );

        ui.add_space(8.0);
        self.geometry_panel(ui, &stack);
    }

    /// The center-of-rotation readout under the projection: the value the
    /// current offset stands for, and where it moved from when it differs
    /// from the checkpoint's.
    fn center_readout(&self, ui: &mut egui::Ui, stack: &LoadedStack, w: usize) {
        let (file_cor, from_file) = checkpoint_center(stack, w);
        let current_cor = center_from_offset(self.params.det_channel_offset, w);
        let moved = (current_cor - file_cor).abs() > 1e-6;
        let file_origin = if from_file {
            "saved in the checkpoint"
        } else {
            "detector center — the checkpoint carries none"
        };
        if moved {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Center of rotation:").strong());
                ui.colored_label(
                    Color32::from_rgb(120, 230, 120),
                    format!("{current_cor:.2} px (current, solid green)"),
                );
                ui.label(RichText::new(format!(
                    "— moved {:+.2} px from",
                    current_cor - file_cor
                )));
                ui.colored_label(
                    Color32::from_rgb(255, 170, 40),
                    format!("{file_cor:.2} px ({file_origin}, dashed)"),
                );
            });
        } else {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Center of rotation:").strong());
                ui.colored_label(
                    Color32::from_rgb(255, 170, 40),
                    format!("{file_cor:.2} px ({file_origin}, dashed line)"),
                );
            });
        }
        ui.label(
            RichText::new(format!(
                "offset from the detector center ({} px): {:+.2} px — adjustable in the \
                 Advanced section",
                w / 2,
                self.params.det_channel_offset
            ))
            .weak()
            .size(11.0),
        );
    }

    /// What the checkpoint records about its geometry (tilt corrections,
    /// pre-processing rebin), and the launcher of the standalone tilt &
    /// center-of-rotation tool.
    fn geometry_panel(&mut self, ui: &mut egui::Ui, stack: &Arc<LoadedStack>) {
        ui.label(RichText::new("Tilt correction").strong());
        let lines = tilt_summary_lines(stack);
        if lines.is_empty() {
            ui.label(
                RichText::new("none recorded in this checkpoint")
                    .weak()
                    .size(12.0),
            );
        }
        for line in &lines {
            ui.label(RichText::new(format!("✔ {line}")).size(12.0));
        }
        if let Some((_, desc)) = stack.metadata.iter().find(|(name, _)| name == "rebin") {
            ui.label(
                RichText::new(format!("pre-processing rebin: {desc}"))
                    .weak()
                    .size(12.0),
            );
        }
        ui.add_space(4.0);
        let tool_open = self.tilt_tool_job.is_some();
        let busy = tool_open || self.recon_job.is_some();
        if ui
            .add_enabled(
                !busy,
                egui::Button::new("🎯 Open the tilt & center-of-rotation tool"),
            )
            .on_hover_text(
                "the standalone tool with the more robust estimators (sub-pixel \
                 0°/180° registration, all-pairs consensus, gridrec test slices). \
                 It opens this checkpoint directly: applying & saving there \
                 rewrites its projections and center of rotation, and this window \
                 reloads the file when the tool closes.",
            )
            .clicked()
        {
            let path = stack.path.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let result = std::process::Command::new(TILT_COR_BIN)
                    .arg(&path)
                    .arg("--called-from-app")
                    .output();
                let _ = tx.send(match result {
                    Err(e) => Err(format!("cannot launch {TILT_COR_BIN}: {e}")),
                    Ok(out) if !out.status.success() => Err(format!(
                        "the tilt & center-of-rotation tool failed ({}): {}",
                        out.status,
                        String::from_utf8_lossy(&out.stderr).trim()
                    )),
                    Ok(_) => Ok(()),
                });
            });
            self.tilt_tool_job = Some(rx);
            self.tilt_tool_note = None;
            self.status = "the tilt & center-of-rotation tool is open".to_owned();
        }
        if tool_open {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    RichText::new(
                        "the tool is open — estimate, apply & save there, then close it \
                         to come back",
                    )
                    .size(12.0),
                );
            });
        }
        match &self.tilt_tool_note {
            Some(Ok(msg)) => {
                let ok_color = if ui.visuals().dark_mode {
                    Color32::from_rgb(120, 200, 120)
                } else {
                    Color32::from_rgb(27, 118, 51)
                };
                ui.colored_label(ok_color, msg);
            }
            Some(Err(e)) => {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
            None => {}
        }
    }

    /// Fold a finished tilt-tool session in: when the checkpoint gained a
    /// correction record, reload it (re-seeding the offset from the new
    /// center of rotation); otherwise nothing changed.
    fn poll_tilt_tool(&mut self) {
        let Some(rx) = &self.tilt_tool_job else { return };
        let outcome = match rx.try_recv() {
            Ok(outcome) => outcome,
            Err(_) => return,
        };
        self.tilt_tool_job = None;
        let Some(stack) = self.stack.clone() else { return };
        match outcome {
            Err(e) => {
                self.tilt_tool_note = Some(Err(e));
                self.status = "the tilt & center-of-rotation tool failed".to_owned();
            }
            Ok(()) => match checkpoint_geometry(&stack.path) {
                Err(e) => {
                    self.tilt_tool_note = Some(Err(format!(
                        "cannot re-read the checkpoint after the tool closed: {e}"
                    )));
                }
                Ok((cor, records)) => {
                    let before = tilt_tool_records(&stack.metadata).len();
                    if records.len() > before {
                        let last = records.last().cloned().unwrap_or_default();
                        let doc: serde_json::Value =
                            serde_json::from_str(&last).unwrap_or_default();
                        let tilt = doc
                            .get("corrected_tilt_deg")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        self.tilt_tool_note = Some(Ok(format!(
                            "applied: tilt {tilt:+.4}° corrected, center of rotation {:.2} px \
                             — the checkpoint was reloaded",
                            cor.unwrap_or(f64::NAN)
                        )));
                        self.reseed_offset_on_load = true;
                        self.start_load(stack.path.clone());
                    } else {
                        self.tilt_tool_note =
                            Some(Ok("closed without applying a correction".to_owned()));
                        self.status = "the tilt & center-of-rotation tool closed — \
                                       nothing changed"
                            .to_owned();
                    }
                }
            },
        }
    }

    fn params_panel(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("MBIRJAX parameters").strong());
        ui.add(
            egui::Slider::new(&mut self.params.sharpness, -1.0..=3.0)
                .step_by(0.1)
                .text("sharpness"),
        )
        .on_hover_text(
            "larger (positive) values produce sharper, more detailed images; smaller \
             (negative) values produce smoother images",
        );

        egui::CollapsingHeader::new(RichText::new("🔒 Advanced").strong())
            .default_open(false)
            .show(ui, |ui| {
                if !self.advanced_unlocked {
                    ui.horizontal(|ui| {
                        ui.label("Password:");
                        let edit = ui.add(
                            egui::TextEdit::singleline(&mut self.advanced_password)
                                .password(true)
                                .desired_width(140.0),
                        );
                        let entered =
                            edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        if ui.button("Unlock").clicked() || entered {
                            if password_matches(&self.advanced_password) {
                                self.advanced_unlocked = true;
                                self.advanced_error = None;
                            } else {
                                self.advanced_error = Some("wrong password".to_owned());
                            }
                            self.advanced_password.clear();
                        }
                    });
                    if let Some(e) = &self.advanced_error {
                        ui.colored_label(ui.visuals().error_fg_color, e);
                    }
                    return;
                }
                ui.add(
                    egui::Slider::new(&mut self.params.snr_db, 25.0..=35.0)
                        .text("signal noise ratio (dB)"),
                )
                .on_hover_text(
                    "assumed signal-to-noise ratio of the data; larger values yield \
                     sharper reconstructions but can amplify noise",
                );
                ui.add(
                    egui::Slider::new(&mut self.params.max_iterations, 10..=25)
                        .text("maximum iterations"),
                )
                .on_hover_text(
                    "more iterations improve convergence but increase the reconstruction time",
                );
                ui.add(
                    egui::Slider::new(&mut self.params.scale, 0.5..=2.0)
                        .step_by(0.1)
                        .text("reconstruction scale"),
                )
                .on_hover_text(
                    "scales the reconstruction grid (row and column together); 1.0 keeps \
                     the detector size",
                );
                ui.checkbox(&mut self.params.positivity, "positivity")
                    .on_hover_text("all reconstructed voxel values are forced to be >= 0");
                let width = self
                    .stack
                    .as_ref()
                    .and_then(|s| s.sample.first())
                    .map(|p| p.width as f64)
                    .unwrap_or(512.0);
                let (file_cor, w) = self
                    .stack
                    .as_ref()
                    .and_then(|s| s.sample.first().map(|p| (checkpoint_center(s, p.width).0, p.width)))
                    .unwrap_or((width / 2.0, width as usize));
                ui.horizontal(|ui| {
                    ui.label("center of rotation offset from center:");
                    ui.add(
                        egui::DragValue::new(&mut self.params.det_channel_offset)
                            .speed(drag_speed(ui, 0.01))
                            .range(-width / 2.0..=width / 2.0),
                    )
                    .on_hover_text(
                        "offset of the center of rotation from the center of the detector \
                         image, in pixels; positive values shift it to the right — the \
                         green line on the projection follows it. Drag or use the arrow \
                         keys; hold Shift to move 10× faster",
                    );
                    ui.label(
                        RichText::new(format!(
                            "= center of rotation {:.2} px",
                            center_from_offset(self.params.det_channel_offset, w)
                        ))
                        .weak(),
                    );
                    let file_offset = offset_from_center(file_cor, w);
                    if (self.params.det_channel_offset - file_offset).abs() > 1e-6
                        && ui
                            .button("↺ checkpoint value")
                            .on_hover_text(format!(
                                "back to the checkpoint's center of rotation ({file_cor:.2} px)"
                            ))
                            .clicked()
                    {
                        self.params.det_channel_offset = file_offset;
                    }
                });
            });
    }

    fn results_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(job) = &mut self.recon_job {
            match job.poll() {
                Some(Ok((rh, rw, top, bottom, seconds))) => {
                    let (top_thumb, tw, th) = Self::downsample(&top, rw, rh, 512);
                    let (bottom_thumb, ..) = Self::downsample(&bottom, rw, rh, 512);
                    self.history.push(HistoryEntry {
                        params: self.params,
                        top_slice: self.top_slice,
                        bottom_slice: self.bottom_slice,
                        test_rebin: self.result_rebin,
                        seconds,
                        thumb_size: (tw, th),
                        top_thumb,
                        bottom_thumb,
                        tex: None,
                    });
                    self.result_tex = Some((
                        Self::grayscale_texture(ctx, "recon_top", &top, rw, rh),
                        Self::grayscale_texture(ctx, "recon_bottom", &bottom, rw, rh),
                    ));
                    self.result = Some((rh, rw, top, bottom, seconds));
                    self.recon_error = None;
                    self.recon_job = None;
                    self.status = format!("Reconstruction done in {seconds:.1} s.");
                }
                Some(Err(e)) => {
                    self.recon_error = Some(e);
                    self.recon_job = None;
                    self.status = "Reconstruction failed.".to_owned();
                }
                None => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(if self.result_rebin > 1 {
                            format!(
                                "mbirjax is reconstructing the two test slices ({0}x{0} \
                                 rebinned)…",
                                self.result_rebin
                            )
                        } else {
                            "mbirjax is reconstructing the two test slices…".to_owned()
                        });
                    });
                    ctx.request_repaint_after(Duration::from_millis(300));
                }
            }
        }

        let busy = self.recon_job.is_some() || self.tilt_tool_job.is_some();
        ui.horizontal(|ui| {
            let evaluate = egui::Button::new(
                RichText::new("▶ Evaluate the reconstruction of the selected slices")
                    .size(16.0)
                    .strong()
                    .color(Color32::WHITE),
            )
            .fill(Color32::from_rgb(230, 126, 0))
            .min_size(egui::vec2(0.0, 36.0));
            if ui
                .add_enabled(self.stack.is_some() && !busy, evaluate)
                .clicked()
            {
                let stack = self.stack.clone().expect("checked above");
                self.recon_error = None;
                self.status = "Running mbirjax…".to_owned();
                self.result_rebin = self.test_rebin;
                self.recon_job = Some(ReconJob::start(
                    stack,
                    self.top_slice,
                    self.bottom_slice,
                    self.params,
                    self.test_rebin,
                ));
            }
        });
        if let Some(e) = &self.recon_error {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }

        if let (Some((rh, rw, .., seconds)), Some((top_tex, bottom_tex))) =
            (&self.result, &self.result_tex)
        {
            ui.label(
                RichText::new(format!(
                    "reconstructed {rh}x{rw} slices in {seconds:.1} s{} — {}",
                    if self.result_rebin > 1 {
                        format!(" from {0}x{0} rebinned data", self.result_rebin)
                    } else {
                        String::new()
                    },
                    self.params.describe()
                ))
                .strong(),
            );
            ui.columns(2, |cols| {
                for (col, tex, label, row) in [
                    (0usize, top_tex, "top slice", self.top_slice),
                    (1, bottom_tex, "bottom slice", self.bottom_slice),
                ] {
                    let ui = &mut cols[col];
                    ui.label(
                        RichText::new(format!("{label} (row {row})"))
                            .strong()
                            .size(13.0),
                    );
                    let size = tex.size_vec2();
                    let width = (ui.available_width() - 12.0).clamp(200.0, 460.0);
                    ui.add(
                        egui::Image::from_texture(tex)
                            .fit_to_exact_size(egui::vec2(width, width * size.y / size.x)),
                    );
                }
            });
        }

        if !self.history.is_empty() {
            ui.add_space(6.0);
            egui::CollapsingHeader::new(RichText::new("Run history").strong())
                .default_open(false)
                .show(ui, |ui| {
                    let mut restore = None;
                    for (i, entry) in self.history.iter_mut().enumerate().rev() {
                        let (tw, th) = entry.thumb_size;
                        let (top_tex, bottom_tex) = entry.tex.get_or_insert_with(|| {
                            (
                                Self::grayscale_texture(
                                    ctx,
                                    &format!("hist_top_{i}"),
                                    &entry.top_thumb,
                                    tw,
                                    th,
                                ),
                                Self::grayscale_texture(
                                    ctx,
                                    &format!("hist_bottom_{i}"),
                                    &entry.bottom_thumb,
                                    tw,
                                    th,
                                ),
                            )
                        });
                        ui.horizontal(|ui| {
                            if ui.button("use").clicked() {
                                restore = Some((entry.params, entry.test_rebin));
                            }
                            for (tex, which) in
                                [(&*top_tex, "top"), (&*bottom_tex, "bottom")]
                            {
                                let size = tex.size_vec2();
                                let thumb_h = 96.0;
                                ui.add(egui::Image::from_texture(tex).fit_to_exact_size(
                                    egui::vec2(thumb_h * size.x / size.y, thumb_h),
                                ))
                                .on_hover_ui(|ui| {
                                    ui.label(
                                        RichText::new(format!("#{} — {which} slice", i + 1))
                                            .strong(),
                                    );
                                    let big = 420.0;
                                    ui.add(egui::Image::from_texture(tex).fit_to_exact_size(
                                        egui::vec2(big, big * size.y / size.x),
                                    ));
                                });
                            }
                            ui.label(
                                RichText::new(format!(
                                    "#{} — rows {}/{} — {}{} — {:.1} s",
                                    i + 1,
                                    entry.top_slice,
                                    entry.bottom_slice,
                                    entry.params.describe(),
                                    if entry.test_rebin > 1 {
                                        format!(", test rebin {0}x{0}", entry.test_rebin)
                                    } else {
                                        String::new()
                                    },
                                    entry.seconds
                                ))
                                .size(12.0),
                            );
                        });
                    }
                    if let Some((params, rebin)) = restore {
                        self.params = params;
                        self.test_rebin = rebin;
                    }
                });
        }
    }

    /// Save the parameters into the checkpoint, record the outcome in
    /// `save_status`, and report success.
    fn save_params_and_report(&mut self) -> bool {
        let path = self.stack.as_ref().expect("stack checked").path.clone();
        let result = save_params(&path, &self.params)
            .map(|()| format!("mbirjax_config saved into {}", path.display()));
        if result.is_ok() && self.called_from_app {
            println!("{}", self.params.to_json());
        }
        let ok = result.is_ok();
        self.save_status = Some(result);
        ok
    }
}

impl eframe::App for OptimizerApp {
    fn on_exit(&mut self) {
        crate::recon::kill_running();
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_tilt_tool();
        if self.tilt_tool_job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(500));
        }
        if let Some(job) = &mut self.load_job {
            match job.poll() {
                Some(Ok(stack)) => {
                    self.adopt_stack(stack);
                    self.load_job = None;
                }
                Some(Err(e)) => {
                    self.load_error = Some(e);
                    self.load_job = None;
                    self.status = "Loading failed.".to_owned();
                }
                None => ctx.request_repaint_after(Duration::from_millis(300)),
            }
        }

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📂 Open a checkpoint HDF5…").clicked() {
                    let mut dialog = rfd::FileDialog::new()
                        .set_title("Select a pre-processed checkpoint HDF5")
                        .add_filter("HDF5", &["h5", "hdf5"]);
                    if let Some(dir) = self
                        .stack
                        .as_ref()
                        .and_then(|s| s.path.parent())
                        .filter(|p| p.is_dir())
                    {
                        dialog = dialog.set_directory(dir);
                    }
                    if let Some(path) = dialog.pick_file() {
                        self.start_load(path);
                    }
                }
                ui.label(RichText::new(&self.status).weak());
                let logos = self.logo_tex.get_or_insert_with(|| {
                    [
                        ("imaging_logo", IMAGING_LOGO_BYTES),
                        ("mbirjax_logo", MBIRJAX_LOGO_BYTES),
                    ]
                    .into_iter()
                    .filter_map(|(name, bytes)| load_logo(&ctx, name, bytes))
                    .collect()
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    for tex in logos.iter() {
                        ui.add(egui::Image::from_texture(tex).max_height(LOGO_MAX_HEIGHT));
                    }
                    ui.separator();
                    crate::theme::toggle_button(ui);
                    crate::zoom::toggle_button(ui);
                });
            });
        });
        if self.stack.is_some() {
            egui::Panel::bottom("actions").show(ui, |ui| {
                // Not while the tilt tool may be rewriting the file.
                let ready = self.recon_job.is_none() && self.tilt_tool_job.is_none();
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let save = egui::Button::new(
                        RichText::new("💾 Save the parameters into the HDF5")
                            .size(16.0)
                            .strong()
                            .color(Color32::WHITE),
                    )
                    .fill(Color32::from_rgb(46, 125, 50))
                    .min_size(egui::vec2(0.0, 36.0));
                    if ui
                        .add_enabled(ready, save)
                        .on_hover_text(
                            "writes mbirjax_config into the checkpoint so later reconstructions \
                             use these parameters",
                        )
                        .clicked()
                    {
                        self.save_params_and_report();
                    }
                    let ret = egui::Button::new(
                        RichText::new("↩ Return to the main application")
                            .size(16.0)
                            .strong()
                            .color(Color32::WHITE),
                    )
                    .fill(Color32::from_rgb(21, 101, 192))
                    .min_size(egui::vec2(0.0, 36.0));
                    if ui
                        .add_enabled(ready, ret)
                        .on_hover_text("saves the parameters into the HDF5 and closes this tool")
                        .clicked()
                        && self.save_params_and_report()
                    {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    match &self.save_status {
                        Some(Ok(msg)) => {
                            // The pale green is only legible on the dark background;
                            // the light theme needs a deeper one.
                            let ok_color = if ui.visuals().dark_mode {
                                Color32::from_rgb(120, 200, 120)
                            } else {
                                Color32::from_rgb(27, 118, 51)
                            };
                            ui.colored_label(ok_color, msg);
                        }
                        Some(Err(e)) => {
                            ui.colored_label(ui.visuals().error_fg_color, e);
                        }
                        None => {}
                    }
                });
                ui.add_space(6.0);
            });
        }
        egui::CentralPanel::default().show(ui, |ui| {
            if self.load_job.is_some() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("loading the stack…");
                });
                return;
            }
            if let Some(e) = &self.load_error {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
            if self.stack.is_none() {
                return;
            }
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.columns(2, |cols| {
                        self.projection_panel(&mut cols[0], &ctx);
                        self.params_panel(&mut cols[1]);
                    });
                    ui.separator();
                    self.results_panel(ui, &ctx);
                });
        });
    }
}
