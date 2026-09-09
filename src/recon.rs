//! MBIRJAX parameters, the test reconstruction on two slice bands (through
//! the real mbirjax of the `all_ct_reconstruction_development` pixi
//! environment), and saving the parameters back into the checkpoint HDF5.

use ct_reconstruction::combine::{LoadedStack, Projection};
use ct_reconstruction::crop::{read_npy, write_npy};
use ct_reconstruction::rebin::{rebin_center, rebin_projection, rebinned_size};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

/// The interpreter of the pixi environment that has mbirjax installed.
pub const MBIRJAX_PYTHON: &str =
    "/SNS/VENUS/shared/software/git/all_ct_reconstruction_development/.pixi/envs/default/bin/python";

/// Slices reconstructed around each selected line (the Python
/// `MARIMO_MBIRJAX_TEST_RECONSTRUCTION_WIDTH`); the middle one is shown.
pub const BAND: usize = 10;

/// The n×n rebin factors the test reconstruction can run on (1 = the
/// checkpoint's own resolution).
pub const TEST_REBIN_FACTORS: [usize; 5] = [1, 2, 3, 4, 6];

/// The center of rotation (px) a detector-channel offset stands for on
/// `width`-pixel-wide projections — the inverse of the seeding rule
/// `offset = -(width/2 - cor)` (integer half width, like the notebook).
pub fn center_from_offset(offset: f64, width: usize) -> f64 {
    (width / 2) as f64 + offset
}

/// The detector-channel offset that puts the center of rotation at `cor`
/// on `width`-pixel-wide projections.
pub fn offset_from_center(cor: f64, width: usize) -> f64 {
    cor - (width / 2) as f64
}

/// The detector-channel offset for the n×n rebinned test data, given the
/// offset on the checkpoint's `width`-pixel-wide projections: the center of
/// rotation follows the pixel grid ([`rebin_center`]) and is re-expressed
/// against the rebinned half width.
pub fn rebinned_offset(offset: f64, width: usize, n: usize) -> f64 {
    if n <= 1 {
        return offset;
    }
    let cor = rebin_center(center_from_offset(offset, width), n);
    let (rw, _) = rebinned_size(width, 1, n);
    offset_from_center(cor, rw)
}

/// Largest n×n test-rebin factor that is useful for a stack of `height`
/// rows: the band must still hold [`BAND`] rebinned slices.
pub fn max_test_rebin(height: usize) -> usize {
    TEST_REBIN_FACTORS
        .iter()
        .copied()
        .filter(|n| BAND * n <= height.max(1))
        .max()
        .unwrap_or(1)
}

/// Does a [`BAND`]-slice test job on `width`-pixel-wide sinograms fit on
/// one of the machine's GPUs? `None` when no GPU can be probed (jax would
/// run on the CPU and memory is not the constraint).
pub fn test_band_fits(width: usize, views: usize) -> Option<bool> {
    let (_, min_mib) = ct_reconstruction::app::gpu_inventory()?;
    let s = ct_reconstruction::app::mbirjax_max_slices(width.max(1), views.max(1), 1, min_mib);
    Some(s.is_finite() && s >= BAND as f64)
}

/// The test-rebin factor to start from: the smallest one whose rebinned
/// width fits the GPU memory model (the checkpoint's own resolution when it
/// fits), the largest useful one when none does.
pub fn default_test_rebin(width: usize, height: usize, views: usize) -> usize {
    let max = max_test_rebin(height);
    for n in TEST_REBIN_FACTORS.iter().copied().filter(|n| *n <= max) {
        let (rw, _) = rebinned_size(width, height, n);
        if test_band_fits(rw, views) != Some(false) {
            return n;
        }
    }
    max
}

/// The MBIRJAX parameters exposed by the marimo notebook, with its defaults
/// and ranges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MbirjaxParams {
    /// General parameter (-1 to 3): larger = sharper.
    pub sharpness: f64,
    /// Assumed signal-to-noise ratio in dB (25 to 35).
    pub snr_db: f64,
    /// Force reconstructed voxels >= 0 (`positivity_flag`).
    pub positivity: bool,
    /// Iterations of the solver (10 to 25).
    pub max_iterations: i64,
    /// Reconstruction-grid scale, driving `row_scale` and `col_scale`
    /// together (0.5 to 2.0).
    pub scale: f64,
    /// Center of rotation as a pixel offset from the detector center
    /// (`det_channel_offset`).
    pub det_channel_offset: f64,
}

impl Default for MbirjaxParams {
    fn default() -> Self {
        Self {
            sharpness: 1.0,
            snr_db: 30.0,
            positivity: false,
            max_iterations: 15,
            scale: 1.0,
            det_channel_offset: 0.0,
        }
    }
}

impl MbirjaxParams {
    /// Defaults seeded from the stack: the saved `mbirjax_config` when the
    /// checkpoint carries one, otherwise the detector channel offset derived
    /// from the stack's center of rotation (`-(width/2 - cor)`).
    pub fn from_stack(stack: &LoadedStack) -> Self {
        if let Some((_, json)) = stack
            .metadata
            .iter()
            .find(|(name, _)| name == "mbirjax_config")
            && let Some(params) = Self::from_json(json)
        {
            return params;
        }
        let mut params = Self::default();
        if let (Some(cor), Some(first)) = (stack.center_of_rotation, stack.sample.first()) {
            params.det_channel_offset = -((first.width / 2) as f64 - cor);
        }
        params
    }

    /// The saved form matches the notebook's `mbirjax_config`: the single
    /// scale is written as `row_scale` and `col_scale`.
    pub fn to_json(&self) -> String {
        serde_json::json!({
            "sharpness": self.sharpness,
            "snr_db": self.snr_db,
            "positivity": self.positivity,
            "max_iterations": self.max_iterations,
            "row_scale": self.scale,
            "col_scale": self.scale,
            "det_channel_offset": self.det_channel_offset,
        })
        .to_string()
    }

    pub fn from_json(text: &str) -> Option<Self> {
        let doc: serde_json::Value = serde_json::from_str(text).ok()?;
        let mut params = Self::default();
        if let Some(v) = doc.get("sharpness").and_then(|v| v.as_f64()) {
            params.sharpness = v;
        }
        if let Some(v) = doc.get("snr_db").and_then(|v| v.as_f64()) {
            params.snr_db = v;
        }
        if let Some(v) = doc.get("positivity").and_then(|v| v.as_bool()) {
            params.positivity = v;
        }
        if let Some(v) = doc.get("max_iterations").and_then(|v| v.as_i64()) {
            params.max_iterations = v;
        }
        if let Some(v) = doc.get("row_scale").and_then(|v| v.as_f64()) {
            params.scale = v;
        }
        if let Some(v) = doc.get("det_channel_offset").and_then(|v| v.as_f64()) {
            params.det_channel_offset = v;
        }
        Some(params)
    }

    pub fn describe(&self) -> String {
        format!(
            "sharpness {:.1}, snr {:.0} dB, {} iter, scale {:.1}, offset {:.2}{}",
            self.sharpness,
            self.snr_db,
            self.max_iterations,
            self.scale,
            self.det_channel_offset,
            if self.positivity { ", positivity" } else { "" }
        )
    }
}

const MBIRJAX_SCRIPT: &str = r#"
import json
import sys

import numpy as np
import mbirjax as mj

sino_file, spec_file, out_file = sys.argv[1:4]
with open(spec_file) as f:
    spec = json.load(f)
sino = np.load(sino_file)  # (n_angles, total_band_slices, width)
angles = np.array(spec["angles_rad"], dtype=np.float32)
p = spec["params"]
slices = []
for a, b in spec["bands"]:
    s = np.ascontiguousarray(sino[:, a:b, :])
    model = mj.ParallelBeamModel(s.shape, angles)
    model.scale_recon_shape(row_scale=p["row_scale"], col_scale=p["col_scale"])
    model.set_params(
        sharpness=p["sharpness"],
        snr_db=p["snr_db"],
        det_channel_offset=p["det_channel_offset"],
        positivity_flag=bool(p["positivity"]),
    )
    recon, _ = model.recon(s, max_iterations=int(p["max_iterations"]))
    volume = np.array(np.swapaxes(np.array(recon, dtype=np.float32), 0, 2), dtype=np.float32)
    slices.append(volume[volume.shape[0] // 2, :, :])
np.save(out_file, np.stack(slices))
"#;

/// One test reconstruction of the two bands on a background thread;
/// resolves to the two reconstructed middle slices
/// `(height, width, values0, values1, seconds)`.
pub struct ReconJob {
    rx: Receiver<Result<(usize, usize, Vec<f32>, Vec<f32>, f64), String>>,
}

impl ReconJob {
    pub fn start(
        stack: Arc<LoadedStack>,
        top_slice: usize,
        bottom_slice: usize,
        params: MbirjaxParams,
        test_rebin: usize,
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let result = run_recon(&stack, top_slice, bottom_slice, params, test_rebin).map(
                |(h, w, top, bottom)| (h, w, top, bottom, started.elapsed().as_secs_f64()),
            );
            let _ = tx.send(result);
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<(usize, usize, Vec<f32>, Vec<f32>, f64), String>> {
        self.rx.try_recv().ok()
    }
}

fn scratch_dir(stack: &LoadedStack) -> Result<PathBuf, String> {
    let base = stack
        .path
        .parent()
        .filter(|p| p.is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!(".mbirjax_optimizer_{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_ok() {
        return Ok(dir);
    }
    let dir = std::env::temp_dir().join(format!("mbirjax_optimizer_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// PID of the mbirjax subprocess currently running, so it can be killed
/// when the application quits instead of finishing orphaned on the GPU.
static RUNNING_PID: Mutex<Option<u32>> = Mutex::new(None);

fn set_running(pid: Option<u32>) {
    if let Ok(mut guard) = RUNNING_PID.lock() {
        *guard = pid;
    }
}

/// Kill the mbirjax subprocess still running, if any (called on quit).
pub fn kill_running() {
    let pid = RUNNING_PID.lock().ok().and_then(|guard| *guard);
    if let Some(pid) = pid {
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .status();
    }
}

/// The one GPU the test reconstruction may use: the first of an
/// already-restricted `CUDA_VISIBLE_DEVICES`, or GPU 0.
fn single_gpu() -> String {
    std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .and_then(|v| v.split(',').next().map(|d| d.trim().to_string()))
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| "0".to_string())
}

/// The rows of one projection feeding a test band around `line`: `BAND`
/// rows at the checkpoint's resolution, `BAND × rebin` rows when the band
/// is rebinned first (so it still holds `BAND` slices afterwards).
fn band_rows(line: usize, height: usize, rebin: usize) -> (usize, usize) {
    let rows = BAND * rebin.max(1);
    let start = line
        .saturating_sub(rows / 2)
        .min(height.saturating_sub(rows));
    (start, start + rows)
}

/// The two test bands of one projection, stacked (top band first), at the
/// test resolution: cut out of the full projection and, for a rebin factor
/// above 1, block-averaged n×n like the pre-processing rebin step.
fn extract_bands(p: &Projection, bands: [(usize, usize); 2], rebin: usize) -> Vec<f32> {
    let w = p.width;
    let mut out = Vec::with_capacity(2 * BAND * w / rebin.max(1));
    for (a, b) in bands {
        let rows = &p.mean[a * w..b * w];
        if rebin <= 1 {
            out.extend_from_slice(rows);
        } else {
            let band = Projection {
                name: String::new(),
                run_number: None,
                angle_deg: None,
                n_images_used: 1,
                height: b - a,
                width: w,
                mean: rows.to_vec(),
                total_counts: 0.0,
            };
            out.extend_from_slice(&rebin_projection(&band, rebin).mean);
        }
    }
    out
}

fn run_recon(
    stack: &LoadedStack,
    top_slice: usize,
    bottom_slice: usize,
    params: MbirjaxParams,
    test_rebin: usize,
) -> Result<(usize, usize, Vec<f32>, Vec<f32>), String> {
    let first = stack
        .sample
        .first()
        .ok_or("no projections in the stack")?;
    let (w, h, n) = (first.width, first.height, stack.sample.len());
    let angles: Vec<f64> = stack
        .sample
        .iter()
        .map(|p| p.angle_deg.map(|a| a.to_radians()))
        .collect::<Option<Vec<f64>>>()
        .ok_or("some projections carry no angle — the reconstruction needs all of them")?;

    // The test runs on n×n rebinned data when asked: smaller sinograms
    // reconstruct much faster (and fit the GPU when the full width would
    // not). The parameters keep the checkpoint's pixel units; only the
    // detector-channel offset is re-expressed for the rebinned width.
    let rebin = test_rebin.clamp(1, max_test_rebin(h));
    let (rw, _) = rebinned_size(w, h, rebin);
    let mut test_params = params;
    test_params.det_channel_offset = rebinned_offset(params.det_channel_offset, w, rebin);

    // One BAND-slice band around each selected line; the middle slice is shown.
    let bands = [
        band_rows(top_slice, h, rebin),
        band_rows(bottom_slice, h, rebin),
    ];

    let dir = scratch_dir(stack)?;
    let sino_npy = dir.join("sino.npy");
    let spec_file = dir.join("spec.json");
    let out_npy = dir.join("recon.npy");
    let script = dir.join("mbirjax_run.py");
    let cleanup = || {
        for f in [&sino_npy, &spec_file, &out_npy, &script] {
            let _ = std::fs::remove_file(f);
        }
        let _ = std::fs::remove_dir(&dir);
    };
    let run = || -> Result<(usize, usize, Vec<f32>, Vec<f32>), String> {
        let mut volume = Vec::with_capacity(n * 2 * BAND * rw);
        for p in &stack.sample {
            volume.extend_from_slice(&extract_bands(p, bands, rebin));
        }
        write_npy(&sino_npy, &[n, 2 * BAND, rw], volume.chunks(2 * BAND * rw))?;
        let spec = serde_json::json!({
            "angles_rad": angles,
            "bands": [[0, BAND], [BAND, 2 * BAND]],
            "params": serde_json::from_str::<serde_json::Value>(&test_params.to_json()).expect("params json"),
        });
        std::fs::write(&spec_file, spec.to_string())
            .map_err(|e| format!("write {}: {e}", spec_file.display()))?;
        std::fs::write(&script, MBIRJAX_SCRIPT)
            .map_err(|e| format!("write {}: {e}", script.display()))?;
        let child = std::process::Command::new(MBIRJAX_PYTHON)
            .arg(&script)
            .arg(&sino_npy)
            .arg(&spec_file)
            .arg(&out_npy)
            // The test bands are tiny; keep JAX off the machine's other GPUs.
            .env("CUDA_VISIBLE_DEVICES", single_gpu())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot launch {MBIRJAX_PYTHON}: {e}"))?;
        set_running(Some(child.id()));
        let output = child.wait_with_output();
        set_running(None);
        let output = output.map_err(|e| format!("mbirjax: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: Vec<&str> = stderr.trim().lines().rev().take(4).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(format!(
                "mbirjax failed ({}): {}",
                output.status,
                tail.join(" | ")
            ));
        }
        // The reconstruction grid is scaled, so the slice size can differ
        // from the detector width.
        let (shape, values) = read_npy(&out_npy)?;
        let [count, rh, rw] = shape.as_slice() else {
            return Err(format!("mbirjax returned shape {shape:?}, expected 3-D"));
        };
        if *count != 2 {
            return Err(format!("mbirjax returned {count} slices, expected 2"));
        }
        let (top, bottom) = values.split_at(rh * rw);
        Ok((*rh, *rw, top.to_vec(), bottom.to_vec()))
    };
    let result = run();
    cleanup();
    result
}

/// The standalone tilt & center-of-rotation tool's correction records in a
/// stack's metadata: one JSON object per applied correction, oldest first.
pub fn tilt_tool_records(metadata: &[(String, String)]) -> Vec<String> {
    metadata
        .iter()
        .find(|(name, _)| name == "tilt_center_of_rotation")
        .map(|(_, value)| {
            value
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// The checkpoint's geometry without loading the projections: its
/// `/center_of_rotation` and the tilt tool's correction records. Used to
/// tell whether the tool changed the file before reloading gigabytes.
pub fn checkpoint_geometry(path: &Path) -> Result<(Option<f64>, Vec<String>), String> {
    use hdf5_metno::types::VarLenUnicode;
    let file = hdf5_metno::File::open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let cor = file
        .dataset("center_of_rotation")
        .and_then(|ds| ds.read_scalar::<f64>())
        .ok();
    let records = file
        .group("metadata")
        .and_then(|g| g.dataset("tilt_center_of_rotation"))
        .and_then(|ds| ds.read_scalar::<VarLenUnicode>())
        .map(|v| {
            v.as_str()
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    Ok((cor, records))
}

/// Write (or replace) the `mbirjax_config` JSON in the checkpoint's
/// `/metadata` group, where the main application reads it back.
pub fn save_params(path: &Path, params: &MbirjaxParams) -> Result<(), String> {
    use hdf5_metno::types::VarLenUnicode;
    let file = hdf5_metno::File::open_rw(path)
        .map_err(|e| format!("cannot open {} for writing: {e}", path.display()))?;
    let metadata = match file.group("metadata") {
        Ok(group) => group,
        Err(_) => file
            .create_group("metadata")
            .map_err(|e| format!("create metadata group: {e}"))?,
    };
    if metadata.dataset("mbirjax_config").is_ok() {
        metadata
            .unlink("mbirjax_config")
            .map_err(|e| format!("replace mbirjax_config: {e}"))?;
    }
    let value: VarLenUnicode = params.to_json().parse().unwrap_or_default();
    metadata
        .new_dataset::<VarLenUnicode>()
        .create("mbirjax_config")
        .and_then(|ds| ds.write_scalar(&value))
        .map_err(|e| format!("write mbirjax_config: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_json_roundtrip() {
        let params = MbirjaxParams {
            sharpness: 2.5,
            snr_db: 27.0,
            positivity: true,
            max_iterations: 12,
            scale: 1.5,
            det_channel_offset: 4.25,
        };
        let back = MbirjaxParams::from_json(&params.to_json()).unwrap();
        assert_eq!(back, params);
        // The saved form carries row_scale and col_scale like the notebook.
        let doc: serde_json::Value = serde_json::from_str(&params.to_json()).unwrap();
        assert_eq!(doc["row_scale"], 1.5);
        assert_eq!(doc["col_scale"], 1.5);
        assert!(MbirjaxParams::from_json("nope").is_none());
    }

    #[test]
    fn offset_and_center_are_inverse() {
        // The seeding rule: offset = -(width/2 - cor).
        assert_eq!(offset_from_center(979.96, 2054), -(1027.0 - 979.96));
        assert!((center_from_offset(offset_from_center(979.96, 2054), 2054) - 979.96).abs() < 1e-9);
        // Odd widths use the integer half width, like the notebook.
        assert_eq!(center_from_offset(0.0, 2055), 1027.0);
    }

    #[test]
    fn rebinned_offset_follows_the_pixel_grid() {
        // No rebin: unchanged.
        assert_eq!(rebinned_offset(-3.5, 2054, 1), -3.5);
        // The detector center stays the detector center of the rebinned
        // image (4096 wide: cor 2047.5 -> 1023.5 at 2x2, offset -0.5 both).
        let c = offset_from_center(2047.5, 4096);
        assert!((rebinned_offset(c, 4096, 2) - offset_from_center(1023.5, 2048)).abs() < 1e-9);
        // A general center: cor 979.96 on 2054 px -> (979.96+0.5)/2-0.5 on 1027 px.
        let cor = 979.96;
        let expected = offset_from_center((cor + 0.5) / 2.0 - 0.5, 1027);
        assert!((rebinned_offset(offset_from_center(cor, 2054), 2054, 2) - expected).abs() < 1e-9);
    }

    #[test]
    fn band_rows_hold_band_slices_after_rebin() {
        // 2x2: 20 rows around row 100 -> rebinned to 10 slices.
        assert_eq!(band_rows(100, 2048, 2), (90, 110));
        // Clamped at the edges.
        assert_eq!(band_rows(0, 2048, 3), (0, 30));
        assert_eq!(band_rows(2047, 2048, 1), (2038, 2048));
        // Tiny stacks limit the useful factor.
        assert_eq!(max_test_rebin(25), 2);
        assert_eq!(max_test_rebin(5), 1);
        assert_eq!(max_test_rebin(2048), 6);
    }

    #[test]
    fn extract_bands_rebins_each_band() {
        // 4 px wide, 8 rows: row r holds the value r everywhere.
        let mut mean = Vec::new();
        for r in 0..8 {
            mean.extend(std::iter::repeat_n(r as f32, 4));
        }
        let p = Projection {
            name: "p".into(),
            run_number: None,
            angle_deg: Some(0.0),
            n_images_used: 1,
            height: 8,
            width: 4,
            mean,
            total_counts: 0.0,
        };
        // Full resolution: the rows verbatim.
        let full = extract_bands(&p, [(0, 2), (6, 8)], 1);
        assert_eq!(full, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0]);
        // 2x2: rows (0,1) -> 0.5, rows (2,3) -> 2.5; 2 px wide.
        let reb = extract_bands(&p, [(0, 4), (4, 8)], 2);
        assert_eq!(reb, vec![0.5, 0.5, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5]);
    }
}
