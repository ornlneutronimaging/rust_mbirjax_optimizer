//! MBIRJAX parameters, the test reconstruction on two slice bands (through
//! the real mbirjax of the `all_ct_reconstruction_development` pixi
//! environment), and saving the parameters back into the checkpoint HDF5.

use ct_reconstruction::combine::LoadedStack;
use ct_reconstruction::crop::{read_npy, write_npy};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// The interpreter of the pixi environment that has mbirjax installed.
pub const MBIRJAX_PYTHON: &str =
    "/SNS/VENUS/shared/software/git/all_ct_reconstruction_development/.pixi/envs/default/bin/python";

/// Slices reconstructed around each selected line (the Python
/// `MARIMO_MBIRJAX_TEST_RECONSTRUCTION_WIDTH`); the middle one is shown.
pub const BAND: usize = 10;

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
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let result = run_recon(&stack, top_slice, bottom_slice, params).map(
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

fn run_recon(
    stack: &LoadedStack,
    top_slice: usize,
    bottom_slice: usize,
    params: MbirjaxParams,
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

    // One 10-slice band around each selected line; the middle slice is shown.
    let band_range = |line: usize| -> (usize, usize) {
        let half = BAND / 2;
        let start = line.saturating_sub(half).min(h.saturating_sub(BAND));
        (start, start + BAND)
    };
    let (top_a, top_b) = band_range(top_slice);
    let (bottom_a, bottom_b) = band_range(bottom_slice);

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
        let mut volume = Vec::with_capacity(n * 2 * BAND * w);
        for p in &stack.sample {
            for range in [(top_a, top_b), (bottom_a, bottom_b)] {
                volume.extend_from_slice(&p.mean[range.0 * w..range.1 * w]);
            }
        }
        write_npy(&sino_npy, &[n, 2 * BAND, w], volume.chunks(2 * BAND * w))?;
        let spec = serde_json::json!({
            "angles_rad": angles,
            "bands": [[0, BAND], [BAND, 2 * BAND]],
            "params": serde_json::from_str::<serde_json::Value>(&params.to_json()).expect("params json"),
        });
        std::fs::write(&spec_file, spec.to_string())
            .map_err(|e| format!("write {}: {e}", spec_file.display()))?;
        std::fs::write(&script, MBIRJAX_SCRIPT)
            .map_err(|e| format!("write {}: {e}", script.display()))?;
        let output = std::process::Command::new(MBIRJAX_PYTHON)
            .arg(&script)
            .arg(&sino_npy)
            .arg(&spec_file)
            .arg(&out_npy)
            .output()
            .map_err(|e| format!("cannot launch {MBIRJAX_PYTHON}: {e}"))?;
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
}
