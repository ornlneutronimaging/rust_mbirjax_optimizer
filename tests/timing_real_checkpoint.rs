//! Timing of the test reconstruction on a real checkpoint, at the
//! checkpoint's resolution and 2x2 rebinned. Needs the GPU node and the
//! file, so it is ignored by default:
//!
//!   MBIRJAX_TIMING_H5=/path/to/checkpoint.h5 cargo test --release --test timing_real_checkpoint -- --ignored --nocapture

use nectar::combine::load_hdf5;
use mbirjax_optimizer::recon::{MbirjaxParams, ReconJob};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

#[test]
#[ignore]
fn timing_full_vs_rebinned() {
    let Ok(path) = std::env::var("MBIRJAX_TIMING_H5") else {
        eprintln!("MBIRJAX_TIMING_H5 not set — skipping");
        return;
    };
    let t = std::time::Instant::now();
    let stack = Arc::new(load_hdf5(Path::new(&path)).expect("load"));
    let first = stack.sample.first().unwrap();
    let (w, h) = (first.width, first.height);
    eprintln!("loaded {} x {h} x {w} in {:.1} s", stack.sample.len(), t.elapsed().as_secs_f64());
    let params = MbirjaxParams::from_stack(&stack);
    eprintln!("params: {}", params.describe());
    for rebin in [1usize, 2, 4] {
        let mut job = ReconJob::start(Arc::clone(&stack), h / 3, 2 * h / 3, params, rebin);
        let result = loop {
            if let Some(r) = job.poll() {
                break r;
            }
            std::thread::sleep(Duration::from_millis(500));
        };
        match result {
            Ok((rh, rw, top, bottom, s)) => {
                let mean = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
                eprintln!(
                    "rebin {rebin}: {rh}x{rw} slices in {s:.1} s (means {:.4} / {:.4})",
                    mean(&top),
                    mean(&bottom)
                );
            }
            Err(e) => eprintln!("rebin {rebin}: FAILED: {e}"),
        }
    }
}
