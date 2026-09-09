# MBIRJAX Optimizer

Standalone GUI to tune MBIRJAX reconstruction parameters on a pre-processed
CT checkpoint (the HDF5 written by `rust_ct_reconstruction`: attenuation
data with `/angles_rad` and `/center_of_rotation`). Modeled on the
`marimo_optimize_mbirjax_parameters` notebook.

## Workflow

1. Open a checkpoint (command-line argument or the 📂 button).
2. Pick two slices on the projection view (red and cyan lines).
3. Adjust the parameters — **sharpness** in the open section; SNR,
   iterations, the reconstruction **scale** (driving `row_scale` and
   `col_scale` together), positivity and the detector channel offset behind
   the password-locked **Advanced** section.
4. Choose the **Test data** resolution: the checkpoint's own, or the
   projections n×n block-averaged first (2x2, 3x3, 4x4, 6x6 — the same
   block mean as the pre-processing rebin step). Smaller sinograms
   reconstruct much faster and fit the GPU when the full width would not;
   the slices are coarser. The starting choice is the smallest factor whose
   width fits one GPU according to the memory model of
   `rust_ct_reconstruction` (the checkpoint's own resolution when it fits).
   This is a shortcut for the test only: the parameters are saved in the
   checkpoint's pixel units and the full reconstruction runs on the
   checkpoint as is (only the detector channel offset is converted for the
   rebinned test run, following the pixel grid like the pre-processing
   rebin does).
5. **▶ Evaluate** reconstructs a 10-slice band around each line through the
   real `mbirjax` (from the `all_ct_reconstruction_development` pixi
   environment) and shows the two middle slices side by side. Repeat until
   satisfied — every run lands in the **Run history**, whose `use` buttons
   restore the parameters (and test resolution) of a previous run.
6. **💾 Save** writes `mbirjax_config` (JSON, with `row_scale`/`col_scale`
   like the notebook) into the checkpoint's `/metadata`;
   `rust_ct_reconstruction` restores it automatically and later MBIRJAX
   reconstructions use these parameters.

## Center of rotation and tilt

The projection view draws the **center of rotation**: the checkpoint's
value as a dashed orange line and, once the detector channel offset is
changed in the Advanced section, the current one as a solid green line,
with a readout of both values and the move between them (`↺ checkpoint
value` goes back). Drag the offset field or use the arrow keys; holding
Shift moves it 10× faster. The detector channel offset defaults to
`-(width/2 - center_of_rotation)` from the checkpoint.

Under the projection, **Tilt correction** lists what the checkpoint
records: the pre-processing step of `rust_ct_reconstruction`
(`tilt_correction`) and every run of the standalone tool
(`metadata/tilt_center_of_rotation`, JSON records), plus the
pre-processing rebin when there was one.

**🎯 Open the tilt & center-of-rotation tool** launches
`rust_tilt_center_of_rotation` on the checkpoint itself (the more robust
estimators: sub-pixel 0°/180° registration, all-pairs consensus, gridrec
test slices). Applying & saving there rewrites the checkpoint's
projections and center of rotation; when the tool closes, this window
detects the new correction record, reloads the file and re-seeds the
detector channel offset from the new center — save the parameters to keep
it. Closing the tool without applying changes nothing.

Note: every evaluation starts a fresh Python process, so JAX recompiles its
kernels each run — a fixed overhead of about a minute on top of the
reconstruction itself (which runs on one of the node's GPUs when
available). Measured on a 490 × 2048 × 2054 checkpoint (A100 40 GB), one
evaluation of the two bands took 99 s at full resolution, 67 s at 2x2 and
65 s at 4x4: past 2x2 the start-up floor is all that is left, so the rebin
buys about a third on data this wide and is mostly there to make
checkpoints wider than ~2048 px testable at all.

## Running

```bash
./launch_mbirjax_optimizer.sh [checkpoint.h5]
```

Requires a graphical session; the launch script rebuilds when sources
changed. `--called-from-app` additionally prints the saved JSON on stdout
for a driving application.
