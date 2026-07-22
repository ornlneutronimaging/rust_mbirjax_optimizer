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
4. **▶ Evaluate** reconstructs a 10-slice band around each line through the
   real `mbirjax` (from the `all_ct_reconstruction_development` pixi
   environment) and shows the two middle slices side by side. Repeat until
   satisfied — every run lands in the **Run history**, whose `use` buttons
   restore the parameters of a previous run.
5. **💾 Save** writes `mbirjax_config` (JSON, with `row_scale`/`col_scale`
   like the notebook) into the checkpoint's `/metadata`;
   `rust_ct_reconstruction` restores it automatically and later MBIRJAX
   reconstructions use these parameters.

The detector channel offset defaults to `-(width/2 - center_of_rotation)`
from the checkpoint.

Note: every evaluation starts a fresh Python process, so JAX recompiles its
kernels each run — expect a couple of minutes of fixed overhead on top of
the reconstruction itself (which runs on the node's GPUs when available).

## Running

```bash
./launch_mbirjax_optimizer.sh [checkpoint.h5]
```

Requires a graphical session; the launch script rebuilds when sources
changed. `--called-from-app` additionally prints the saved JSON on stdout
for a driving application.
