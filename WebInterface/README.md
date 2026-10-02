# MetaSymbO Design Studio

A research workbench for existing MetaSymbO results and its model workflow. [Try MetaSymbO Online](http://zhoulab-1.cs.vt.edu:5559/). The frontend is plain HTML, CSS, and JavaScript. FastAPI serves the API and the Plotly bundle already installed in the Python environment; no frontend build step or CDN is required.

## Development

From the repository root, use the existing model environment:

```bash
conda activate mat_env
python -m WebInterface.app --port 18766
```

Open [the local workbench](http://127.0.0.1:18766). Port 18765 is reserved for production on this machine. The environment's Python is `/data/home/grads/jianpengc/miniconda3/envs/mat_env/bin/python`.

The browsing service needs NumPy, FastAPI, Uvicorn, and Plotly. All are already installed in `mat_env`; generation and prediction additionally use the project's existing model dependencies. A browser with WebGL supports the interactive viewport. The geometry table and SVG library thumbnails remain available as numerical/static alternatives.

## Current functionality

- Browse and search saved NPZ lattices by collection; import compatible geometry without deserializing object arrays.
- Inspect a lattice in 3D, including rotated views, cell repetition, node selection, cell geometry, and graph checks.
- Save favorites and brief drafts locally in the browser. Compare up to four lattices with synchronized cameras and a common coordinate scale.
- Review a natural-language brief before starting a generation run. Use the existing symbolic operations, a bounded attempt count, and a recorded random seed.
- Run elastic-property prediction separately from generation. Model jobs run in an isolated process with a 15-minute timeout, leaving the UI responsive.
- Persist run status, completed candidate snapshots, predictions, source information, supervisor suggestions, and supervisor scores. Refreshing the browser does not restart a job.
- Export the original geometry, a CSV comparison, a viewport image, or a JSON report containing evidence and the current draft brief.
- Save candidate-specific validation assumptions. **Simulation itself is not enabled** until the existing voxelization and homogenization routines have a verified geometry and units contract.

The library found 2,455 archives in this checkout. All 1,455 direct result archives passed the geometry reader. Nested baseline archives with object-valued geometry are shown as unsupported; they are never loaded with pickle enabled.

## Credentials and deployment

The user-provided root `.env` is read only by the server. `OPENAI_API_KEY` in the process environment takes precedence. `WebInterface/.env.local` is also supported. The server reads only the `OPENAI_API_KEY` assignment; it does not execute or source either file. Restart the server after rotating an already-loaded key.

Both credential-file locations are excluded from Git. The root `.env` has owner-only permissions (`0600`) on this machine. Credentials are never returned to JavaScript, embedded in HTML, included in run requests or reports, or passed on command lines. Model workers receive the credential through their process environment. Provider exceptions are reduced to safe error categories instead of being echoed into browser-visible logs.

Only the `static` directory, the installed Plotly bundle, and explicit API responses are served. The repository root, `.env`, checkpoints, and arbitrary filesystem paths are not public routes. Direct and traversal attempts to fetch credential files are covered by tests. Original NPZ downloads are resolved through indexed IDs within the configured result directories, not user-supplied paths. NPZ uploads have compressed and uncompressed size limits and reject object-valued geometry arrays and invalid array headers.

The app binds to **127.0.0.1**. Production uses a Caddy reverse proxy on public port **5559**, user systemd services with lingering enabled, a separate `.public` workspace, anonymous browser sessions, and persistent usage limits. No account or user-supplied API key is required. See [production operations, installation, limits, cleanup, and HTTPS instructions](../deploy/README.md). The services survive logout and start at boot; neither Codex nor SSH forwarding is part of the serving path.

## Model settings

Configuration stays on the server:

| Environment variable | Default |
| --- | --- |
| `METASYMBO_RESULTS` | Repository `results` directory |
| `METASYMBO_WORKSPACE` | `WebInterface/.local` |
| `METASYMBO_CHECKPOINT` | `checkpoints/vae_cond_128_beta001_dis_same_100_frac` |
| `METASYMBO_DATASET` | `~/datasets/metamaterial/LatticeModulus` |
| `METASYMBO_DEVICE` | `cpu` |
| `METASYMBO_DESIGNER` | Existing workflow's `gpt-4o-mini` |
| `METASYMBO_SUPERVISOR` | Existing workflow's `gpt-4.1` |
| `METASYMBO_PUBLIC` | `0`; deployment sets `1` for sessions and public quotas |
| `METASYMBO_ALLOWED_HOSTS` | Additional comma-separated public hostnames/IPs |

The model worker uses the existing orchestration rather than implementing another optimizer. The UI records snapshots at completed outer collaboration boundaries. It does not yet record every inner optimization step, pause/resume optimizer state, or force the currently inspected geometry to be the generation seed. One model job can run at a time. Graceful server shutdown stops the active worker; a restarted server marks unfinished history as interrupted instead of silently resubmitting it.

The generation API also accepts an optional complete 12-value condition vector, validated for length and finite values. The first UI exposes prompt guidance only, pending verification of the dataset's directional conventions. An unspecified numerical target is never filled with zero or inferred from text.

## Scientific interpretation

Existing `prop_list` fields are conditioning, not predicted or measured properties. The viewer therefore does not display them as evidence. New predictions are saved explicitly in `y_pred` with provenance. Young's, shear, and Poisson families retain the stored component order; units and directional mapping are explicitly marked unverified. The interface does not fabricate a full stiffness tensor, uncertainty bands, density, stress, or deformation fields.

The reader uses the same cell orientation as `utils.mat_utils.lattice_params_to_matrix`, implemented in NumPy to avoid importing the model stack for browsing. It preserves original coordinates and edges, while drawing undirected duplicate edges once. Isolated nodes count in connectivity checks. Repetition changes only the display and does not establish periodic connectivity. Geometry checks do not establish physical stability or manufacturability.

The Validate tab links to the existing [PhyVer demo](http://zhoulab-1.cs.vt.edu:5557/demo) as a simulation reference. PhyVer's deployed workflow performs atomistic UMA optimization and optional ORCA DFT; it is separate from mechanical truss validation. MetaSymbO and the PhyVer repository already share identical `visualization/voxel.py`, `visualization/homo3D.py`, and `visualization/visualize_Cij.py` implementations. Refer to those existing utilities when defining a verified lattice-simulation workflow; no duplicate solver is introduced here.

The shared supervisor parser now preserves negative signs and scientific notation and rejects malformed property groups. The unsupported reasoning-effort argument was removed from the existing supervisor request so its configured default non-reasoning model can accept the request.

## Verification

Run the tests using the existing model environment; no additional Python test dependency is needed:

```bash
python -m unittest WebInterface.test_app -v
node --check WebInterface/static/app.js
```

The suite covers geometry, supervisor parsing, import/export, upload and prompt limits, private-file and traversal rejection, symlink confinement, session isolation, persistent per-IP/global quotas, worker timeout, provider budgets, safe diagnostics, cleanup, and immediate access to saved results. Production browser and live-model smoke tests are separate; see [deployment validation](../deploy/README.md#validation).

No paid API request runs as part of the Python test suite. Development run history lives under ignored `.local`; production uses ignored `.public`. Favorites, validation drafts, and the current draft brief are browser-local. Production runs belong to an opaque HttpOnly browser session, and do not appear in another visitor's library or run history. Export a report when drafts or results need to be shared or archived.
