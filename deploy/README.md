# Try MetaSymbO Online

**Public URL:** [http://zhoulab-1.cs.vt.edu:5559/](http://zhoulab-1.cs.vt.edu:5559/)  
**IP fallback:** [http://128.173.237.13:5559/](http://128.173.237.13:5559/)  
**Health:** [http://zhoulab-1.cs.vt.edu:5559/api/health](http://zhoulab-1.cs.vt.edu:5559/api/health)

This is an HTTP research demo. No Codex session, SSH tunnel, laptop forwarding, account, or visitor API key is needed. Do not submit confidential inputs over HTTP. Predictions and supervisor scores are research outputs, not verified physical measurements; simulation remains setup-only.

## Serving architecture

```text
Internet → zhoulab-1.cs.vt.edu:5559 → Caddy → 127.0.0.1:18765
                                             FastAPI
                                               └─ one asynchronous model subprocess
                                                   ├─ local predictor / VAE / dataset
                                                   └─ OpenAI designer and supervisor
```

The server's public IPv4 interface is `enp97s0f0`, address `128.173.237.13`. Public DNS resolves the hostname to that address; the bare `zhoulab-1` name resolves to a private lab address and is not the shareable URL. Port 5559 was free before deployment. Caddy listens on port 5559; the application remains on loopback. Other services, including port 5557, were left untouched.

Neither Nginx nor Caddy was preinstalled and unattended sudo is unavailable. A checksum-verified official Caddy **2.11.6** binary is installed in ignored `deploy/.runtime/bin`. Two **user systemd services** are enabled with **Linger=yes**, so the user manager and services start at boot and remain after logout. Both restart on failure. A host reboot was deliberately not performed.

The supplied units use read-only filesystem protection, private temporary directories, restrictive umasks, process limits, and no new privileges. The backend can write only its `.public` workspace (and private temporary directory), with an 8 GiB memory limit and two CPU cores' worth of quota. The proxy can write only `deploy/.runtime`. Both use the existing account, not root. The existing model and checkpoints are unchanged.

## Installation and operations

These files are configured for this checkout and existing `mat_env` environment on zhoulab-1. On a different machine, edit paths in the two service units and `public.env` first. The installer downloads the pinned Linux amd64 Caddy binary if missing, checks its SHA256, validates the proxy configuration, installs service links, enables lingering, and enables both services. It never overwrites or prints the API key.

From the repository root:

```bash
deploy/service.sh install
deploy/service.sh start
deploy/service.sh stop
deploy/service.sh restart
deploy/service.sh status
deploy/service.sh logs
```

`install` does not stop an existing server. Before the initial `start`, port 18765 must be free; stop only the old MetaSymbO process, after verifying no run is active. Repeat installation after editing a unit file to reload systemd. Use `restart` after app, environment, or proxy changes; static asset changes take effect on refresh.

Equivalent individual service operations:

```bash
systemctl --user restart metasymbo.service
systemctl --user restart metasymbo-proxy.service
systemctl --user is-enabled metasymbo.service metasymbo-proxy.service
loginctl show-user jianpengc -p Linger
journalctl --user -u metasymbo.service -u metasymbo-proxy.service -n 100 --no-pager
curl --fail http://127.0.0.1:18765/api/health
curl --fail http://zhoulab-1.cs.vt.edu:5559/api/health
ss -lntp '( sport = :5559 or sport = :18765 )'
```

Check run status before restarting. Browser disconnects do not stop jobs. A deliberate service stop terminates the worker cleanly; a crash/restart marks unfinished runs interrupted and never silently repeats paid API calls. Finished results and usage allowances survive restart. Use one web process: a workspace file lock rejects a second instance, and process-local scheduling deliberately supports only one worker.

The health endpoint returns only `status`, `web`, `prediction_available`, and `generation_available`. Availability means required dependencies, local data, and configuration are present; the endpoint makes no paid provider request. It does not report API-account credit or guarantee that the next design will succeed.

## Environment and credentials

`deploy/public.env` contains only non-secret configuration: public mode, permitted public hosts, the isolated workspace, CPU/thread settings, and plotting cache paths. The service directly invokes `/data/home/grads/jianpengc/miniconda3/envs/mat_env/bin/python`; shell activation is unnecessary.

The server reads `OPENAI_API_KEY` from the existing repository `.env`; an environment variable takes precedence. It parses the assignment without sourcing or executing the file. `.env` is ignored by Git and remains mode **0600**. Restart the backend after changing it. The proxy does not receive the key, and users never supply their own key.

The key is never included in HTML, JavaScript, API responses, reports, job files, command arguments, or logs. Workers receive it in their process environment. Legacy model console output is suppressed; structured logs retain timestamps, run IDs, stages, durations, exception types, and stack locations without provider exception text or source lines. Caddy access logs omit request/response headers and URIs, so cookies and query values are not recorded.

## Limits and multi-user behavior

| Protection | Public demo setting |
| --- | --- |
| Active expensive jobs | **1 total**, generation or prediction; excess requests get 409 and a friendly busy message |
| Prompt | 2,000 characters; bounded request bodies |
| Generation attempts | 1 by default, at most 2 |
| Runtime | 900 seconds per worker |
| Provider requests | At most 12 per generation; 60-second request timeout, no SDK retries, 2,500 output tokens per call |
| Internal geometry trials | At most 8 |
| Generation per IP | 2 per rolling hour, 6 per rolling day |
| Generation global | 24 per rolling day, including failed submissions that started |
| Prediction per IP / global | 6/hour, 24/day per IP; 120/day global |
| Import per IP / global | 10/hour, 30/day per IP; 200/day global |
| API requests | 120/minute per IP, 2,400/minute total; health excluded |
| Upload | 4 MiB compressed, 20 MiB uncompressed; numeric geometry only, never pickle |
| Public storage | New jobs/imports stop at 1 GiB or 8,000 workspace files |

Daily and expensive-action quotas are persisted atomically in `.public/limits.json`; restart or clearing browser cookies does not reset them. Minute-scale read counters reset each minute or process restart. IP limits apply to shared campus/VPN networks as a group. Caddy overwrites forwarding headers and Uvicorn trusts only the loopback proxy, preventing visitors from choosing their rate-limit identity. For a large launch, provider-side project spending limits are also useful; IP quotas are conservative abuse controls, not an account system or a distributed denial-of-service shield.

Browser sessions use random 256-bit HttpOnly, SameSite=Lax cookies. Public runs, imports, and generated candidates are visible only to the same browser session. Different sessions cannot list or open one another's private outputs, even with a run/candidate ID. Curated repository results are shared. Clearing the session cookie loses access to previous private runs; export important results. Cookies become Secure when served through HTTPS.

Run IDs and output names contain full UUIDs. Request-selected paths are prohibited. Atomic file replacement prevents partially written JSON/results from being served. Import filenames are generated by the server. Candidates are resolved through indexed IDs, rechecked against permitted directories, and protected against symlink escapes. Unsafe geometry, oversized NPZ members, duplicate fields, non-finite numbers, and object-valued geometry are rejected.

## Routes and proxy

| Route | Purpose and protection |
| --- | --- |
| `/`, `/static/app.js`, `/static/styles.css`, `/plotly.min.js` | Fixed UI files only; no repository file server or CDN |
| `GET /api/health` | Cheap operational booleans, no secrets or paths |
| `GET /api/status` | Safe capability flags, limits and busy state |
| `GET /api/candidates`, `/{id}`, `/{id}/download` | Bounded pagination, constrained IDs/directories, session ownership for public outputs |
| `POST /api/import` | Size-limited numeric NPZ validation, generated names, quota and ownership |
| `POST /api/runs` | Strict schema, finite conditions, limits, one worker, no shell execution |
| `GET /api/runs`, `/{id}` | Session-filtered status/progress; at most 100 recent runs in the list |

Unknown paths return 404. Documentation/schema routes are disabled. Cross-origin writes and unknown Host headers are rejected. The frontend exclusively uses same-origin `/api/...` URLs. The proxy has no `file_server` directive and forwards only the application routes. `.env`, checkpoints, source files, and `/etc/passwd` are not served.

`Caddyfile` supplies request/header/body timeouts, a 4 MiB body limit, security headers, safe error responses, and explicit trusted forwarding. Jobs remain asynchronous, so the 30-second upstream response-header timeout does not limit a 15-minute job. The frontend polls progress; it does not hold a generation HTTP request open. Caddy supports streaming proxy responses and WebSockets if later needed; the current UI does not depend on them.

## Logs and cleanup

Application and worker logs are in the user journal; inspect with `deploy/service.sh logs`. Journald retention follows host policy. Caddy access logs live at `deploy/.runtime/access.log`, rotate at 10 MiB, and retain at most five rotated files for seven days. These files and browser test session artifacts are ignored and owner-protected.

**Public run retention target: seven days**, with explicit operator cleanup. No automatic task deletes research data. Cleanup only considers completed/failed/interrupted public jobs and public candidate metadata older than the retention window; it skips active jobs and never traverses the permanent `results` or private `.local` directories.

```bash
python deploy/cleanup_public_runs.py                    # preview; deletes nothing
systemctl --user stop metasymbo.service                 # wait for jobs first
python deploy/cleanup_public_runs.py --days 7 --apply
systemctl --user start metasymbo.service
```

The cleanup script refuses `--apply` while the application holds the workspace lock. Run the preview weekly, apply after draining current work, and preserve important public results by exporting them before cleanup. Do not delete `limits.json` to make room; it preserves spending limits.

## Validation

On 2026-10-02:

- 15 Python tests passed, including geometry/parser regressions, public session isolation, persisted quotas, body/prompt limits, traversal/symlink rejection, timeout, provider budgets, diagnostics, and cleanup.
- JavaScript, Python and shell syntax checks and Caddy/systemd configuration validation passed.
- Chromium loaded the **real public URL** without any tunnel and checked the lattice viewer, library, comparison, import/export, review, progress, signed predictions, supervisor score, saved result reopening, refresh, and a 390 px mobile viewport without horizontal overflow.
- A live local prediction and one bounded OpenAI-backed generation completed through the public API. Concurrent duplicate submissions returned the expected busy response.
- Production rate limiting returned 429 after the import allowance was consumed, even with spoofed forwarding headers. Raw traversal and private-file requests all returned 404. The actual credential was checked in memory and was absent from static assets, public JSON state, and both service journals.
- Restarting both services preserved browser sessions, predictions, supervisor evidence, saved geometry, and exports. A newly submitted prediction also completed after restart. Killing only the idle backend process verified automatic failure recovery and proxy reconnection. No unrelated process was stopped.
- Independent off-host HTTP probes returned **200** for [the homepage](https://check-host.net/check-report/4ea587a9k513), [JavaScript](https://check-host.net/check-report/4ea599d1k699), [the candidate API](https://check-host.net/check-report/4ea599dfka56), and [health from three countries](https://check-host.net/check-report/4ea550fakc13). The [.env probe returned 404](https://check-host.net/check-report/4ea599e7kda5).

The full Chromium/model tests run on zhoulab-1 against the public hostname; off-host probes independently verify Internet HTTP reachability, not WebGL execution or paid job submission in a remote browser. No host reboot is required or performed.

Reproduce Python tests from the repository root:

```bash
conda activate mat_env
python -m unittest WebInterface.test_app -v
python -m compileall -q WebInterface deploy
node --check WebInterface/static/app.js
bash -n deploy/service.sh
```

Browser tests require Node, Playwright and Chromium **only for testing**, not for serving. The existing temporary test installation can be used on this machine:

```bash
export NODE_PATH=/tmp/metasymbo-browser/node_modules
export PLAYWRIGHT_BROWSERS_PATH=/tmp/metasymbo-browsers
export METASYMBO_TEST_URL=http://zhoulab-1.cs.vt.edu:5559
node WebInterface/browser_test.cjs          # browsing and security, no paid calls
node WebInterface/browser_test.cjs --jobs   # one prediction + one paid generation
deploy/service.sh restart
node WebInterface/browser_test.cjs --reopen # prior saved session/results after restart
```

The smoke-test artifacts, screenshots and private browser session are stored in ignored `deploy/.runtime`. Test submissions count against normal quotas. Reinstall the temporary Playwright tools if `/tmp` is cleared.

## Remaining HTTPS step

The public hostname already exists, but no usable certificate or unattended root/DNS administration is available to this account. HTTP is fully deployed; **HTTPS is not configured**. Host firewall rules could not be inspected without root, but successful off-host probes establish that port 5559 is reachable. No firewall change is currently needed for HTTP.

An administrator can issue a trusted certificate for `zhoulab-1.cs.vt.edu` using the institution's certificate service or ACME. For example, if the administrator allows TCP 80 for HTTP validation and confirms it is unused:

```bash
sudo certbot certonly --standalone --preferred-challenges http -d zhoulab-1.cs.vt.edu
sudo install -d -m 700 -o jianpengc -g Grads /data/home/grads/jianpengc/projects/materials/MetaSymbO/deploy/.runtime/tls
sudo install -m 600 -o jianpengc -g Grads /etc/letsencrypt/live/zhoulab-1.cs.vt.edu/fullchain.pem /data/home/grads/jianpengc/projects/materials/MetaSymbO/deploy/.runtime/tls/fullchain.pem
sudo install -m 600 -o jianpengc -g Grads /etc/letsencrypt/live/zhoulab-1.cs.vt.edu/privkey.pem /data/home/grads/jianpengc/projects/materials/MetaSymbO/deploy/.runtime/tls/privkey.pem
```

Install Certbot through the administrator's package policy if absent. Port 80 admission is an administrator/network task; do not change unrelated firewall rules or require it for the current HTTP demo. A DNS challenge is an alternative when the administrator controls the DNS zone. Certificate renewal must repeat the protected copy and restart the proxy; configure that as the certificate renewal deploy hook.

Then change the site address in `deploy/Caddyfile` from `http://:5559` to `https://zhoulab-1.cs.vt.edu:5559` and add this inside the site block (keep `auto_https off` for this manually supplied certificate):

```caddyfile
tls .runtime/tls/fullchain.pem .runtime/tls/privkey.pem
```

Validate and restart the proxy as jianpengc:

```bash
cd /data/home/grads/jianpengc/projects/materials/MetaSymbO/deploy
.runtime/bin/caddy validate --config Caddyfile --adapter caddyfile
systemctl --user restart metasymbo-proxy.service
curl --fail https://zhoulab-1.cs.vt.edu:5559/api/health
```

Verify that last HTTPS URL from an off-host browser/probe without disabling certificate verification, rerun browser checks with the HTTPS URL, and update the shareable link. The application already accepts the hostname and trusts Caddy's scheme header. No API key belongs in TLS files, proxy configuration, or browser assets.

References: [Caddy installation](https://caddyserver.com/docs/install), [reverse proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy), [external probe API](https://check-host.net/about/api).
