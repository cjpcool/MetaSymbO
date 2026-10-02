#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-status}" in
  install)
    mkdir -p "$repo_dir/WebInterface/.public" "$repo_dir/deploy/.runtime" "$HOME/.config/systemd/user"
    chmod 700 "$repo_dir/WebInterface/.public" "$repo_dir/deploy/.runtime"
    chmod 600 "$repo_dir/.env"
    if [[ ! -x "$repo_dir/deploy/.runtime/bin/caddy" ]]; then
      archive_dir="$(mktemp -d)"
      trap 'rm -rf -- "$archive_dir"' EXIT
      curl -fsSL --retry 2 https://github.com/caddyserver/caddy/releases/download/v2.11.6/caddy_2.11.6_linux_amd64.tar.gz -o "$archive_dir/caddy.tar.gz"
      printf '%s  %s\n' 22c84f8d2d4e4e0e2d422f8049fdd0fc1ed8d5665d0fe166f506c7fd863b4555 "$archive_dir/caddy.tar.gz" | sha256sum --check --status
      mkdir -p "$repo_dir/deploy/.runtime/bin"
      tar -xzf "$archive_dir/caddy.tar.gz" -C "$repo_dir/deploy/.runtime/bin" --no-same-owner caddy
      chmod 755 "$repo_dir/deploy/.runtime/bin/caddy"
    fi
    (cd "$repo_dir/deploy" && .runtime/bin/caddy validate --config Caddyfile --adapter caddyfile)
    loginctl enable-linger "$USER"
    ln -sfn "$repo_dir/deploy/metasymbo.service" "$HOME/.config/systemd/user/metasymbo.service"
    ln -sfn "$repo_dir/deploy/metasymbo-proxy.service" "$HOME/.config/systemd/user/metasymbo-proxy.service"
    systemctl --user daemon-reload
    systemctl --user enable metasymbo.service metasymbo-proxy.service
    ;;
  start|restart)
    systemctl --user "$1" metasymbo.service metasymbo-proxy.service
    ;;
  stop)
    systemctl --user stop metasymbo-proxy.service metasymbo.service
    ;;
  status)
    systemctl --user --no-pager status metasymbo.service metasymbo-proxy.service
    ;;
  logs)
    journalctl --user -u metasymbo.service -u metasymbo-proxy.service -n 100 -f
    ;;
  *)
    echo "Usage: $0 {install|start|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac
