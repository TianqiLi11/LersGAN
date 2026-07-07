#!/usr/bin/env bash
set -euo pipefail

(
ROOT="${PWD}"
OUT="${ROOT}/data/raw/source_archives"
mkdir -p "$OUT"

CLASH_HTTP_PROXY="${CLASH_HTTP_PROXY:-http://127.0.0.1:17890}"
CLASH_HTTPS_PROXY="${CLASH_HTTPS_PROXY:-$CLASH_HTTP_PROXY}"
CLASH_ALL_PROXY="${CLASH_ALL_PROXY:-socks5h://127.0.0.1:17890}"
CLASH_API="${CLASH_API:-}"
CLASH_SELECTOR="${CLASH_SELECTOR:-}"
CLASH_NODE="${CLASH_NODE:-}"

api_tmp=""
selector_previous=""
selector_switched=0

cleanup() {
  local status=$?
  if [[ "$selector_switched" == "1" ]]; then
    if ! clash_put_selector "$CLASH_SELECTOR" "$selector_previous"; then
      echo "Failed to restore Clash selector '$CLASH_SELECTOR' to '$selector_previous'." >&2
      echo "Stop here and restore it manually before continuing." >&2
      exit 9
    fi
    echo "Restored Clash selector '$CLASH_SELECTOR' to '$selector_previous'."
  fi
  exit "$status"
}
trap cleanup EXIT

download_env() {
  env \
    -u NO_PROXY -u no_proxy -u FTP_PROXY -u ftp_proxy \
    HTTP_PROXY="$CLASH_HTTP_PROXY" HTTPS_PROXY="$CLASH_HTTPS_PROXY" ALL_PROXY="$CLASH_ALL_PROXY" \
    http_proxy="$CLASH_HTTP_PROXY" https_proxy="$CLASH_HTTPS_PROXY" all_proxy="$CLASH_ALL_PROXY" \
    "$@"
}

local_env() {
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u FTP_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u ftp_proxy \
    "$@"
}

urlencode() {
  local_env python3 - "$1" <<'PY'
import sys
from urllib.parse import quote
print(quote(sys.argv[1], safe=""))
PY
}

detect_clash_api() {
  if [[ -n "$CLASH_API" ]]; then
    echo "$CLASH_API"
    return 0
  fi
  for candidate in "http://127.0.0.1:19090" "http://127.0.0.1:9090"; do
    if local_env curl -fsS -m 2 "$candidate/proxies" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

clash_get_selector_now() {
  local selector="$1"
  local encoded
  encoded="$(urlencode "$selector")"
  local_env python3 - "$api_tmp/proxies/$encoded" <<'PY'
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=5) as resp:
    data = json.load(resp)
print(data.get("now", ""))
PY
}

clash_put_selector() {
  local selector="$1"
  local node="$2"
  local encoded payload
  encoded="$(urlencode "$selector")"
  payload="$(local_env python3 - "$node" <<'PY'
import json
import sys
print(json.dumps({"name": sys.argv[1]}, ensure_ascii=False))
PY
)"
  local_env curl -fsS -m 5 -X PUT "$api_tmp/proxies/$encoded" \
    -H 'Content-Type: application/json' \
    --data "$payload" >/dev/null
}

if [[ -n "$CLASH_SELECTOR" || -n "$CLASH_NODE" ]]; then
  if [[ -z "$CLASH_SELECTOR" || -z "$CLASH_NODE" ]]; then
    echo "Set both CLASH_SELECTOR and CLASH_NODE to switch a Clash selector." >&2
    exit 10
  fi
  if ! api_tmp="$(detect_clash_api)"; then
    echo "Clash API is unavailable; cannot switch selector safely." >&2
    exit 11
  fi
  selector_previous="$(clash_get_selector_now "$CLASH_SELECTOR")"
  if [[ -z "$selector_previous" ]]; then
    echo "Cannot read current selector state for '$CLASH_SELECTOR'; refusing to switch." >&2
    exit 12
  fi
  echo "Current Clash selector '$CLASH_SELECTOR': $selector_previous"
  clash_put_selector "$CLASH_SELECTOR" "$CLASH_NODE"
  selector_switched=1
  echo "Switched Clash selector '$CLASH_SELECTOR' to '$CLASH_NODE'."
fi

probe_code="$(
  ( download_env curl -fsS -m 15 -L -o /tmp/lersgan_clash_probe.out -w '%{http_code}' \
      'https://www.gstatic.com/generate_204' ) || true
)"
if [[ "$probe_code" != "204" ]]; then
  echo "Clash proxy probe failed: expected HTTP 204, got ${probe_code:-curl-error}." >&2
  exit 13
fi

is_html_or_login() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    return 0
  fi
  if head -c 4096 "$path" | grep -Eqi '<html|Dr\.COM|WebLogin|login|captcha|DOCTYPE html'; then
    return 0
  fi
  return 1
}

validate_archive() {
  local path="$1"
  local kind="$2"
  if is_html_or_login "$path"; then
    echo "Invalid download, got HTML/login page: $path" >&2
    rm -f "$path"
    return 20
  fi

  case "$kind" in
    zip)
      local_env python3 - "$path" <<'PY'
import sys
import zipfile
path = sys.argv[1]
if not zipfile.is_zipfile(path):
    raise SystemExit(f"Not a valid zip archive: {path}")
PY
      ;;
    rar)
      local_env python3 - "$path" <<'PY'
import sys
path = sys.argv[1]
with open(path, "rb") as f:
    sig = f.read(8)
if not (sig.startswith(b"Rar!\x1a\x07\x00") or sig.startswith(b"Rar!\x1a\x07\x01\x00")):
    raise SystemExit(f"Not a valid rar archive: {path}")
PY
      if command -v unrar >/dev/null 2>&1; then
        local_env unrar t -idq "$path" >/dev/null
      elif command -v 7z >/dev/null 2>&1; then
        local_env 7z t -bd "$path" >/dev/null
      elif command -v bsdtar >/dev/null 2>&1; then
        local_env bsdtar -tf "$path" >/dev/null
      elif command -v unar >/dev/null 2>&1; then
        local_env unar -t "$path" >/dev/null
      else
        echo "Warning: no RAR tester found; checked only RAR magic header for $path." >&2
      fi
      ;;
    *)
      echo "Unknown archive kind: $kind" >&2
      return 21
      ;;
  esac
}

archive_is_complete() {
  local out_file="$1"
  local kind="$2"
  [[ -s "$out_file" ]] || return 1
  if is_html_or_login "$out_file"; then
    echo "Invalid download, got HTML/login page: $out_file" >&2
    rm -f "$out_file"
    return 20
  fi
  validate_archive "$out_file" "$kind"
}

download_gdrive_range() {
  local follow_url="$1"
  local cookie_file="$2"
  local out_file="$3"
  local headers_file probe_file total start end chunk_file chunk_size expected actual

  headers_file="$(mktemp)"
  probe_file="$(mktemp)"
  download_env curl -L --fail --retry 5 -b "$cookie_file" --range 0-0 \
    -D "$headers_file" "$follow_url" -o "$probe_file"
  if is_html_or_login "$probe_file"; then
    echo "Google Drive range probe returned HTML/login page for $out_file." >&2
    rm -f "$headers_file" "$probe_file"
    return 20
  fi
  total="$(local_env python3 - "$headers_file" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="ignore")
matches = re.findall(r"(?im)^content-range:\s*bytes\s+\d+-\d+/(\d+)\s*$", text)
print(matches[-1] if matches else "")
PY
)"
  rm -f "$headers_file" "$probe_file"
  if [[ -z "$total" ]]; then
    echo "Could not determine Google Drive file size from range probe for $out_file." >&2
    return 31
  fi

  chunk_size="${GDRIVE_CHUNK_BYTES:-268435456}"
  if [[ -s "$out_file" ]]; then
    start="$(local_env python3 - "$out_file" <<'PY'
import os
import sys
print(os.path.getsize(sys.argv[1]))
PY
)"
    if (( start > total )); then
      echo "Existing file is larger than remote size; restarting $out_file." >&2
      rm -f "$out_file"
      start=0
    fi
  else
    : > "$out_file"
    start=0
  fi

  while (( start < total )); do
    end=$(( start + chunk_size - 1 ))
    if (( end >= total )); then
      end=$(( total - 1 ))
    fi
    chunk_file="$(mktemp)"
    echo "Downloading $(basename "$out_file") bytes ${start}-${end}/${total}"
    download_env curl -L --fail --retry 5 -b "$cookie_file" --range "${start}-${end}" \
      "$follow_url" -o "$chunk_file"
    if is_html_or_login "$chunk_file"; then
      echo "Invalid download chunk, got HTML/login page for $out_file bytes ${start}-${end}." >&2
      rm -f "$chunk_file"
      return 20
    fi
    expected=$(( end - start + 1 ))
    actual="$(local_env python3 - "$chunk_file" <<'PY'
import os
import sys
print(os.path.getsize(sys.argv[1]))
PY
)"
    if (( actual != expected )); then
      echo "Short Google Drive chunk for $out_file: expected $expected bytes, got $actual." >&2
      rm -f "$chunk_file"
      return 32
    fi
    cat "$chunk_file" >> "$out_file"
    rm -f "$chunk_file"
    start=$(( end + 1 ))
  done
}

download_gdrive() {
  local file_id="$1"
  local out_file="$2"
  local kind="$3"
  local tmp_file cookie_file follow_url
  if [[ -s "$out_file" ]]; then
    if archive_is_complete "$out_file" "$kind"; then
      return 0
    fi
    if [[ ! -e "$out_file" ]]; then
      return 20
    fi
    echo "Existing archive is incomplete; attempting resume: $out_file"
  fi

  tmp_file="$(mktemp)"
  cookie_file="$(mktemp)"
  download_env curl -L --fail --retry 5 -c "$cookie_file" \
    "https://drive.google.com/uc?export=download&id=${file_id}" \
    -o "$tmp_file"

  if is_html_or_login "$tmp_file"; then
    follow_url="$(local_env python3 - "$tmp_file" "$file_id" <<'PY'
import html
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="ignore")
file_id = sys.argv[2]

patterns = [
    r'href="(https://drive\.usercontent\.google\.com/download[^"]+)"',
    r'"downloadUrl":"([^"]+)"',
    r"'downloadUrl':'([^']+)'",
]
for pattern in patterns:
    match = re.search(pattern, text)
    if match:
        print(html.unescape(match.group(1)).replace("\\u003d", "=").replace("\\u0026", "&"))
        raise SystemExit

confirm = re.search(r'confirm=([0-9A-Za-z_%.-]+)', text)
uuid = re.search(r'uuid=([0-9A-Za-z_%.-]+)', text)
if confirm and uuid:
    print(
        "https://drive.usercontent.google.com/download?"
        f"id={file_id}&export=download&confirm={confirm.group(1)}&uuid={uuid.group(1)}"
    )
elif confirm:
    print(
        "https://drive.google.com/uc?"
        f"export=download&confirm={confirm.group(1)}&id={file_id}"
    )
else:
    action = re.search(r'<form[^>]+action="([^"]+)"', text)
    hidden = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', text))
    if action and hidden.get("confirm"):
        from urllib.parse import urlencode
        params = {
            "id": hidden.get("id", file_id),
            "export": hidden.get("export", "download"),
            "confirm": hidden["confirm"],
        }
        if hidden.get("uuid"):
            params["uuid"] = hidden["uuid"]
        print(html.unescape(action.group(1)) + "?" + urlencode(params))
PY
)"
    if [[ -z "$follow_url" ]]; then
      echo "Google Drive returned HTML but no downloadable URL for ${file_id}." >&2
      echo "Response saved for inspection: $tmp_file" >&2
      rm -f "$cookie_file"
      return 30
    fi
    if [[ -s "$out_file" ]]; then
      download_gdrive_range "$follow_url" "$cookie_file" "$out_file"
    else
      download_gdrive_range "$follow_url" "$cookie_file" "$out_file"
    fi
    rm -f "$tmp_file" "$cookie_file"
  else
    if [[ -s "$out_file" ]]; then
      echo "Google Drive returned direct payload but a partial file already exists; restarting $out_file." >&2
      rm -f "$out_file"
    fi
    mv "$tmp_file" "$out_file"
    rm -f "$cookie_file"
  fi
  validate_archive "$out_file" "$kind"
}

download_url() {
  local url="$1"
  local out_file="$2"
  local kind="$3"
  if [[ ! -s "$out_file" ]]; then
    download_env curl -L --fail --retry 5 --continue-at - "$url" -o "$out_file"
  fi
  validate_archive "$out_file" "$kind"
}

echo "Using Clash proxies:"
echo "  HTTP_PROXY=$CLASH_HTTP_PROXY"
echo "  ALL_PROXY=$CLASH_ALL_PROXY"
echo "Downloading to: $OUT"

# [11] LOL / RetinexNet
download_gdrive "157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB" "$OUT/LOLdataset.zip" zip

# [53] Deep HDR
download_url "https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Trainingset.zip" "$OUT/SIGGRAPH17_HDR_Trainingset.zip" zip
download_url "https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Testset.zip" "$OUT/SIGGRAPH17_HDR_Testset.zip" zip

# [54] SICE
download_gdrive "1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN" "$OUT/SICE_Dataset_Part1.rar" rar
download_gdrive "16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN" "$OUT/SICE_Dataset_Part2.rar" rar

# [57] VisDrone DET train/val
download_gdrive "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn" "$OUT/VisDrone2019-DET-train.zip" zip
download_gdrive "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59" "$OUT/VisDrone2019-DET-val.zip" zip

cat <<'EOF'

Downloaded primary public archives through Clash.

RAISE is form-gated:
  https://loki.disi.unitn.it/RAISE/download.html
Use only a small/sample package as optional normal_light supplement; do not download the 350GB full set.

NWPU-RESISC45 options:
  HuggingFace mirror: https://huggingface.co/datasets/jonathan-roberts1/NWPU-RESISC45
  OneDrive: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
Place the downloaded archive or extracted images under data/raw/source_archives or data/raw/source_extracted/NWPU_RESISC45.

This rebuild is source-exact but file-list-not-exact. It is not the authors' original split.
EOF
)
