#!/usr/bin/env bash
set -euo pipefail

(
ROOT="${PWD}"
OUT="${ROOT}/data/raw/source_archives"
mkdir -p "$OUT"

no_vpn_env() {
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u FTP_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u ftp_proxy \
    NO_PROXY='*' no_proxy='*' \
    "$@"
}

default_iface="$(
  ip route show default 2>/dev/null | awk 'NR == 1 {for (i = 1; i <= NF; i++) if ($i == "dev") print $(i + 1)}'
)"
if [[ -z "$default_iface" ]]; then
  echo "Cannot verify default route. Refusing to download because no-VPN routing cannot be checked." >&2
  exit 2
fi
if [[ "$default_iface" =~ ^(tun|wg|ppp) ]]; then
  cat >&2 <<EOF
Default route is $default_iface, which looks like a VPN interface.
Google Drive downloads cannot be bound to a non-VPN interface per process here.
Stopping without changing global network settings.
EOF
  exit 3
fi

CURL_IFACE_ARGS=()
if [[ -n "${NONVPN_IFACE:-}" ]]; then
  CURL_IFACE_ARGS=(--interface "$NONVPN_IFACE")
fi

CURL_TLS_ARGS=()
if [[ "${ALLOW_INSECURE_TLS:-0}" == "1" ]]; then
  CURL_TLS_ARGS=(--insecure)
  echo "ALLOW_INSECURE_TLS=1 is set: curl will skip TLS certificate verification." >&2
fi

download_gdrive() {
  local file_id="$1"
  local out_file="$2"
  local tmp_file cookie_file confirm direct_url
  if [[ ! -s "$out_file" ]]; then
    tmp_file="$(mktemp)"
    cookie_file="$(mktemp)"
    (
      no_vpn_env curl --noproxy '*' "${CURL_IFACE_ARGS[@]}" \
        "${CURL_TLS_ARGS[@]}" \
        -L --fail --retry 5 -c "$cookie_file" \
        "https://drive.google.com/uc?export=download&id=${file_id}" \
        -o "$tmp_file"
    )

    if head -c 512 "$tmp_file" | grep -qi '<html'; then
      confirm="$(
        ( grep -aoE 'confirm=([0-9A-Za-z_%.-]+)' "$tmp_file" || true ) |
          head -n 1 |
          sed 's/^confirm=//'
      )"
      direct_url="$(
        ( grep -aoE 'https://drive\.usercontent\.google\.com/download[^"]+' "$tmp_file" || true ) |
          head -n 1 |
          sed 's/&amp;/\&/g'
      )"
      if [[ -n "$direct_url" ]]; then
        (
          no_vpn_env curl --noproxy '*' "${CURL_IFACE_ARGS[@]}" \
            "${CURL_TLS_ARGS[@]}" \
            -L --fail --retry 5 -b "$cookie_file" "$direct_url" -o "$out_file"
        )
      elif [[ -n "$confirm" ]]; then
        (
          no_vpn_env curl --noproxy '*' "${CURL_IFACE_ARGS[@]}" \
            "${CURL_TLS_ARGS[@]}" \
            -L --fail --retry 5 -b "$cookie_file" \
            "https://drive.google.com/uc?export=download&confirm=${confirm}&id=${file_id}" \
            -o "$out_file"
        )
      else
        echo "Google Drive did not return a direct download token for ${file_id}." >&2
        echo "Saved response for inspection: ${tmp_file}" >&2
        rm -f "$cookie_file"
        return 5
      fi
      rm -f "$tmp_file" "$cookie_file"
    else
      mv "$tmp_file" "$out_file"
      rm -f "$cookie_file"
    fi
  fi
}

download_url() {
  local url="$1"
  local out_file="$2"
  if [[ ! -s "$out_file" ]]; then
    (
      no_vpn_env curl --noproxy '*' "${CURL_IFACE_ARGS[@]}" \
        "${CURL_TLS_ARGS[@]}" \
        -L --fail --retry 5 --continue-at - "$url" -o "$out_file"
    )
  fi
}

echo "Default route interface: $default_iface"
echo "Downloading to: $OUT"

portal_check="$(mktemp)"
portal_status="$(
  (
    no_vpn_env curl --noproxy '*' "${CURL_IFACE_ARGS[@]}" "${CURL_TLS_ARGS[@]}" \
      -L -sS -o "$portal_check" -w '%{http_code}' \
      'https://www.gstatic.com/generate_204'
  ) || true
)"
if [[ "$portal_status" != "204" ]]; then
  echo "No-VPN route does not have direct internet access; expected HTTP 204, got ${portal_status:-curl-error}." >&2
  echo "This is commonly a captive portal or campus login page. Stopping without using VPN/proxy." >&2
  echo "Probe response saved for inspection: $portal_check" >&2
  exit 6
fi
rm -f "$portal_check"

# [11] LOL / RetinexNet
download_gdrive "157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB" "$OUT/LOLdataset.zip"

# [53] Deep HDR
download_url "https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Trainingset.zip" "$OUT/SIGGRAPH17_HDR_Trainingset.zip"
download_url "https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Testset.zip" "$OUT/SIGGRAPH17_HDR_Testset.zip"

# [54] SICE
download_gdrive "1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN" "$OUT/SICE_Dataset_Part1.rar"
download_gdrive "16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN" "$OUT/SICE_Dataset_Part2.rar"

# [57] VisDrone DET train/val
download_gdrive "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn" "$OUT/VisDrone2019-DET-train.zip"
download_gdrive "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59" "$OUT/VisDrone2019-DET-val.zip"

cat <<'EOF'

Downloaded primary public archives.

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
