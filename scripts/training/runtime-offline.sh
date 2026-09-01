# Shared offline runtime boundary. This file is sourced after `repo_root` is set.
if [[ -z "${repo_root:-}" ]]; then
  echo "FloodSight offline runtime requires repo_root." >&2
  return 2
fi

runtime_assets="/data/floodsight-workspace/floodsight-cache/ml/runtime-assets/ultralytics-v8.3.222-v1"
runtime_asset_hashes="${repo_root}/ml/training/ultralytics-runtime-assets-v1.sha256"
runtime_settings_template="${repo_root}/ml/training/ultralytics-settings-v1.json"
runtime_cache="${FLOODSIGHT_ML_RUNTIME_CACHE:-/data/floodsight-workspace/floodsight-cache/ml/runtime/locked-v1}"

if [[ ! -d "${runtime_assets}" || -L "${runtime_assets}" ]]; then
  echo "Audited Ultralytics runtime assets are missing or unsafe: ${runtime_assets}" >&2
  return 2
fi
if [[ -L "${runtime_assets}/Arial.ttf" || -L "${runtime_assets}/yolo11n.pt" ]]; then
  echo "Audited Ultralytics runtime assets must not be symbolic links." >&2
  return 2
fi
if ! (cd "${runtime_assets}" && sha256sum --check --strict "${runtime_asset_hashes}") >/dev/null; then
  echo "Audited Ultralytics runtime assets failed SHA-256 verification." >&2
  return 2
fi

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export ULTRALYTICS_OFFLINE=true
export YOLO_OFFLINE=true
export PIP_NO_INDEX=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
if [[ "${CUBLAS_WORKSPACE_CONFIG}" != ":4096:8" && "${CUBLAS_WORKSPACE_CONFIG}" != ":16:8" ]]; then
  echo "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8." >&2
  return 2
fi
export HF_HOME="${runtime_cache}/huggingface"
export TORCH_HOME="${runtime_cache}/torch"
# Ultralytics appends its own `Ultralytics` leaf to this parent directory.
export YOLO_CONFIG_DIR="${runtime_cache}/yolo-config"
ultralytics_user_config="${YOLO_CONFIG_DIR}/Ultralytics"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${ultralytics_user_config}"

runtime_settings="${ultralytics_user_config}/settings.json"
runtime_font="${ultralytics_user_config}/Arial.ttf"

# Ultralytics serializes its SettingsManager JSON without a terminal newline.
# Accept only the audited template bytes or that one exact, deterministic
# serialization difference; every key/value byte must otherwise remain equal.
runtime_settings_match() {
  local template="$1"
  local candidate="$2"
  local template_bytes
  local candidate_bytes
  local final_template_byte
  if cmp -s "${template}" "${candidate}"; then
    return 0
  fi
  template_bytes="$(wc -c < "${template}")"
  candidate_bytes="$(wc -c < "${candidate}")"
  if (( template_bytes != candidate_bytes + 1 )); then
    return 1
  fi
  final_template_byte="$(tail -c 1 -- "${template}" | od -An -tu1 | tr -d '[:space:]')"
  if [[ "${final_template_byte}" != "10" ]]; then
    return 1
  fi
  cmp -s <(head -c "${candidate_bytes}" -- "${template}") "${candidate}"
}

if [[ -e "${runtime_settings}" || -L "${runtime_settings}" ]]; then
  if [[ -L "${runtime_settings}" ]] || \
    ! runtime_settings_match "${runtime_settings_template}" "${runtime_settings}"; then
    echo "Refusing drifted Ultralytics runtime settings: ${runtime_settings}" >&2
    return 2
  fi
else
  install -m 0444 "${runtime_settings_template}" "${runtime_settings}"
fi
if [[ -e "${runtime_font}" || -L "${runtime_font}" ]]; then
  if [[ -L "${runtime_font}" ]] || ! cmp -s "${runtime_assets}/Arial.ttf" "${runtime_font}"; then
    echo "Refusing drifted Ultralytics runtime font: ${runtime_font}" >&2
    return 2
  fi
else
  install -m 0444 "${runtime_assets}/Arial.ttf" "${runtime_font}"
fi
