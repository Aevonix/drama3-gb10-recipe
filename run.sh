#!/usr/bin/env bash
# Drama 3 on a single GB10, from a clean box to a serving endpoint, in one command.
#
#   ./run.sh            set up what is missing, then serve (idempotent)
#   ./run.sh --check     report what is in place and whether the service is healthy
#   ./run.sh --stop      stop the service
#
# Every step is skipped if it is already done, so re-running after a failure
# resumes rather than starting over. Override anything:
#
#   DRAMA3_HOME=/srv/drama3    where the env, weights and logs live (~/drama3)
#   DRAMA3_HOST / DRAMA3_PORT  bind address (127.0.0.1 / 18020)
#   DRAMA3_MTP_SCOPE           residual_layers (default) or "all"
#   DRAMA3_TOWER               1 to opt into the vendor fused tower instead of
#                              MTP FP8 (they are mutually exclusive; see README)
#   DRAMA3_BACKBONE            1 (default) for online FP8 on the backbone, 0 off.
#                              This is the big one for speed: decode 28.58 ->
#                              38.85 tok/s. It also raises the LOAD-time memory
#                              peak, because the conversion runs after the bf16
#                              weights are resident. Set 0 on a tight box.
#   DRAMA3_BACKBONE_SCOPE      comma list of attention, shared_experts,
#                              linear_attn (default: all three)
set -euo pipefail

HOME_DIR="${DRAMA3_HOME:-$HOME/drama3}"
HOST="${DRAMA3_HOST:-127.0.0.1}"
PORT="${DRAMA3_PORT:-18020}"
SCOPE="${DRAMA3_MTP_SCOPE:-residual_layers}"
TOWER="${DRAMA3_TOWER:-0}"
BACKBONE="${DRAMA3_BACKBONE:-1}"
BACKBONE_SCOPE="${DRAMA3_BACKBONE_SCOPE:-attention,shared_experts,linear_attn}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$HOME_DIR/conda-env"
SRC_DIR="$HOME_DIR/sglang-omni"
WEIGHTS="$HOME_DIR/drama-3"
CONFIG="$HOME_DIR/drama3.yaml"
LOG="$HOME_DIR/serve.log"
PIDFILE="$HOME_DIR/serve.pid"
STAMPS="$HOME_DIR/.stamps"

SGLANG_OMNI_REV=2cfb3b67becb437b70751defe37e63d3af4eedb0
DRAMA3_REV=dd64a85cd82e9521f7202610ede97b367fc83946
MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
done_with() { [ -f "$STAMPS/$1" ]; }
mark()      { mkdir -p "$STAMPS"; : >> "$STAMPS/$1"; }

model_dir() {
  "$ENV_DIR/bin/python" - <<'PY'
import os, sglang_omni
print(os.path.join(os.path.dirname(sglang_omni.__file__), "models", "fishaudio_drama3"))
PY
}

health_json() {
  curl -fsS --max-time 5 "http://$HOST:$PORT/health" 2>/dev/null || true
}

caveat() {
  cat <<'TXT'

    The sm_121 kernel opt-in is UNVALIDATED BY THE VENDOR.
    Upstream gates the fused tower and the Triton codebook sampler on an exact
    match with compute capability 9.0, and this recipe opts past that gate. The
    kernels passed every shape check here and the output held up, but nobody has
    validated them on this architecture. If you care about that, run without
    DRAMA3_ALLOW_UNTESTED_GPU: everything else in this recipe still applies.

TXT
}

# ---------------------------------------------------------------- preflight ---

preflight() {
  [ "$(uname -m)" = "aarch64" ] || note "warning: this recipe was measured on aarch64; you are on $(uname -m)"
  command -v curl >/dev/null || die "curl is required"
  command -v git  >/dev/null || die "git is required"
  command -v patch >/dev/null || die "patch is required"
  command -v nvidia-smi >/dev/null || die "no nvidia-smi: this needs a CUDA GPU"
  mkdir -p "$HOME_DIR"
}

# --------------------------------------------------------------- conda env ---

step_env() {
  if done_with env && [ -x "$ENV_DIR/bin/python" ]; then
    note "conda env present"; return
  fi
  say "Creating the conda environment (this is the route that gets pynini on aarch64)"
  if [ ! -x "$ENV_DIR/bin/conda" ]; then
    curl -fsSL "$MINIFORGE_URL" -o "$HOME_DIR/miniforge.sh"
    bash "$HOME_DIR/miniforge.sh" -b -p "$ENV_DIR"
    rm -f "$HOME_DIR/miniforge.sh"
  fi
  # pynini publishes no aarch64 wheel and needs OpenFst 1.8; conda-forge brings both.
  "$ENV_DIR/bin/conda" install -y -c conda-forge python=3.12 pynini=2.1.6.post1
  mark env
}

# ------------------------------------------------------------ serving stack ---

step_stack() {
  if done_with stack && "$ENV_DIR/bin/python" -c "import sglang_omni" 2>/dev/null; then
    note "sglang-omni installed"; return
  fi
  say "Installing sglang-omni at the pinned revision"
  if [ ! -d "$SRC_DIR/.git" ]; then
    git clone https://github.com/fishaudio/sglang-omni "$SRC_DIR"
  fi
  git -C "$SRC_DIR" fetch --quiet origin "$SGLANG_OMNI_REV" 2>/dev/null || git -C "$SRC_DIR" fetch --quiet origin
  git -C "$SRC_DIR" checkout --quiet "$SGLANG_OMNI_REV"
  "$ENV_DIR/bin/pip" install --prerelease=allow -e "$SRC_DIR[fishaudio]"
  mark stack
}

# ----------------------------------------------------------------- weights ---

step_weights() {
  local n
  n=$(find "$WEIGHTS" -maxdepth 1 -type f 2>/dev/null | wc -l)
  if done_with weights && [ "$n" -ge 18 ]; then
    note "weights present ($n files)"; return
  fi
  say "Fetching the pinned weights (41.5 GB, 18 files)"
  "$ENV_DIR/bin/pip" install --quiet "huggingface_hub[cli]"
  "$ENV_DIR/bin/hf" download fishaudio/drama-3 --revision "$DRAMA3_REV" --local-dir "$WEIGHTS"
  mark weights
}

# ----------------------------------------------------------------- patches ---

step_patches() {
  local model; model="$(model_dir)"
  [ -d "$model" ] || die "no Drama 3 model directory at $model"
  if done_with patches && [ -f "$model/mtp_fp8.py" ]; then
    note "patches applied"; return
  fi
  say "Applying the patches (all default OFF until a flag is set)"
  "$REPO/scripts/apply-patches.sh" "$SRC_DIR"
  mark patches
}

# ------------------------------------------------------------------ config ---

step_config() {
  if [ -f "$CONFIG" ]; then
    note "config present: $CONFIG"; return
  fi
  say "Writing the measured configuration"
  cat > "$CONFIG" <<YAML
# stream_stride 4 is the knee: 10 -> 4 halves TTFB at flat RTF, 2 degrades RTF.
# The three CUDA graph flags are where the rest of the TTFB win comes from.
config_cls: Drama3PipelineConfig
model_path: $WEIGHTS
stages:
  preprocessing:
    total_reserve_bytes: 6GiB
    factory:
      max_concurrency: 1
      max_reference_audio_seconds: 40
  tts_engine:
    gpu_memory_fraction: null
    total_reserve_bytes: 60GiB
    engine:
      mem_fraction_static: null
      max_total_tokens: 4096
      max_mamba_cache_size: 1
      disable_cuda_graph: false
    factory:
      context_length: 4096
      max_prompt_rows: 2048
      max_new_tokens: 256
      max_running_requests: 1
      first_frame_cuda_graphs: true
  vocoder:
    gpu_memory_fraction: null
    total_reserve_bytes: 8GiB
    factory:
      stream_stride: 4
      codec_path: $WEIGHTS
      max_batch_size: 1
      window_cuda_graphs: true
YAML
}

# ------------------------------------------------------------------- serve ---

running_pid() {
  [ -f "$PIDFILE" ] || return 1
  local pid; pid=$(cat "$PIDFILE")
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "$pid"
}

step_serve() {
  if [ -n "${1:-}" ] && [ -n "$(health_json)" ]; then
    note "already serving on http://$HOST:$PORT"; return
  fi
  if running_pid >/dev/null; then
    note "a server process is running but not healthy yet; waiting on it"
  else
    say "Starting the server"
    local env_args=(
      "DRAMA3_LM_HEAD_FP8=1"
      "HF_HUB_OFFLINE=1"
      "TRANSFORMERS_OFFLINE=1"
      "PYTHONUNBUFFERED=1"
    )
    if [ "$TOWER" = "1" ]; then
      # Fusion requires plain bf16 nn.Linear, so it excludes MTP FP8. Slower.
      env_args+=("DRAMA3_ALLOW_UNTESTED_GPU=tower")
      note "fused tower selected: MTP FP8 stays off (mutually exclusive)"
    else
      env_args+=("DRAMA3_MTP_FP8=1" "DRAMA3_MTP_FP8_HEADS=0")
      [ "$SCOPE" = "all" ] || env_args+=("DRAMA3_MTP_FP8_GROUPS=$SCOPE")
      note "MTP FP8 scope: $SCOPE, codebook heads off"
    fi
    if [ "$BACKBONE" = "1" ]; then
      env_args+=("DRAMA3_BACKBONE_FP8=1" "DRAMA3_BACKBONE_FP8_SCOPE=$BACKBONE_SCOPE")
      note "backbone FP8 scope: $BACKBONE_SCOPE"
      note "check the conversion summary in the log before trusting a measurement:"
      note "  grep 'backbone FP8' $LOG"
      note "  a scope that selects nothing still serves, and reads as a result"
    else
      note "backbone FP8 off (DRAMA3_BACKBONE=0); expect about 28.6 tok/s decode"
    fi
    caveat
    ( cd "$SRC_DIR" && env "${env_args[@]}" nohup "$ENV_DIR/bin/sgl-omni" serve \
        --config "$CONFIG" --host "$HOST" --port "$PORT" \
        --model-name drama3 >> "$LOG" 2>&1 < /dev/null & echo $! > "$PIDFILE" )
    note "log: $LOG"
  fi
  say "Waiting for health (the first load reads 41.5 GB and takes a few minutes)"
  local i
  for i in $(seq 1 180); do
    if [ -n "$(health_json)" ]; then
      note "healthy after ~$((i * 5))s"
      note "endpoint: http://$HOST:$PORT/v1/audio/speech"
      return
    fi
    if ! running_pid >/dev/null; then
      tail -n 30 "$LOG" >&2
      die "the server exited during startup; see $LOG"
    fi
    sleep 5
  done
  die "not healthy within 15 minutes; see $LOG"
}

# ------------------------------------------------------------------ checks ---

do_check() {
  say "Check"
  local model=""
  [ -x "$ENV_DIR/bin/python" ] && model="$(model_dir 2>/dev/null || true)"
  printf '    %-22s %s\n' "conda env" "$([ -x "$ENV_DIR/bin/python" ] && echo "$ENV_DIR" || echo MISSING)"
  printf '    %-22s %s\n' "sglang-omni" "$(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo MISSING)"
  printf '    %-22s %s\n' "pinned revision" "${SGLANG_OMNI_REV:0:7}"
  printf '    %-22s %s files\n' "weights" "$(find "$WEIGHTS" -maxdepth 1 -type f 2>/dev/null | wc -l)"
  printf '    %-22s %s\n' "patches" "$([ -n "$model" ] && [ -f "$model/mtp_fp8.py" ] && echo applied || echo "not applied")"
  printf '    %-22s %s\n' "config" "$([ -f "$CONFIG" ] && echo "$CONFIG" || echo MISSING)"
  printf '    %-22s %s\n' "process" "$(running_pid || echo "not running")"
  local h; h="$(health_json)"
  printf '    %-22s %s\n' "health" "${h:-unreachable on http://$HOST:$PORT}"
  [ -n "$h" ]
}

do_stop() {
  local pid; pid="$(running_pid || true)"
  if [ -z "$pid" ]; then
    note "not running"; return
  fi
  say "Stopping $pid"
  kill "$pid" 2>/dev/null || true
  local i
  for i in $(seq 1 30); do
    running_pid >/dev/null || { note "stopped"; rm -f "$PIDFILE"; return; }
    sleep 1
  done
  note "still up after 30s, sending SIGKILL"
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
}

case "${1:-}" in
  --check) preflight; do_check ;;
  --stop)  do_stop ;;
  "")      preflight; step_env; step_stack; step_weights; step_patches; step_config; step_serve skip-if-healthy ;;
  *)       die "unknown argument: $1 (use --check or --stop)" ;;
esac
