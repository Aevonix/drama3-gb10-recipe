# Changelog

## 0.2.0 - 2026-10-14

Backbone FP8, the largest single step so far. Same hardware and same pinned
revisions as 0.1.0.

- `patches/0004-backbone-online-fp8.patch` - online FP8 for the backbone linear
  layers, selectable by scope over attention, shared experts and linear
  attention. Decode 28.58 -> 38.85 tok/s (+36%), short TTFB 0.420 -> 0.363 s,
  long RTF 0.787 -> 0.582. Default OFF; `run.sh` turns it on with
  `DRAMA3_BACKBONE=1`, which is now the default there, and `DRAMA3_BACKBONE=0`
  returns to the previous recommendation.
- Apply 0002 before 0004: both touch `quantization.py`. The numbering is
  historical, not an ordering.
- Steady-state memory improves slightly, 52.50 -> 51.48 GiB, because the
  converted weights are smaller. The load-time peak gets worse, 54.13 -> 59.91
  GiB, because the conversion runs after the bf16 weights are resident. On a
  dedicated box the peak is free; on a shared one the transient is what bites.
- Measured in a later series than the 0.1.0 table, which re-measured its own
  baseline at 28.58 tok/s against the 28.2 recorded there, inside the 2% noise
  floor. The two tables are kept separate; do not compute a percentage across
  them.
- The per-token byte budget predicted 37.3 tok/s and the measurement was 38.85,
  so for linear attention the budget is a lower bound rather than an equality:
  converting those blocks removes arithmetic as well as bytes.
- Three things are excluded from every scope on purpose: the MoE router and the
  shared-expert gate, the linear-attention `in_proj_ba`, and the
  linear-attention `conv1d`. Reasons in the patch header.
- Scopes name runtime modules, not checkpoint tensors. A scope written from
  checkpoint names selects nothing, serves perfectly and reads as a result, so
  the conversion summary is logged and the README says to check it.

## 0.1.0 - 2026-10-14

First release. Measured on one GB10 (120 GB unified, aarch64, sm_121), single
GPU, single stream, against sglang-omni
`2cfb3b67becb437b70751defe37e63d3af4eedb0` (sglang_omni 0.1.8, sglang 0.5.21)
and fishaudio/drama-3 `dd64a85cd82e9521f7202610ede97b367fc83946`.

- `run.sh`: one command from a clean box to a healthy endpoint, idempotent,
  with `--check` and `--stop`.
- Install route for aarch64 through conda-forge, which is what gets pynini
  2.1.6.post1 and OpenFst in place.
- Configuration: `stream_stride: 4` plus the three CUDA graph flags. Short TTFB
  0.806 -> 0.467 s with no code change.
- `patches/0001-sm121-kernel-gate-opt-in.patch` - per-feature opt-in past the
  hardcoded `VALIDATED_COMPUTE_CAPABILITY = (9, 0)` gate. Default OFF.
  Unvalidated by the vendor on this architecture.
- `patches/0002-output-head-online-fp8.patch` - online FP8 output head,
  +7.57% decode. Default OFF.
- `patches/0003-audio-decoder-mtp-online-fp8.patch` plus `mtp_fp8.py` - online
  FP8 for the audio-decoder projections, with selectable scope. Residual-only
  28.55 tok/s, full 29.74 tok/s, from a 23.90 baseline. Default OFF.
- Recommended against: Triton codebook sampling (+2.14%, one noise width, only
  change touching the sampler) and the codebook heads (+1.1%, below noise).
- The 2% noise floor, the forecast record, and the things that did not work.
