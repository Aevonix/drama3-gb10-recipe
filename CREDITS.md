# Credits

- **Fish Audio** - Drama 3, and the SGLang-Omni serving stack these patches apply
  to, including the fused MTP tower and the Triton codebook sampling kernels that
  patch 0001 merely un-gates, and the FP8 machinery both quantization patches
  stand on. Neither patch implements a kernel: both route Drama 3's layers into
  kernels that already existed in the stack. Built with Fish Audio.

Patches, measurements and this write-up: Kurcide (Aevonix), contact@aevonix.com.

The per-token byte model is not original and is not a contribution; it is the
standard bandwidth-bound decode argument, applied with the actual safetensors
header sizes instead of guesses. Its only merit here is that the arithmetic was
written down before each measurement, which is what makes the forecast table in
the README worth reading at all.
