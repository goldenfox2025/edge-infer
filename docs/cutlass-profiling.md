# Historical CUTLASS profiler commands

These commands are historical third-party GEMM experiments. The maintained
source contains no CUTLASS checkout or profiler build target. To reproduce an
old experiment, recover its pinned source from Git history or provide an
external CUTLASS profiler. These commands do not validate the current decoder;
use `scripts/profile.sh` for maintained native generation profiling.

```sh
./cutlass_profiler \
  --operation=gemm \
  --m=20 --n=2048 --k=1536 \
  --A=bf16:column --B=bf16:column --C=bf16:column --D=bf16:column \
  --alpha=1 --beta=1 \
  --op_class=tensorop --accumulator-type=f32 \
  --verbose=1


./cutlass_profiler \
  --operation=gemm \
  --m=20 --n=2048 --k=1536 \
  --A=bf16:column --B=bf16:column --C=bf16:column --D=bf16:column \
  --alpha=1 --beta=1 \
  --op_class=tensorop --accumulator-type=f32 \
  --split-k-mode=parallel --split-k-slices=2 \
  --enable-best-kernel-for-fixed-shape \
  --verbose=1
```
