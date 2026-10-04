# Historical CUTLASS profiler commands

Build the profiler from the pinned checkout before using these manual development references.

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
