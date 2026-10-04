# Small integer to FP16 conversion

A normalized FP16 value uses one sign bit, five exponent bits with bias 15,
and ten fraction bits:

```text
value = (-1)^sign * 2^(exponent - 15) * (1 + fraction / 1024)
```

1024 has the FP16 representation `0x6400`. For an integer `Y` with
`0 <= Y < 1024`, the fraction bits of `1024 + Y` are exactly those of `Y`:

```text
bits(1024 + Y) = 0x6400 | Y
```

For signed INT4 weights biased by 8, take `Y = weight + 8`. A weight of 5 gives
`Y = 13`; `0x6400 | 13 = 0x640D` represents 1037. Subtracting the FP16 value
1032 (`0x6408`) recovers 5.0 (`0x4500`). Use floating-point subtraction, not
subtraction of the bit patterns. The old note's `0x4240` for 5.0 was incorrect.

The identity is specific to restricted integer ranges and exact FP16 layouts.
It does not generalize unchanged to BF16 or FP32. Any kernel benefit depends
on generated instructions, packing and surrounding work; this identity is not
a performance measurement.
