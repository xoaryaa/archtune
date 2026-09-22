# Archtune

Archtune is my attempt to understand why code that does the exact same math can run very differently depending on how it is written and how the hardware sees it.

I’m starting with matrix multiplication and using it to experiment with loop ordering, cache locality, tiling, compiler behavior, and eventually vectorization/threading.

Current setup: **C++ on Apple M4**.

---

## what I've done so far

I started with a basic 512×512 matrix multiplication with:

- different loop orders
- cache tiling
- static vs runtime tile sizes
- 30 runs after 5 warmup runs for every configuration


The main thing I've learned so far is that performance is a lot less predictable than just saying "better cache locality = faster."

---

## static tile size

For this version, the tile size is hard-coded in the C++ file and the program is recompiled for each value.

| Tile Size | Mean (ms) | Median (ms) | Std Dev |
|---:|---:|---:|---:|
| 8   | 5.062 | 5.086 | 0.121 |
| 16  | 8.698 | 8.758 | 0.333 |
| 32  | 6.728 | 7.150 | 1.173 |
| 64  | 10.467 | 10.480 | 0.477 |
| 128 | 11.211 | 11.186 | 0.707 |

**Tile size 8 was the fastest here.**

The latency didn't follow a clean trend as the tile size increased. It was expected that the result to mostly come down to cache behavior.

---

## with dynamic tile size

Instead of changing the source and recompiling every time, I changed the program so I could run:

```bash
./matmul 256 
```
Now the tile size is only known at runtime.

The results changed quite a bit — in this version, tile size 256 gave the lowest latency.

So the same tiling idea produced a different "best" tile size depending on whether the value was known at compile time or runtime.

With a hard-coded tile size, the compiler already knows the loop bounds while generating the executable. That can give it more room for things like constant propagation, loop optimization, and vectorization.

With a runtime value, it has to generate code that works for whatever tile size the user gives it.

So this isn't really "compile time getting added to runtime." It's more that the generated machine code itself can be different.

That was probably the most useful thing from these experiments.