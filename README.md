[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/r05YygS1)
# Homework: SGEMM optimization arc (K2–K5)

Implement four progressively faster matrix-multiply kernels in
`numba.cuda`.

You implement K2, K3, K4, and K5. K1 (naive) is given to you as a
worked example so you can see the `@cuda.jit` signature shape every
kernel must match.

## How submission works

This assignment is a GitHub Classroom repo. **You only edit
`kernels.py`** the launch wrappers, tile-size constants, and harness
code are already in place. Push your edits to the `main` branch of this
repo. Each push that touches `kernels.py` triggers the autograder,
which runs on a Modal A100 40GB and posts your grade as a comment on
the commit.

You have **5 graded submissions per assignment**. Use them don't push
half-baked attempts. Run the local sanity check first.

## Grading rubric

The autograder runs each of your kernels on a battery of shapes and
checks two things per kernel:

1. **Correctness**: output matches a reference within fp32 tolerance.
2. **Performance floor**: your kernel runs at least as fast as a
   per-kernel threshold calibrated on the autograder hardware.

Both checks must pass for a kernel to count. Grading is **cumulative**:

| Kernels passed         | Grade |
| ---------------------- | ----- |
| K2                     | C     |
| K2 + K3                | B−    |
| K2 + K3 + K4           | B     |
| K2 + K3 + K4 + K5      | B+    |

K3 only counts if K2 also passes, K4 only if K2+K3, etc. So implement
in order!

## Where to run the sanity check

`sanity_check.py` runs your kernels at 256² and 512² and compares
against a numpy reference. It is **correctness only** — passing does
not guarantee a passing grade, since the autograder also enforces a
performance floor. K2–K5 will show up as `FAIL` until you implement
them; K1 (worked example) should `PASS` immediately.

You have three options for where to run it:

### Option A: Locally (best dev experience)

Requires a system-installed CUDA 12.x toolkit and a CUDA-capable GPU
(sm_70+). Set up a Python 3.12 virtual environment and install deps:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sanity_check.py
```

### Option B: Google Colab (no local GPU needed)

Colab provides a free T4 GPU.

1. Visit <https://colab.research.google.com> and create a new notebook.
2. Runtime -> Change runtime type -> Hardware accelerator: **T4 GPU**.
3. Run these three cells:

   ```python
   # Cell 1 — upload kernels.py and sanity_check.py from your local clone.
   from google.colab import files
   files.upload()  # browse and select both files
   ```

   ```bash
   # Cell 2 install dependencies. Colab ships CUDA already.
   !pip install -q numpy==2.4.4 numba==0.65.0 numba-cuda==0.30.0
   ```

   ```bash
   # Cell 3 run the sanity check.
   !python sanity_check.py
   ```

After editing your kernels, either re-run cell 1 to upload again or use
a `%%writefile kernels.py` magic cell to edit in-notebook. Colab's T4
has a different perf profile than the autograder's A100 correctness
results will track, but timings will not.

### Option C — Modal A100 (matches the autograder)

Modal gives every account $30/month of free credit, which is enough
for thousands of sanity checks on the same A100 40GB the autograder
uses. One run costs about $0.01.

```
pip install modal
modal token new            # one-time browser auth, links your account
modal run modal_sanity.py
```

The script (`modal_sanity.py`, included in this repo) ships your local
`kernels.py` and `sanity_check.py` to a Modal container, runs them on
A100, and streams the output back. Use this when you want
representative timings before spending a graded submission.

## Files in this repo

- `kernels.py`            — your edit surface; numba kernels for K1–K5
- `sanity_check.py`       — local correctness + rough timing
- `modal_sanity.py`       — `modal run` wrapper that runs sanity_check on A100 (Option C above)
- `kernel6_stretch.cu`    — optional stretch goal (cupy.RawKernel; see below)
- `requirements.txt`      — Python deps
- `README.md`             — this file
- `.github/workflows/sgemm-grader.yml` — the autograder Action (don't edit)

## Stretch (K6)

Once K2–K5 pass, you can optionally implement K6 (vectorized `float4`
loads) in `kernel6_stretch.cu`. K6 cannot be expressed in `numba.cuda`
You have to write CUDA C++ launched via `cupy.RawKernel`. To run it
locally you'll need `cupy-cuda12x` (also in `requirements.txt`). The
reward for passing K6 on the autograder is not yet decided; for now,
treat it as a learning bonus.

## References

- siboehm's blog: <https://siboehm.com/articles/22/CUDA-MMM>
- Numba CUDA docs: <https://numba.readthedocs.io/en/stable/cuda/>
- Course lectures 10 and 11 cover the optimization arc this assignment
  walks through.
