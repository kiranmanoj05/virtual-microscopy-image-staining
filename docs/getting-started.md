# First local session

Use Python 3.11–3.13; Python 3.12 is the initial CI target. Run `python3 --version` first. If it is outside this range, install a supported Python before creating the virtual environment.

## Setup

While the foundation PR is open:

```bash
git clone --branch chore/project-foundation https://github.com/kiranmanoj05/virtual-microscopy-image-staining.git
cd virtual-microscopy-image-staining
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[ml]'
python -m unittest discover -s tests -v
staining doctor
```

The `ml` extra installs the libraries for later image/model work. For the lightweight audit tools alone, use `python -m pip install -e .`. After installing the ML extra on a compatible Mac, the doctor reports whether PyTorch can use MPS. A CPU result is diagnostic information; investigate it before starting a long run. The command does not benchmark performance or run training.

Send the doctor output and any error text back before the data milestone. Do not download an entire dataset yet. Expected test result: 11 passing tests.

## Audit command

The example CSV is a header-only template. It must be populated from verified dataset metadata and actual file hashes; it intentionally fails the audit while empty.

```bash
staining audit path/to/manifest.csv --data-root path/to/images
```

The command emits JSON. Exit codes: 0 for a valid bookkeeping audit, 1 for detected problems, 2 for file/CSV read failures. It does not validate image registration, axes or biological independence.

## After the PR is merged

Switch to main and create the next branch from the updated main:

```bash
git switch main
git pull --ff-only
git switch -c feat/data-intake
```

Check `git status` before switching if you have made local changes. Keep the virtual environment, raw images, checkpoints and run outputs out of Git.

## What comes next

Inspect one candidate dataset's terms, download size, channel identities and grouping metadata. Obtain a small set of paired fields. Build the image intake adapter, visual pair audit and frozen group split before writing the first training experiment.
