# visiontext

<p align="center">
<a href="https://github.com/simon-ging/visiontext/actions/workflows/build-py39-cpu.yml">
  <img alt="minimal build 3.9 status" title="build 3.9 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/visiontext/build-py39-cpu.yml?branch=main&label=minimal%20build%203.9%20cpu" />
</a>
<a href="https://github.com/simon-ging/visiontext/actions/workflows/build-py310-cpu.yml">
  <img alt="minimal build 3.10 status" title="build 3.10 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/visiontext/build-py310-cpu.yml?branch=main&label=minimal%20build%203.10%20cpu" />
</a>
<a href="https://github.com/simon-ging/visiontext/actions/workflows/build-py312-cpu.yml">
  <img alt="minimal build 3.12 status" title="build 3.12 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/visiontext/build-py312-cpu.yml?branch=main&label=minimal%20build%203.12%20cpu" />
</a>
<br />
<a href="https://github.com/simon-ging/visiontext/actions/workflows/build-py39-cpu.yml">
  <img alt="full build 3.9 status" title="build 3.9 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/visiontext/build-py39-cpu-full.yml?branch=main&label=full%20build%203.9%20cpu" />
</a>
<a href="https://github.com/simon-ging/visiontext/actions/workflows/build-py310-cpu.yml">
  <img alt="full build 3.10 status" title="build 3.10 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/visiontext/build-py310-cpu-full.yml?branch=main&label=full%20build%203.10%20cpu" />
</a>
<a href="https://github.com/simon-ging/visiontext/actions/workflows/build-py312-cpu.yml">
  <img alt="full build 3.12 status" title="build 3.12 status" src="https://img.shields.io/github/actions/workflow/status/simon-ging/visiontext/build-py312-cpu-full.yml?branch=main&label=full%20build%203.12%20cpu" />
</a>
<br />
<img alt="coverage" title="coverage" src="https://raw.githubusercontent.com/simon-ging/visiontext/main/docs/coverage.svg" />
<a href="https://pypi.org/project/visiontext/">
  <img alt="version" title="version" src="https://img.shields.io/pypi/v/visiontext?color=success" />
</a>
</p>

Utilities for deep learning on multimodal data.

* jupyter notebooks / jupyter lab / ipython
* matplotlib
* pandas
* webdataset / tar
* pytorch

## Install

Requires `python>=3.10`.

```bash
pip install visiontext
```

The base install is small. Each area of the package has an extra with what it needs:

| extra | for |
| --- | --- |
| `torch` | `distutils`, `mathutils`, `torchutils`, `denormalize`, `iotools.feature_compression` |
| `images` | `images`, includes `torch`. Additionally requires `libjpeg-turbo` |
| `plot` | `colormaps`, `bboxes`, `plots`, `visualize_ratios` |
| `notebook` | `imports`, `htmltools`, `pandatools`, includes `plot` |
| `webdataset` | `webdataset_pipeline`, includes `torch` |
| `nlp` | `nlp` |
| `profiling` | `profiling`, includes `torch` |
| `download` | `image_downloader`, `font` |
| `audio` | `audiotools` |
| `cache` | `cacheutils` |
| `sql` | `sqlalchemist`. Additionally requires `sqlite` |
| `config` | `configutils` |

```bash
pip install visiontext[torch,notebook]
```

### Full build

Everything at once:

```bash
pip install visiontext[full]
```

## Dev install

Clone repository and cd into, then:

```bash
pip install pytest pytest-cov pylint black[jupyter]
pylint visiontext
pylint tests

# full build
pip install -e .[full]
python -m pytest --cov

# minimal build
pip install -e .
python -m pytest --cov -m "not full"

```

## Changelog

- 0.23.1: Drop python 3.8 support since PyTorch dropped it. Set minimum python version as 3.10.
  - PyTorch disabled the install via conda, change builds to install PyTorch CPU via pip.
- 0.10.1: Test with python 3.12
- 0.8.1: Set minimum python version to 3.8 since PyTorch requires it
