# Neural Graph Mapping

This repository contains the official implementation for the paper [*Neural Graph Mapping for Dense SLAM with Efficient Loop Closure*](https://kth-rpl.github.io/neural_graph_mapping/) by Leonard Bruns, Jun Zhang, and Patric Jensfelt.

## Installation

### pixi
To quickly get the code running and reproduce the results you can use [pixi](https://pixi.sh/latest/). First, [install pixi](https://pixi.sh/latest/#installation). Then run the following command to install the package, download the data, and run an example scene (the datasets will by default be stored in `./datasets/`; you can modify the dataset dir in the `.pixi.sh` file):
```bash
pixi run nrgbd_br --rerun_vis True
```

To run all the scenes and datasets you can run:
```bash
pixi run all
```
You can optionally add arguments to all commands by settings `NGM_EXTRA_ARGS` prior to running the command. For example, to run all datasets with visualization enabled run:
```bash
export NGM_EXTRA_ARGS="--rerun_vis True" pixi run all
```

### Manual
First you need to install `torch==2.2.*` and the corresponding CUDA version (such that `nvcc` is available and matches the torch's CUDA version). This is necessary because some dependencies are installed from source.

To install this package and all dependencies, clone this repo, and run
```bash
pip install --no-build-isolation -e .
```

### Development
- Use `pip install --no-build-isolation -e .` to install the package in editable mode
- Use `pip install -r requirements-dev.txt` to install dev tools

<!-- ## Reference -->

<!-- If you find this useful in your research, consider citing our publication: -->

<!-- ``` -->
<!-- placeholder -->
<!-- ``` -->
