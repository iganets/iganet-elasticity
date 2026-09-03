# IgANet-Elasticity

This repository contains elasticity example applications built on top of the
[IgANet core](https://github.com/iganets/iganet).

The core repository provides the spline, collocation, multipatch, constraint,
and network infrastructure. This repository adds concrete elasticity examples,
example configurations, reference runs, result export, and visualization
scripts.

This README is written for newcomers. It explains:

- where the important files are
- how to install the required dependencies
- how to configure and build the examples
- how to run the executables
- how to visualize the generated results

## Repository Layout

```text
iganet-elasticity/
├── CMakeLists.txt
├── README.md
├── filedata/
│   ├── bone_simplified.xml
│   ├── five_patch_cantilever.xml
│   ├── two_cube_parametric.xml
│   └── two_patch_smoke.xml
├── results/
├── src/
│   ├── headers/
│   │   ├── lin_elasticity_multipatch_net.hpp
│   │   ├── lin_elasticity_net.hpp
│   │   └── lin_elasticity_utils.hpp
│   ├── utils/
│   │   ├── config.hpp
│   │   └── paths.hpp
│   ├── examples2D/
│   │   ├── singlePatch/
│   │   └── multiPatch/
│   └── examples3D/
│       ├── singlePatch/
│       └── multiPatch/
└── std_collocation_python/
```

## Main Files

### Example sources

- `src/examples2D/singlePatch/iganet_lin_elasticity_2D.cxx`
- `src/examples2D/singlePatch/iganet_lin_elasticity_2D_optimized.cxx`
- `src/examples2D/multiPatch/iganet_lin_elasticity_2D_multipatch_parametric.cxx`
- `src/examples3D/singlePatch/iganet_lin_elasticity_3D.cxx`
- `src/examples3D/singlePatch/iganet_lin_elasticity_3D_optimized.cxx`
- `src/examples3D/multiPatch/iganet_lin_elasticity_3D_multipatch_bone.cxx`
- `src/examples3D/multiPatch/iganet_lin_elasticity_3D_multipatch_parametrized.cxx`

### Example configs

- `src/examples2D/singlePatch/sim_config_2D_single_patch.json`
- `src/examples2D/multiPatch/sim_config_2D_multi_patch.json`
- `src/examples3D/singlePatch/sim_config_3D_single_patch.json`
- `src/examples3D/multiPatch/sim_config_3D_multi_patch_bone.json`
- `src/examples3D/multiPatch/sim_config_3D_multi_patch_parametrized.json`

### Visualization scripts

- `src/examples2D/singlePatch/show_2D_single_patch.py`
- `src/examples2D/multiPatch/show_2D_multi_patch.py`
- `src/examples3D/singlePatch/show_3D_single_patch.py`
- `src/examples3D/multiPatch/show_3D_multi_patch_bone.py`

### Shared elasticity headers

- `src/headers/lin_elasticity_net.hpp`
- `src/headers/lin_elasticity_multipatch_net.hpp`
- `src/headers/lin_elasticity_utils.hpp`

### Reference solver

- `std_collocation_python/run_std_coll.py`

Some examples call the Python reference solver automatically and append its data
to the same JSON result file.

## Relationship To The Core Repository

This repository depends on `iganet`.

The core is fetched with CMake `FetchContent` from GitHub. By default, CMake
uses the repository and branch configured in `CMakeLists.txt`.

You can override the branch, tag, or commit at configure time:

```bash
cmake -S . -B build \
  -DTorch_DIR=$HOME/libtorch/share/cmake/Torch \
  -DIGANET_CORE_GIT_TAG=your_branch_or_commit
```

A local `../iganet` checkout is not used automatically.

## Prerequisites

You need:

- a C++20 compiler
- CMake `>= 3.24`
- Python
- LibTorch

For visualization, install these Python packages:

```bash
pip install numpy splinepy gustaf vedo
```

## Installing LibTorch

Follow the same procedure as in the IgANet core README. The essential point is:
download a LibTorch build that matches your compiler ABI and then point CMake to
its `Torch_DIR`.

Example for Linux CPU builds, analogous to the core README:

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip -O libtorch.zip
unzip libtorch.zip -d $HOME/
```

Then configure with:

```bash
cmake -S . -B build -DTorch_DIR=$HOME/libtorch/share/cmake/Torch
```

If you use a different LibTorch install location, adjust `Torch_DIR`
accordingly.

## CPU Build (Default)

From the repository root:

```bash
cmake -S . -B build -DTorch_DIR=$HOME/libtorch/share/cmake/Torch
cmake --build build -j 16
```

This builds all example executables directly into `build/`.

If you only want one target:

```bash
cmake --build build -j 16 --target iganet_lin_elasticity_2D
cmake --build build -j 16 --target iganet_lin_elasticity_2D_optimized
cmake --build build -j 16 --target iganet_lin_elasticity_2D_multipatch_parametric
cmake --build build -j 16 --target iganet_lin_elasticity_3D
cmake --build build -j 16 --target iganet_lin_elasticity_3D_optimized
cmake --build build -j 16 --target iganet_lin_elasticity_3D_multipatch_bone
cmake --build build -j 16 --target iganet_lin_elasticity_3D_multipatch_parametrized
```

## CUDA Build (Optional)

If you want CUDA, install a CUDA-enabled LibTorch build that matches your CUDA
toolchain, then configure CMake with the matching CUDA settings.

Example:

```bash
cmake -S . -B build \
  -DTorch_DIR=/path/to/libtorch/share/cmake/Torch \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda
cmake --build build -j 16
```

The exact paths and architecture depend on your machine. CPU build is the
default and should be your starting point.

## Which Config File Belongs To Which Example?

Each executable reads its own config file from the matching example folder.

- `iganet_lin_elasticity_2D`
  - `src/examples2D/singlePatch/sim_config_2D_single_patch.json`
- `iganet_lin_elasticity_2D_optimized`
  - `src/examples2D/singlePatch/sim_config_2D_single_patch.json`
- `iganet_lin_elasticity_2D_multipatch_parametric`
  - `src/examples2D/multiPatch/sim_config_2D_multi_patch.json`
- `iganet_lin_elasticity_3D`
  - `src/examples3D/singlePatch/sim_config_3D_single_patch.json`
- `iganet_lin_elasticity_3D_optimized`
  - `src/examples3D/singlePatch/sim_config_3D_single_patch.json`
- `iganet_lin_elasticity_3D_multipatch_bone`
  - `src/examples3D/multiPatch/sim_config_3D_multi_patch_bone.json`
- `iganet_lin_elasticity_3D_multipatch_parametrized`
  - `src/examples3D/multiPatch/sim_config_3D_multi_patch_parametrized.json`

In practice:

- edit the matching JSON config
- rebuild only if you changed C++ code
- if you changed only the config, rerun the executable

## Running The Examples

Run the executables from the repository root:

### 2D single-patch

```bash
./build/iganet_lin_elasticity_2D
./build/iganet_lin_elasticity_2D_optimized
```

### 2D multi-patch

```bash
./build/iganet_lin_elasticity_2D_multipatch_parametric
```

### 3D single-patch

```bash
./build/iganet_lin_elasticity_3D
./build/iganet_lin_elasticity_3D_optimized
```

### 3D multi-patch

```bash
./build/iganet_lin_elasticity_3D_multipatch_bone
./build/iganet_lin_elasticity_3D_multipatch_parametrized
```

## Where Results Are Written

The executables write JSON files into `results/`.

Typical files are:

- `results/result_iganet_lin_elasticity_2D.json`
- `results/result_iganet_lin_elasticity_2D_optimized.json`
- `results/result_iganet_lin_elasticity_2D_multipatch_parametric.json`
- `results/result_iganet_lin_elasticity_3D.json`
- `results/result_iganet_lin_elasticity_3D_optimized.json`
- `results/result_iganet_lin_elasticity_3D_multipatch_bone.json`
- `results/result_iganet_lin_elasticity_3D_multipatch_parametrized.json`

The exact JSON contents depend on the example. Depending on the case, a result
file may contain:

- geometry and deformed control points
- displacements
- stresses or derived quantities
- collocation point data
- interface diagnostics
- optional reference data from the Python standard-collocation run

Do not assume that every result file contains every field.

## Visualizing Results

Each show script already has a matching default result file path. In the normal
case you can just run the script directly without passing an argument.

### 2D single-patch

```bash
python src/examples2D/singlePatch/show_2D_single_patch.py
```

### 2D multi-patch

```bash
python src/examples2D/multiPatch/show_2D_multi_patch.py
```

### 3D single-patch

```bash
python src/examples3D/singlePatch/show_3D_single_patch.py
```

### 3D multi-patch

```bash
python src/examples3D/multiPatch/show_3D_multi_patch_bone.py
```

If you explicitly want to visualize a different JSON file, you can still pass
the result path manually, for example:

```bash
python src/examples2D/singlePatch/show_2D_single_patch.py \
  results/result_iganet_lin_elasticity_2D_optimized.json
```

The show scripts also write PNG screenshots into `results/`.

## Suggested First Steps

If you are new to the repository, start in this order:

1. `iganet_lin_elasticity_2D_optimized`
2. `iganet_lin_elasticity_2D`
3. `iganet_lin_elasticity_2D_multipatch_parametric`
4. `iganet_lin_elasticity_3D_optimized`
5. `iganet_lin_elasticity_3D`
6. `iganet_lin_elasticity_3D_multipatch_parametrized`
7. `iganet_lin_elasticity_3D_multipatch_bone`

That keeps the early workflow simple and makes debugging easier.

## Typical Newcomer Workflow

### 1. Build one small example

```bash
cmake --build build -j 16 --target iganet_lin_elasticity_2D_optimized
./build/iganet_lin_elasticity_2D_optimized
python src/examples2D/singlePatch/show_2D_single_patch.py \
  results/result_iganet_lin_elasticity_2D_optimized.json
```

### 2. Inspect the config

Open:

- `src/examples2D/singlePatch/sim_config_2D_single_patch.json`

Typical things to change:

- material parameters
- spline settings
- optimizer settings
- boundary conditions

### 3. Move to multipatch

```bash
cmake --build build -j 16 --target iganet_lin_elasticity_2D_multipatch_parametric
./build/iganet_lin_elasticity_2D_multipatch_parametric
python src/examples2D/multiPatch/show_2D_multi_patch.py
```

### 4. Move to 3D

```bash
cmake --build build -j 16 --target iganet_lin_elasticity_3D_optimized
./build/iganet_lin_elasticity_3D_optimized
python src/examples3D/singlePatch/show_3D_single_patch.py \
  results/result_iganet_lin_elasticity_3D_optimized.json
```

## Notes About XML Geometry

Some examples support:

- `geometry.mode = "parametric"`
- `geometry.mode = "xml"`

In XML mode, geometries are typically loaded from `filedata/`, for example:

- `filedata/two_cube_parametric.xml`
- `filedata/five_patch_cantilever.xml`
- `filedata/bone_simplified.xml`

## Common Places To Edit

If you want to change...

### example-specific settings

Edit the matching `sim_config_*.json`.

### one concrete example

Edit the corresponding `.cxx` file in `src/examples2D/...` or
`src/examples3D/...`.

### shared single-patch elasticity logic

Edit:

- `src/headers/lin_elasticity_net.hpp`

### shared multi-patch elasticity logic

Edit:

- `src/headers/lin_elasticity_multipatch_net.hpp`

### shared config parsing

Edit:

- `src/utils/config.hpp`

### visualization logic

Edit the matching `show_*.py`.

## Troubleshooting

### CMake cannot fetch the core repository

Check that the machine has network access and that
`IGANET_CORE_GIT_REPOSITORY` and `IGANET_CORE_GIT_TAG` point to an existing
repository and branch, tag, or commit.

### CMake cannot find LibTorch

Check that `Torch_DIR` points to:

```text
.../share/cmake/Torch
```

inside your LibTorch installation.

### A target is not known

Re-run CMake configure:

```bash
cmake -S . -B build -DTorch_DIR=$HOME/libtorch/share/cmake/Torch
```

Then build again.

### A show script cannot open the result file

Check that:

- the matching executable has been run
- the expected JSON file exists in `results/`
- you are using the correct show script for that example

### Visualization fails because of missing Python packages

Install:

```bash
pip install numpy splinepy gustaf vedo
```

## Final Remark

The shortest path into this repository is still:

- build `iganet_lin_elasticity_2D_optimized`
- inspect `src/examples2D/singlePatch/sim_config_2D_single_patch.json`
- run `src/examples2D/singlePatch/show_2D_single_patch.py`

That gives you the cleanest first overview of the full workflow.
