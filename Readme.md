<!--
SPDX-FileCopyrightText: 2025 kalypsso-core authors

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

![C/C++ build](https://github.com/pkestene/kalypsso-core-prive/actions/workflows/ci.yml/badge.svg?branch=main)

# What is kalypsso-core ?

kalypsso-core is just another adaptive mesh refinement package mainly (but not only) for CFD applications but designed with performance portability in mind. No CFD application can be found here, but you will find some in the companion code named [kalypsso-app-public](https://github.com/kalypsso-dev/kalypsso-app-public). kalypsso-core is designed as a library and it only provides core concepts for developing distributed octree-based adaptive mesh refinement applications. As such, distributed mesh management is provided by AMR library [p4est](https://github.com/cburstedde/p4est) and all data containers and device aware algorithms are implemented using the C++ [kokkos](https://github.com/kokkos/kokkos/) library for performance portability. We do not write from scratch but try to couple some of the best state-of-the-art tools.

- AMR is delegated to library [p4est](https://github.com/cburstedde/p4est). p4est is written in C, and has about 40 kSLOC; it implements cell-based AMR; manage of a forest of octrees, i.e. the physical domain is made of a coarse mesh (p4est connectivity), and each cell of this coarse mesh serves as a root to an octree.
- numerical schemes are designed on top of the p4est mesh, in a decoupled manner, using [kokkos](https://github.com/kokkos/kokkos) for shared memory parallelism.

You will find more technical details about kalypsso in the following article:

- Kestener, Pierre, Kalypsso: A Performance Portable Platform for Compressible Hydrodynamics Simulations using Adaptive Mesh Refinement. https://doi.org/10.1016/j.cpc.2026.110275

kalypso can be used on a laptop as well as largest parallel clusters of GPUs.

# How to build  (short version) ?

## Get the source code

Make sure to clone this repository recursively, this will also download required third party sources as git submodules.

```bash
git clone --recurse-submodules git@github.com:kalypsso-dev/kalypsso-core.git
```

Kokkos and p4est are (optionally) built as part of kalypsso with the cmake build system.

## Prerequisites

External dependencies are:

- [kokkos](https://github.com/kokkos/kokkos) 4.7.0
- [p4est](https://github.com/cburstedde/p4est) 2.8.7
- [spdlog](https://github.com/gabime/spdlog)
- [HighFive](https://highfive-devs.github.io/highfive/) and also HDF5 (preferably a parallel version of HDF5)
- [better-enums](https://aantron.github.io/better-enums/)
- optional [cpptrace](https://github.com/jeremy-rifkin/cpptrace)
- optional [cnpy](https://github.com/pkestene/cnpy-cmake) for numpy array outputs

You'll also need [cmake](https://cmake.org/) (minimum version 3.18) and optionally an [MPI](https://www.mpi-forum.org/) implementation (only )

These dependencies can either be :
- built along kalypsso (most convenient and recommended for a beginner)
- built in a separate cmake sub-project (located in sub-directory `dependencies`); this third option is a bit cleaner, it additionally provides a modulefiles to ease the use of these dependencies
- detected from your system's environment

## Build kalypsso-core

### build p4est, Kokkos and kalypsso all together

You can chose to build p4est, Kokkos and kalypsso all together at once.

Please note that
- p4est will be built using cmake  macro `ExternalProject`
- unfortunately kokkos cannot be built as easily using `ExternalProject` (mostly because kokkos heavily modify the compilation environment); one must use either `add_subdirectory` or `FetchContent` to a tight integration of kokkos build into kalypsso. Alternatively, if kokkos is already installed on your system for the target hardware, kalypsso will detect it (provided `CMAKE_INSTALL_PREFIX` is adequatly set).

Here is an example command line to build kalypsso with p4est and Kokkos for Kokkos/OpenMP backend target (which is the default):

```bash
cd kalypsso-core
cmake -B _build/openmp -S . \
    -DKALYPSSO_CORE_KOKKOS_BUILD:BOOL=ON \
    -DKALYPSSO_CORE_KOKKOS_BACKEND=OpenMP \
    -DKALYPSSO_CORE_BUILD_P4EST:BOOL=ON
cmake --build _build/openmp -j 8
```

The same for Kokkos/CUDA (target CUDA architecture will be detected during the build):
```bash
cmake -B _build/cuda -S . \
    -DKALYPSSO_CORE_KOKKOS_BUILD:BOOL=ON \
    -DKALYPSSO_CORE_KOKKOS_BACKEND=Cuda \
    -DKALYPSSO_CORE_BUILD_P4EST:BOOL=ON
cmake --build _build/cuda -j 8
```

Important note regarding Kokkos/Cuda backend:
- if you build on the same platform as the one used to run kalypsso-core, you're all set, kokkos build system will auto-detect GPU architecture;
- if you build on a different system, you need to specify the target architecture, e.g. `-DKokkos_ARCH_HOPPER90=ON` (for Nvidia Hopper aka `sm_90` architecture). Run `ccmake --build _build/cuda` to navigate all available Kokkos architecture cmake options;
- using a Cuda-aware MPI implementation is absolutely required only if you plan to use more than one GPU. So by default, MPI implementation is expected to by cuda-aware. Conversely, if you only have access to machine that has a single GPU, you can safely deactivate the use of MPI; to do that just use cmake variable `KALYPSSO_CORE_USE_MPI=OFF`.


Please note that you don't have to specify environment variable CXX (set to `nvcc_wrapper` when targeting CUDA backend), each sub-project (p4est / Kokkos / kalypsso) is built with a custom specific `CMAKE_CXX_COMPILER` variable; if `KALYPSSO_CORE_KOKKOS_BACKEND` is `Cuda`, internally `nvcc_wrapper` will be selected to build both Kokkos and kalypsso-core.

## What should I do if I want to use an already installed version of Kokkos ?

Just set environment variable `CMAKE_PREFIX_PATH` to ensure it contains path to file `KokkosConfig.cmake`:
```shell
export CMAKE_PREFIX_PATH={KOKKOS_INSTALL_DIR}/lib/cmake/Kokkos:$CMAKE_PREFIX_PATH
```
then Kokkos will be recognized, and building Kokkos will be skipped.

In that case, you don't need to specify `KALYPSSO_CORE_KOKKOS_BACKEND`, kokkos will be detected by `find_package(Kokkos)`

## What should I do if I want to use an already installed version of p4est ?

Use environment variable `CMAKE_PREFIX_PATH`.

```bash
```shell
# P4EST_ROOT is assumed to be the root directory where p4est was installed
export CMAKE_PREFIX_PATH={P4EST_ROOT}/cmake:$CMAKE_PREFIX_PATH
```

# More information

## build and run unit tests

```shell
# configure cmake for building unit tests
# make sure to have boost installed (with libs)
ccmake -DKALYPSSO_CORE_ENABLE_UNIT_TESTING=ON ..
make

# run unit tests
make test
```

## Build documentation

### Requirements

- [doxygen](https://www.doxygen.nl/)
- (optional, but recommended) [mkdocs](https://www.mkdocs.org/) for building a static webpage with documentation, written in markdown, with [MkDoxy plugin](https://github.com/JakubAndrysek/MkDoxy)
   ```shell
   # we recommend using miniconda for installing python packages
   conda create -n MkDoxy
   conda activate MkDoxy
   conda install pip
   cd doc
   pip install -r requirements.txt
   ```

### [doxygen](https://www.doxygen.nl/)

```shell
# re-run cmake with additional options
cmake -B _build/doc -S . -DKALYPSSO_CORE_BUILD_DOC=ON -DKALYPSSO_CORE_DOC=doxygen
cd _build/doc
make kalypsso-doc
```

This will generate the html doxygen page in `doc/doxygen/html`

### [mkdocs](https://www.mkdocs.org/) with [MkDoxy plugin](https://github.com/JakubAndrysek/MkDoxy)

```shell
# generate mkdocs sources
cd build
cmake --build _build/doc -S . -DKALYPSSO_CORE_BUILD_DOC=ON -DKALYPSSO_CORE_DOC=mkdocs
cd _build/doc
make
make mkdocs
```

This will generate the markdown sources for the mkdocs static webpage.

```shell
# from the build directory
cd doc/mkdocs

# preview of the webpage
mkdocs serve
# open url localhost:8000

# if you want to build the html sources (before deployment)
mkdocs build

# this will create directory `site` that can directly be uploaded to
# a web server
```

# Citing kalypsso

If you use this software, please cite it using the following reference.

```
@article{kalypsso_core_cpc26,
  author={Kestener, Pierre},
  journal={Computer Physics Communication},
  title={kalypsso: a performance portable platform for compressible hydrodynamics simulations using adaptive mesh refinement},
  year={2026},
  volume={},
  number={},
  pages={},
  doi={10.1016/j.cpc.2026.110275}}
  ```
