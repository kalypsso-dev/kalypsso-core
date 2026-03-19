# What is `kalypsso_dependencies`

This is just a small cmake sub-project that only aims at facilitating the build of some optional or required dependencies of kalypsso.

## Quick start

Example of use when building only `p4est`:

```shell
# from kalypsso top-level source directory
mkdir -p _build/dependencies
cd _build/dependencies
cmake -DKALYPSSO_P4EST_BUILD=ON ../../dependencies
make
```

By default, all dependencies will be installed in `$HOME/.kalypsso_dependencies`.
You can change that by setting cmake variable `CMAKE_INSTALL_PREFIX`.

Additionally, a modulefile is also provided; once you built all chosen dependencies,
you just need to do the following:

```shell
module use ~/.kalypsso_dependencies/share/modulefiles
module use kalypsso_dependencies/1.0
```

and you're good to go building `kalypsso` with this dependencies.

## List of dependencies

Here is the list of dependencies that can be built:

- [p4est](https://github.com/cburstedde/p4est)
- [hdf5](https://www.hdfgroup.org/solutions/hdf5/)

## Some features

### Specify source to compile

For most dependencies, you can chose which source you want to build, either a git tag or release archive (remote or local file).

E.g. if your already have downloaded a p4est archive, you can configure `kalypsso_dependencies` using cmake var `KALYPSSO_P4EST_SOURCE_ARCHIVE` to specify the tarball path on your local machine.

```shell
# from kalypsso top-level source directory
mkdir -p _build/dependencies
cd _build/dependencies
cmake -DKALYPSSO_P4EST_BUILD=ON -DKALYPSSO_P4EST_SOURCE_ARCHIVE=$HOME/p4est-2.8.5.tar.gz -DKALYPSSO_HDF5_BUILD=ON ../../dependencies
make
```

Warning: don't use `~` in the archive filepath, because this is passed to cmake as a string, that can either be a local filesystem path or a remote url; just the full path, of `$HOME`.

### Recompile a dependency

Suppose you want to recompile `p4est`, just remove directory `external/p4est` and re-execute cmake configure and then `make` again.

We also advise you to also clean the install location (default is $HOME/.kalypsso_dependencies), to make sure to restart in a clean directory.

### dependencies of dependencies

Currently all dependencies listed above are independent one for another, but we could have the situation where two dependencies actually depend from a third one. In this, this actually possible since cmake macro `ExternalProject_Add` can manage dependencies upon completion of another `ExternalProject_Add`.
