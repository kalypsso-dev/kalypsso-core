message(STATUS "Building kalypsso dependencies")

#
# Do we want to build p4est (https://github.com/cburstedde/p4est) ?
#
option(KALYPSSO_P4EST_BUILD "Turn ON if you want to build p4est (default: OFF)" OFF)
include(cmake/build_p4est.cmake)

#
# Do we want to build hdf5-parallel (https://github.com/HDFGroup/hdf5) ?
#
option(KALYPSSO_HDF5_BUILD "Turn ON if you want to build hdf5-parallel (default: OFF)" OFF)
include(cmake/build_hdf5.cmake)
