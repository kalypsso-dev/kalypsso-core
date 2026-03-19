# HighFive is a high level HDF5 wrapper library. If HDF5 is parallel, HighFive will use it; no
# special flags to use.
set(HIGHFIVE_UNIT_TESTS OFF)
set(HIGHFIVE_EXAMPLES OFF)
set(HIGHFIVE_BUILD_DOCS OFF)

add_subdirectory(external/HighFive SYSTEM)
