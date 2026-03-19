# =======================================================================
# =======================================================================
add_executable(test_io_vtk "")
target_sources(test_io_vtk PRIVATE test_io.cpp)
target_compile_options(test_io_vtk PRIVATE -DKALYPSSO_IO_VTK)
target_link_libraries(test_io_vtk PUBLIC kalypsso::core kalypsso::p4est_wrap kalypsso::config
                                         kalypsso::mpiUtils)

add_test(NAME io_vtk_2d COMMAND ${TEST_MPI_PREFIX_CMD} test_io_vtk brick
                                ${KALYPSSO_TEST_AMR_MAX_LEVEL} 5 3)
set_property(TEST io_vtk_2d PROPERTY LABELS io 2D)
add_test(NAME io_vtk_3d COMMAND ${TEST_MPI_PREFIX_CMD} test_io_vtk brick3
                                ${KALYPSSO_TEST_AMR_MAX_LEVEL} 4 3 2)
set_property(TEST io_vtk_2d PROPERTY LABELS io 3D)

if(Kokkos_ENABLE_OPENMP)
  set_property(TEST io_vtk_2d PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                       OMP_PROC_BIND=spread OMP_PLACES=threads)
  set_property(TEST io_vtk_3d PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                       OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

# =======================================================================
# =======================================================================
if(KALYPSSO_CORE_USE_HDF5)
  add_executable(test_io_hdf5_block "")
  target_sources(test_io_hdf5_block PRIVATE test_io.cpp)
  target_compile_options(test_io_hdf5_block PRIVATE -DKALYPSSO_IO_HDF5_BLOCK)
  target_link_libraries(test_io_hdf5_block PUBLIC kalypsso::core kalypsso::p4est_wrap
                                                  kalypsso::config kalypsso::mpiUtils)

  add_test(NAME io_hdf5_block_2d COMMAND ${TEST_MPI_PREFIX_CMD} test_io_hdf5_block brick
                                         ${KALYPSSO_TEST_AMR_MAX_LEVEL} 5 3)

  set_property(TEST io_hdf5_block_2d PROPERTY LABELS io 2D)

  add_test(NAME io_hdf5_block_3d COMMAND ${TEST_MPI_PREFIX_CMD} test_io_hdf5_block brick3
                                         ${KALYPSSO_TEST_AMR_MAX_LEVEL} 4 3 2)
  set_property(TEST io_hdf5_block_3d PROPERTY LABELS io 3D)

  if(Kokkos_ENABLE_OPENMP)
    set_property(
      TEST io_hdf5_block_2d PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                     OMP_PROC_BIND=spread OMP_PLACES=threads)
    set_property(
      TEST io_hdf5_block_3d PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                     OMP_PROC_BIND=spread OMP_PLACES=threads)
  endif()
endif()

# =======================================================================
# =======================================================================
if(KALYPSSO_CORE_USE_HDF5)
  add_executable(test_io_hdf5_ghosted_block "")
  target_sources(test_io_hdf5_ghosted_block PRIVATE test_io.cpp)
  target_compile_options(test_io_hdf5_ghosted_block PRIVATE -DKALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  target_link_libraries(test_io_hdf5_ghosted_block PUBLIC kalypsso::core kalypsso::p4est_wrap
                                                          kalypsso::config kalypsso::mpiUtils)
endif()

# =======================================================================
# =======================================================================
if(KALYPSSO_CORE_USE_HDF5)
  add_executable(test_io_brick_block "")
  target_sources(test_io_brick_block PRIVATE test_io_brick.cpp)
  target_compile_options(test_io_brick_block PRIVATE -DKALYPSSO_IO_HDF5_BLOCK)
  target_link_libraries(test_io_brick_block PUBLIC kalypsso::core kalypsso::p4est_wrap
                                                   kalypsso::config kalypsso::mpiUtils)
endif()

# =======================================================================
# =======================================================================
if(KALYPSSO_CORE_USE_HDF5)
  add_executable(test_io_brick_ghosted_block "")
  target_sources(test_io_brick_ghosted_block PRIVATE test_io_brick.cpp)
  target_compile_options(test_io_brick_ghosted_block PRIVATE -DKALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  target_link_libraries(test_io_brick_ghosted_block PUBLIC kalypsso::core kalypsso::p4est_wrap
                                                           kalypsso::config kalypsso::mpiUtils)
endif()

# =======================================================================
# =======================================================================

configure_file(test_io_vtk.ini test_io_vtk.ini COPYONLY)
configure_file(test_io_hdf5_block.ini test_io_hdf5_block.ini COPYONLY)

#
# cleanup any artefact hdf5 / xmf / vtk files produced by running the targets
#
add_custom_target(clean_io_data COMMAND ${CMAKE_COMMAND} -P
                                        ${CMAKE_CURRENT_SOURCE_DIR}/clean_data.cmake)
