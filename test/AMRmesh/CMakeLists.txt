add_executable(test_AMRmesh "")
target_sources(test_AMRmesh PRIVATE test_AMRmesh.cpp)
target_link_libraries(test_AMRmesh PUBLIC kalypsso::core kalypsso::core_config kalypsso::p4est_wrap
                                          kalypsso::config kalypsso::mpiUtils)

add_test(NAME AMRmesh COMMAND ${TEST_MPI_PREFIX_CMD} test_AMRmesh)

if(Kokkos_ENABLE_OPENMP)
  set_property(TEST AMRmesh PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                     OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

configure_file(test_AMRmesh_2d.ini test_AMRmesh_2d.ini COPYONLY)
configure_file(test_AMRmesh_3d.ini test_AMRmesh_3d.ini COPYONLY)
configure_file(test_AMRmesh_simple_2d.ini test_AMRmesh_simple_2d.ini COPYONLY)
configure_file(test_AMRmesh_simple_3d.ini test_AMRmesh_simple_3d.ini COPYONLY)
configure_file(test_AMRmesh_brick_2d.ini test_AMRmesh_brick_2d.ini COPYONLY)
configure_file(test_AMRmesh_brick_3d.ini test_AMRmesh_brick_3d.ini COPYONLY)
configure_file(test_AMRmesh_brick_2d_ghost_exchange_mpi.ini
               test_AMRmesh_brick_2d_ghost_exchange_mpi.ini COPYONLY)
configure_file(test_AMRmesh_brick_3d_ghost_exchange_mpi.ini
               test_AMRmesh_brick_3d_ghost_exchange_mpi.ini COPYONLY)
configure_file(test_AMRmesh_fill_ghost_brick_2d.ini test_AMRmesh_fill_ghost_brick_2d.ini COPYONLY)
configure_file(test_AMRmesh_fill_ghost_brick_3d.ini test_AMRmesh_fill_ghost_brick_3d.ini COPYONLY)
configure_file(test_AMRmesh_brick_2d.ini test_AMRmesh_stencil_2d.ini COPYONLY)
configure_file(test_AMRmesh_brick_3d.ini test_AMRmesh_stencil_3d.ini COPYONLY)
configure_file(test_AMRmesh_userdata_2d.ini test_AMRmesh_userdata_2d.ini COPYONLY)
configure_file(test_AMRmesh_userdata_3d.ini test_AMRmesh_userdata_3d.ini COPYONLY)

add_executable(test_AMRmesh_userdata "")
target_sources(test_AMRmesh_userdata PRIVATE test_AMRmesh_userdata.cpp)
target_include_directories(
  test_AMRmesh_userdata PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                               $<INSTALL_INTERFACE:include/kalypsso/test>)
target_link_libraries(test_AMRmesh_userdata PUBLIC kalypsso::core kalypsso::p4est_wrap
                                                   kalypsso::config kalypsso::AMRtest)

add_test(NAME AMRmesh_userdata_2d COMMAND ${TEST_MPI_PREFIX_CMD} test_AMRmesh_userdata --ini
                                          test_AMRmesh_userdata_2d.ini)
# add_test(NAME AMRmesh_userdata_3d COMMAND ${TEST_MPI_PREFIX_CMD} test_AMRmesh_userdata --ini
# test_AMRmesh_userdata_3d.ini --3d)

if(Kokkos_ENABLE_OPENMP)
  set_property(
    TEST AMRmesh_userdata_2d PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                      OMP_PROC_BIND=spread OMP_PLACES=threads)
  # set_property(TEST AMRmesh_userdata_3d PROPERTY ENVIRONMENT
  # OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS} OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################
add_executable(test_AMRContext "")
target_sources(test_AMRContext PRIVATE test_AMRContext.cpp)
target_include_directories(test_AMRContext PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                                  $<INSTALL_INTERFACE:include/kalypsso/test>)
target_link_libraries(
  test_AMRContext PUBLIC kalypsso::core kalypsso::core_config kalypsso::p4est_wrap kalypsso::config
                         kalypsso::mpiUtils kalypsso::AMRtest)
target_link_libraries(test_AMRContext PRIVATE kalypsso_warnings)

if(KALYPSSO_CORE_USE_SPDLOG)
  target_compile_definitions(test_AMRContext
                             PRIVATE -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})
  target_link_libraries(test_AMRContext PUBLIC spdlog::spdlog)
endif()

if(KALYPSSO_CORE_USE_CNPY)
  target_link_libraries(test_AMRContext PUBLIC cnpy::cnpy)
endif()

add_test(NAME AMRContext_2d COMMAND ${TEST_MPI_PREFIX_CMD} test_AMRContext --ini
                                    test_AMRmesh_brick_2d.ini --refine_level 6)
add_test(NAME AMRContext_3d COMMAND ${TEST_MPI_PREFIX_CMD} test_AMRContext --ini
                                    test_AMRmesh_brick_2d.ini --refine_level 6 --3d)

if(Kokkos_ENABLE_OPENMP)
  set_property(
    TEST AMRContext_2d PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                OMP_PROC_BIND=spread OMP_PLACES=threads)
  set_property(
    TEST AMRContext_3d PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################
if(KALYPSSO_CORE_USE_MPI)

  add_executable(test_AMR_ghost_exchange_mpi "")

  target_sources(test_AMR_ghost_exchange_mpi PRIVATE test_AMR_ghost_exchange_mpi.cpp)

  target_include_directories(
    test_AMR_ghost_exchange_mpi PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                       $<INSTALL_INTERFACE:include/kalypsso/test>)

  target_link_libraries(
    test_AMR_ghost_exchange_mpi PUBLIC kalypsso::core kalypsso::p4est_wrap kalypsso::config
                                       kalypsso::mpiUtils kalypsso::AMRtest)
  target_link_libraries(test_AMR_ghost_exchange_mpi PRIVATE kalypsso_warnings)

  if(KALYPSSO_CORE_USE_SPDLOG)
    target_compile_definitions(test_AMR_ghost_exchange_mpi
                               PRIVATE -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})
    target_link_libraries(test_AMR_ghost_exchange_mpi PUBLIC spdlog::spdlog)
  endif(KALYPSSO_CORE_USE_SPDLOG)

endif(KALYPSSO_CORE_USE_MPI)

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################
add_executable(test_AMRmesh_fill_block_ghost_cells_v2 "")
target_sources(test_AMRmesh_fill_block_ghost_cells_v2
               PRIVATE test_AMRmesh_fill_block_ghost_cells_v2.cpp)

target_include_directories(
  test_AMRmesh_fill_block_ghost_cells_v2 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                                $<INSTALL_INTERFACE:include/kalypsso/test>)

target_link_libraries(
  test_AMRmesh_fill_block_ghost_cells_v2
  PUBLIC kalypsso::core kalypsso::core_config kalypsso::p4est_wrap kalypsso::config
         kalypsso::mpiUtils kalypsso::AMRtest)

target_link_libraries(test_AMRmesh_fill_block_ghost_cells_v2 PRIVATE kalypsso_warnings)

if(KALYPSSO_CORE_USE_SPDLOG)
  target_compile_definitions(test_AMRmesh_fill_block_ghost_cells_v2
                             PRIVATE -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})
  target_link_libraries(test_AMRmesh_fill_block_ghost_cells_v2 PUBLIC spdlog::spdlog)
endif()

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################
add_executable(test_AMRmesh_fill_block_ghost_cells "")
target_sources(test_AMRmesh_fill_block_ghost_cells PRIVATE test_AMRmesh_fill_block_ghost_cells.cpp)

target_include_directories(
  test_AMRmesh_fill_block_ghost_cells PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                             $<INSTALL_INTERFACE:include/kalypsso/test>)

target_link_libraries(
  test_AMRmesh_fill_block_ghost_cells PUBLIC kalypsso::core kalypsso::p4est_wrap kalypsso::config
                                             kalypsso::mpiUtils kalypsso::AMRtest)

target_link_libraries(test_AMRmesh_fill_block_ghost_cells PRIVATE kalypsso_warnings)

if(KALYPSSO_CORE_USE_SPDLOG)
  target_compile_definitions(test_AMRmesh_fill_block_ghost_cells
                             PRIVATE -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})
  target_link_libraries(test_AMRmesh_fill_block_ghost_cells PUBLIC spdlog::spdlog)
endif()

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################
add_executable(test_AMRmesh_fill_block_ghost_cells_in_place "")
target_sources(test_AMRmesh_fill_block_ghost_cells_in_place
               PRIVATE test_AMRmesh_fill_block_ghost_cells_in_place.cpp)
target_include_directories(
  test_AMRmesh_fill_block_ghost_cells_in_place
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
         $<INSTALL_INTERFACE:include/kalypsso/test>)
target_link_libraries(
  test_AMRmesh_fill_block_ghost_cells_in_place
  PUBLIC kalypsso::core kalypsso::core_config kalypsso::p4est_wrap kalypsso::config
         kalypsso::mpiUtils kalypsso::AMRtest)

target_link_libraries(test_AMRmesh_fill_block_ghost_cells_in_place PRIVATE kalypsso_warnings)

if(KALYPSSO_CORE_USE_SPDLOG)
  target_compile_definitions(test_AMRmesh_fill_block_ghost_cells_in_place
                             PRIVATE -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})
  target_link_libraries(test_AMRmesh_fill_block_ghost_cells_in_place PUBLIC spdlog::spdlog)
endif()

#
# ##################################################################################################
#
# ##################################################################################################
#
# ##################################################################################################
add_executable(test_AMRmesh_fill_block_ghost_faces "")
target_sources(test_AMRmesh_fill_block_ghost_faces PRIVATE test_AMRmesh_fill_block_ghost_faces.cpp)

target_include_directories(
  test_AMRmesh_fill_block_ghost_faces PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                             $<INSTALL_INTERFACE:include/kalypsso/test>)

target_link_libraries(
  test_AMRmesh_fill_block_ghost_faces PUBLIC kalypsso::core kalypsso::p4est_wrap kalypsso::config
                                             kalypsso::mpiUtils kalypsso::AMRtest)

if(KALYPSSO_CORE_USE_SPDLOG)
  target_compile_definitions(test_AMRmesh_fill_block_ghost_faces
                             PRIVATE -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})
  target_link_libraries(test_AMRmesh_fill_block_ghost_faces PUBLIC spdlog::spdlog)
endif()

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################
add_executable(test_AMRmesh_refine_flags "")
target_sources(test_AMRmesh_refine_flags PRIVATE test_AMRmesh_refine_flags.cpp)
target_include_directories(
  test_AMRmesh_refine_flags PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                   $<INSTALL_INTERFACE:include/kalypsso/test>)
target_link_libraries(
  test_AMRmesh_refine_flags PUBLIC kalypsso::core kalypsso::core_config kalypsso::p4est_wrap
                                   kalypsso::config kalypsso::mpiUtils kalypsso::AMRtest)
target_link_libraries(test_AMRmesh_refine_flags PRIVATE kalypsso_warnings)

if(KALYPSSO_CORE_USE_SPDLOG)
  target_compile_definitions(test_AMRmesh_refine_flags
                             PRIVATE -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})
  target_link_libraries(test_AMRmesh_refine_flags PUBLIC spdlog::spdlog)
endif()

if(KALYPSSO_CORE_USE_CNPY)
  target_link_libraries(test_AMRmesh_refine_flags PUBLIC cnpy::cnpy)
endif()

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################
add_executable(test_AMRmesh_stencil "")
target_sources(test_AMRmesh_stencil PRIVATE test_AMRmesh_stencil.cpp)
target_include_directories(
  test_AMRmesh_stencil PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                              $<INSTALL_INTERFACE:include/kalypsso/test>)
target_link_libraries(
  test_AMRmesh_stencil PUBLIC kalypsso::core kalypsso::p4est_wrap kalypsso::config
                              kalypsso::mpiUtils kalypsso::AMRtest)
target_link_libraries(test_AMRmesh_stencil PRIVATE kalypsso_warnings)

if(KALYPSSO_CORE_USE_SPDLOG)
  target_compile_definitions(test_AMRmesh_stencil
                             PRIVATE -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})
  target_link_libraries(test_AMRmesh_stencil PUBLIC spdlog::spdlog)
endif()

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################
add_executable(test_first_derivative "")
target_sources(test_first_derivative PRIVATE test_first_derivative.cpp)
target_include_directories(
  test_first_derivative PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                               $<INSTALL_INTERFACE:include/kalypsso/test>)
target_link_libraries(
  test_first_derivative PUBLIC kalypsso::core kalypsso::core_config kalypsso::p4est_wrap
                               kalypsso::config kalypsso::mpiUtils kalypsso::AMRtest)
target_link_libraries(test_first_derivative PRIVATE kalypsso_warnings)

if(KALYPSSO_CORE_USE_SPDLOG)
  target_compile_definitions(test_first_derivative
                             PRIVATE -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})
  target_link_libraries(test_first_derivative PUBLIC spdlog::spdlog)
endif()

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################
add_executable(test_AMRmesh_stencil_helper "")
target_sources(test_AMRmesh_stencil_helper PRIVATE test_AMRmesh_stencil_helper.cpp)
target_include_directories(
  test_AMRmesh_stencil_helper PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                     $<INSTALL_INTERFACE:include/kalypsso/test>)
target_link_libraries(
  test_AMRmesh_stencil_helper PUBLIC kalypsso::core kalypsso::p4est_wrap kalypsso::config
                                     kalypsso::mpiUtils kalypsso::AMRtest)
target_link_libraries(test_AMRmesh_stencil_helper PRIVATE kalypsso_warnings)

if(KALYPSSO_CORE_USE_SPDLOG)
  target_compile_definitions(test_AMRmesh_stencil_helper
                             PRIVATE -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})
  target_link_libraries(test_AMRmesh_stencil_helper PUBLIC spdlog::spdlog)
endif()

# ##################################################################################################
# ##################################################################################################
# ##################################################################################################
add_executable(test_brick_connectivity "")
target_sources(test_brick_connectivity PRIVATE test_brick_connectivity.cpp)
target_link_libraries(test_brick_connectivity PUBLIC kalypsso::core kalypsso::core_config)
# target_link_libraries(test_brick_connectivity PRIVATE kalypsso_warnings)

if(KALYPSSO_CORE_USE_CNPY)
  target_link_libraries(test_brick_connectivity PUBLIC cnpy::cnpy)
endif()

#
# copy python script to display orchard keys versus index (across all MPI process)
#
configure_file(plot_orchard_keys.py plot_orchard_keys.py)
configure_file(plot_brick_connectivity_2d.py plot_brick_connectivity_2d.py)
configure_file(plot_brick_connectivity_3d.py plot_brick_connectivity_3d.py)

#
# cleanup any artefact hdf5 / xmf / npy files produced by running the targets
#
add_custom_target(clean_AMRmesh_data COMMAND ${CMAKE_COMMAND} -P
                                             ${CMAKE_CURRENT_SOURCE_DIR}/clean_data.cmake)
