# ==========================================
add_executable(test_MeshMap "")
target_sources(test_MeshMap PRIVATE test_MeshMap.cpp)
target_link_libraries(test_MeshMap PUBLIC kalypsso::core kalypsso::p4est_wrap kalypsso::config
                                          kalypsso::mpiUtils)
target_link_libraries(test_MeshMap PRIVATE kalypsso_warnings)

add_test(NAME MeshMap COMMAND test_MeshMap)
set_property(TEST MeshMap PROPERTY LABELS MeshMap)

if(Kokkos_ENABLE_OPENMP)
  set_property(TEST MeshMap PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                     OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

# ==========================================
add_executable(test_MeshMap_fillOutside "")
target_sources(test_MeshMap_fillOutside PRIVATE test_MeshMap_fillOutside.cpp)
target_include_directories(
  test_MeshMap_fillOutside PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                  $<INSTALL_INTERFACE:include/kalypsso/test>)
target_link_libraries(
  test_MeshMap_fillOutside PUBLIC kalypsso::core kalypsso::p4est_wrap kalypsso::config
                                  kalypsso::mpiUtils kalypsso::AMRtest)
target_link_libraries(test_MeshMap_fillOutside PRIVATE kalypsso_warnings)

# add_test(NAME MeshMap COMMAND test_MeshMap_fillOutside) set_property(TEST MeshMap PROPERTY LABELS
# MeshMap_fillOutside)

configure_file(test_MeshMap_brick_2d.ini test_MeshMap_brick_2d.ini COPYONLY)
configure_file(test_MeshMap_brick_3d.ini test_MeshMap_brick_3d.ini COPYONLY)
