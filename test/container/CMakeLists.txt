# add_executable(test_FaceDataArrayBlock test_FaceDataArrayBlock.cpp)

# target_link_libraries(test_FaceDataArrayBlock PUBLIC kalypsso::core kalypsso::config)

# add_test(NAME FaceDataArrayBlock COMMAND test_FaceDataArrayBlock) set_property(TEST
# FaceDataArrayBlock PROPERTY LABELS AMRkey)

# if(Kokkos_ENABLE_OPENMP) set_property(TEST FaceDataArrayBlock PROPERTY ENVIRONMENT
# OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS} OMP_PROC_BIND=spread OMP_PLACES=threads) endif()

#
# this is not a test, just a utility to play with DataArrayBlock
#
add_executable(create_DataArrayBlock create_DataArrayBlock.cpp)

target_link_libraries(create_DataArrayBlock PUBLIC kalypsso::core kalypsso::mpiUtils kalypsso::log)

#
# this is not a test, just a utility to play with FaceDataArrayBlock
#
add_executable(create_FaceDataArrayBlock create_FaceDataArrayBlock.cpp)

target_link_libraries(create_FaceDataArrayBlock PUBLIC kalypsso::core kalypsso::mpiUtils
                                                       kalypsso::log)

#
# this is not a test, just a utility to play with EdgeDataArrayBlock
#
add_executable(create_EdgeDataArrayBlock create_EdgeDataArrayBlock.cpp)

target_link_libraries(create_EdgeDataArrayBlock PUBLIC kalypsso::core kalypsso::mpiUtils
                                                       kalypsso::log)
