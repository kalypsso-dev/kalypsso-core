# ##################################################################################################
# googletest (for unit testing)
# ##################################################################################################
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  # GIT_REPOSITORY git@github.com:google/googletest.git
  GIT_TAG release-1.12.0
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/googletest)
FetchContent_MakeAvailable(googletest)

# make googletest cmake macros available
include(GoogleTest)

set(TEST_USE_MPI_ON ON)
set(TEST_USE_MPI_OFF OFF)

# ##################################################################################################
#
# function to simplify creating a unit test
#
# Parameters: * unit_test_name : the executable test name * test_use_mpi   : boolean value to
# indicate if test uses MPI * all other arg are expected to be source files
#
# ##################################################################################################
function(kalypsso_add_test unit_test_name test_use_mpi)
  add_executable(${unit_test_name} "")
  target_sources(${unit_test_name} PRIVATE ${ARGN} kalypsso_unittest_utils.h)

  target_include_directories(${unit_test_name} PUBLIC ${PROJECT_SOURCE_DIR}/unit_test
                                                      ${PROJECT_SOURCE_DIR}/test)
  target_link_libraries(
    ${unit_test_name}
    PUBLIC kalypsso::config
           Kokkos::kokkos
           dl
           kalypsso::monitoring
           kalypsso::core
           kalypsso::AMRtest
           gtest)

  if(KALYPSSO_CORE_USE_MPI)
    target_link_libraries(${unit_test_name} PUBLIC kalypsso::mpiUtils)
  endif()

  # when test_use_mpi is ON, run the test with ${KALYPSSO_TEST_MPIEXEC_NUMPROCS} MPI processes
  # KALYPSSO_TEST_MPIEXEC_NUMPROCS is defined in test/CMakeLists.txt; default value is currently 2
  if(KALYPSSO_CORE_USE_MPI)
    if(${test_use_mpi})
      add_test(
        NAME "${unit_test_name}_MPI"
        COMMAND cmake -E env OMP_NUM_THREADS=4 ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
                ${KALYPSSO_TEST_MPIEXEC_NUMPROCS} ./${unit_test_name}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/unit_test/)
    else()
      # test doesn't require MPI, so just use a single MPI process
      add_test(
        NAME "${unit_test_name}_MPI"
        COMMAND cmake -E env OMP_NUM_THREADS=4 ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                ./${unit_test_name}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/unit_test/)
    endif()
  else()
    add_test(
      NAME ${unit_test_name}
      COMMAND cmake -E env OMP_NUM_THREADS=4 ./${unit_test_name}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/unit_test/)
  endif()

endfunction(kalypsso_add_test)

# ##################################################################################################
#
# here we create our tests
#
# ##################################################################################################

#
# core
#
kalypsso_add_test(geometry_utils_test TEST_USE_MPI_OFF core/geometry_utils_test.cpp main.cpp)

kalypsso_add_test(morton_utils_test TEST_USE_MPI_OFF core/morton_utils_test.cpp main.cpp)

kalypsso_add_test(orchard_key_test TEST_USE_MPI_OFF core/orchard_key_test.cpp main.cpp)

kalypsso_add_test(BitFieldInteger_test TEST_USE_MPI_OFF core/BitFieldInteger_test.cpp main.cpp)

kalypsso_add_test(kalypsso_data_container_test TEST_USE_MPI_OFF
                  core/kalypsso_data_container_test.cpp main.cpp)

kalypsso_add_test(DataArrayBlock_utils_test TEST_USE_MPI_OFF core/DataArrayBlock_utils_test.cpp
                  main.cpp)

kalypsso_add_test(Kokkos_Array_extensions_test TEST_USE_MPI_OFF
                  core/Kokkos_Array_extensions_test.cpp main.cpp)

kalypsso_add_test(mesh_utils_test TEST_USE_MPI_OFF core/mesh_utils_test.cpp main.cpp)

kalypsso_add_test(EdgeDataArrayBlock_utils_test TEST_USE_MPI_OFF
                  core/EdgeDataArrayBlock_utils_test.cpp main.cpp)

kalypsso_add_test(StencilHelper_test TEST_USE_MPI_OFF core/StencilHelper_test.cpp main.cpp)

configure_file(${PROJECT_SOURCE_DIR}/test/AMRmesh/test_AMRmesh_brick_2d.ini StencilHelper_2d.ini
               COPYONLY)
# configure_file(${PROJECT_SOURCE_DIR}/test/AMRmesh/test_AMRmesh_brick_3d.ini StencilHelper_3d.ini
# COPYONLY)
configure_file(StencilHelper_3d.ini StencilHelper_3d.ini COPYONLY)

kalypsso_add_test(DataArrayBlockMultiVar_test TEST_USE_MPI_OFF core/DataArrayBlockMultiVar_test.cpp
                  main.cpp)

kalypsso_add_test(MaterialPresence_test TEST_USE_MPI_OFF core/MaterialPresence_test.cpp main.cpp)

kalypsso_add_test(interface_tracking_utils_test TEST_USE_MPI_OFF
                  core/vof/interface_tracking_utils_test.cpp main.cpp)
kalypsso_add_test(youngs_test TEST_USE_MPI_OFF core/vof/youngs_test.cpp main.cpp)

# TODO: Must change "kalypsso_add_test" function because this explicitly requires MPI
# kalypsso_add_test(MaterialPresence_test_mpi TEST_USE_MPI_ON core/MaterialPresence_test_mpi.cpp
# main.cpp)

#
# utils/config
#
kalypsso_add_test(ConfigMap_test TEST_USE_MPI_OFF utils/config/ConfigMap_test.cpp main.cpp)

#
# utils/monitoring
#
kalypsso_add_test(HostTimer_test TEST_USE_MPI_OFF utils/monitoring/HostTimer_test.cpp main.cpp)

#
# utils/mpi
#
kalypsso_add_test(ParallelEnv_test TEST_USE_MPI_ON utils/mpi/ParallelEnv_test.cpp
                  main_kalypsso_unittest.h main.cpp)
