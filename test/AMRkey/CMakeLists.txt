add_executable(test_orchard test_orchard.cpp)

target_link_libraries(test_orchard PUBLIC kalypsso::core kalypsso::p4est_wrap kalypsso::config)

add_test(NAME orchard COMMAND test_orchard)
set_property(TEST orchard PROPERTY LABELS AMRkey)

if(Kokkos_ENABLE_OPENMP)
  set_property(TEST orchard PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                     OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

#
# this is not a test, just a utility to learn how orchard keys work
#
find_package(Boost COMPONENTS program_options)
if(Boost_FOUND)

  add_executable(decode_orchard decode_orchard.cpp)

  target_link_libraries(decode_orchard PUBLIC kalypsso::core Boost::program_options)

  add_executable(encode_orchard encode_orchard.cpp)

  target_link_libraries(encode_orchard PUBLIC kalypsso::core Boost::program_options)

endif()
