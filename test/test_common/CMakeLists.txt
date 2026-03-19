# ##################################################################################################
add_library(
  AMRtest
  InitialAMRSetup.h
  InitialAMRSetup.cpp
  DataWriter.h
  DataWriter.cpp
  FillOutside.h
  FillOutside.cpp
  FillOutside2.h
  FillOutside2.cpp
  test_func.h)

target_include_directories(AMRtest PUBLIC ${CMAKE_SOURCE_DIR}/test)

target_link_libraries(AMRtest PUBLIC Kokkos::kokkos kalypsso::core kalypsso::log)
target_link_libraries(AMRtest PRIVATE kalypsso_warnings)

add_library(kalypsso::AMRtest ALIAS AMRtest)
