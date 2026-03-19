add_library(
  kalypsso_p4est_wrap
  ${CMAKE_CURRENT_SOURCE_DIR}/connectivity.h ${CMAKE_CURRENT_SOURCE_DIR}/connectivity.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/geometry.h ${CMAKE_CURRENT_SOURCE_DIR}/geometry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/p4est_wrapper.h ${CMAKE_CURRENT_SOURCE_DIR}/p4est_wrapper.cpp)

# add kalypsso/utils to include dirs for kalypsso_core_config.h
target_include_directories(kalypsso_p4est_wrap PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                      $<INSTALL_INTERFACE:include/kalypsso/utils>)

target_link_libraries(kalypsso_p4est_wrap PRIVATE kalypsso_warnings kalypsso_core_config)
target_link_libraries(kalypsso_p4est_wrap PUBLIC kalypsso::config P4EST::P4EST SC::SC)

if(KALYPSSO_CORE_USE_MPI)
  target_link_libraries(kalypsso_p4est_wrap PUBLIC MPI::MPI_CXX)
endif(KALYPSSO_CORE_USE_MPI)

# Add an alias so that library can be used inside the build tree
add_library(kalypsso::p4est_wrap ALIAS kalypsso_p4est_wrap)

#
# Get prepared for install
#
include(${PROJECT_SOURCE_DIR}/cmake/setup_install.cmake)

#
# install libraries and headers
#
install(
  TARGETS kalypsso_p4est_wrap
  EXPORT kalypsso-targets
  LIBRARY DESTINATION ${INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${INSTALL_LIBDIR}
  RUNTIME DESTINATION ${INSTALL_BINDIR})

#
# install headers (exclude cpp files)
#
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DESTINATION ${INSTALL_INCLUDEDIR}/kalypsso/utils
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cpp" EXCLUDE)
