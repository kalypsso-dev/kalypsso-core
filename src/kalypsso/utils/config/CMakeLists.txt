add_library(kalypsso_config ConfigMap.cpp ConfigMap.h inih/ini.c inih/ini.h)

target_include_directories(
  kalypsso_config
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inih>
         $<INSTALL_INTERFACE:include/kalypsso/utils>)

target_link_libraries(kalypsso_config PUBLIC Kokkos::kokkos)
target_link_libraries(kalypsso_config PRIVATE kalypsso_warnings kalypsso_core_config)

if(KALYPSSO_CORE_USE_MPI)
  target_link_libraries(kalypsso_config PUBLIC MPI::MPI_CXX)
endif(KALYPSSO_CORE_USE_MPI)

# Add an alias so that library can be used inside the build tree
add_library(kalypsso::config ALIAS kalypsso_config)

#
# Get prepared for install
#
include(${PROJECT_SOURCE_DIR}/cmake/setup_install.cmake)

#
# install libraries and headers
#
install(
  TARGETS kalypsso_config
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
