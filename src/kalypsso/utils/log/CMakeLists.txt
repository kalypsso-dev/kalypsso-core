add_library(kalypsso_log kalypsso_log.h kalypsso_log.cpp)

target_include_directories(
  kalypsso_log
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
         $<INSTALL_INTERFACE:include/kalypsso/utils>)

if(KALYPSSO_CORE_USE_MPI)
  target_link_libraries(kalypsso_log PUBLIC MPI::MPI_CXX)
endif(KALYPSSO_CORE_USE_MPI)

if(KALYPSSO_CORE_USE_SPDLOG)
  target_compile_definitions(kalypsso_log PUBLIC KALYPSSO_CORE_USE_SPDLOG)

  target_compile_definitions(kalypsso_log PUBLIC -DSPDLOG_ACTIVE_LEVEL=${KALYPSSO_CORE_LOG_LEVEL})

  target_link_libraries(kalypsso_log PRIVATE kalypsso_warnings kalypsso_core_config)

  target_link_libraries(kalypsso_log PUBLIC spdlog::spdlog)
endif(KALYPSSO_CORE_USE_SPDLOG)

# Add an alias so that library can be used inside the build tree
add_library(kalypsso::log ALIAS kalypsso_log)

#
# Get prepared for install
#
include(${PROJECT_SOURCE_DIR}/cmake/setup_install.cmake)

#
# install libraries and headers
#
install(
  TARGETS kalypsso_log
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
