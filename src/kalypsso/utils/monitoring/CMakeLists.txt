add_library(kalypsso_monitoring HostTimer.h CudaBlockingTimer.h CudaAsyncTimer.h memory_utils.h
                                ProfilingManager.h ProfilingManager.cpp)

target_include_directories(
  kalypsso_monitoring
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/utils>
         $<INSTALL_INTERFACE:include/kalypsso/utils>)

target_link_libraries(kalypsso_monitoring PRIVATE kalypsso_warnings kalypsso_core_config)

target_link_libraries(kalypsso_monitoring PUBLIC kalypsso::config kalypsso::log kalypsso::mpiUtils
                                                 kalypsso::p4est_wrap Kokkos::kokkos)

if(KALYPSSO_CORE_NVTX_ANNOTATION_ENABLE AND KALYPSSO_CORE_USE_NVTX_FROM_GITHUB)
  target_link_libraries(kalypsso_monitoring PUBLIC nvtx3-c)
endif(KALYPSSO_CORE_NVTX_ANNOTATION_ENABLE AND KALYPSSO_CORE_USE_NVTX_FROM_GITHUB)

# Add an alias so that library can be used inside the build tree
add_library(kalypsso::monitoring ALIAS kalypsso_monitoring)

#
# Get prepared for install
#
include(${PROJECT_SOURCE_DIR}/cmake/setup_install.cmake)

#
# install libraries and headers
#
install(
  TARGETS kalypsso_monitoring
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
