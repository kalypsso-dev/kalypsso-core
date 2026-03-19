add_library(kalypsso_io_c libb64/cencode.c)

add_library(
  kalypsso_io
  FileHandler.h
  FileHandler.cpp
  FileHandlerVtk.h
  FileHandlerVtk.cpp
  VTKWriter.h
  VTKWriter.cpp
  IO_VTK_shared.h
  IO_VTK_shared.cpp
  libb64/libb64.h)

target_include_directories(kalypsso_io PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
                                              $<INSTALL_INTERFACE:include/kalypsso/utils>)

target_link_libraries(kalypsso_io PRIVATE kalypsso_warnings kalypsso_core_config)

target_link_libraries(kalypsso_io PUBLIC kalypsso::config Kokkos::kokkos kalypsso_io_c)

if(KALYPSSO_CORE_USE_MPI)
  target_link_libraries(kalypsso_io PUBLIC MPI::MPI_CXX)
endif()

# Add an alias so that library can be used inside the build tree
add_library(kalypsso::io ALIAS kalypsso_io)

#
# install libraries and headers
#
install(
  TARGETS kalypsso_io kalypsso_io_c
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
