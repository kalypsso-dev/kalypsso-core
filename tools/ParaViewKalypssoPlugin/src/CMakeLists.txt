paraview_add_plugin(
  pvKalypssoHDF5Reader
  VERSION
  "1.0"
  REQUIRED_ON_SERVER
  MODULES
  pvKalypssoHDF5Reader::KalypssoHDF5Reader
  MODULE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/Reader/vtk.module")

option(KALYPSSO_PVPLUGIN_BUILD_TESTING "Build Plugin Testing" OFF)
if(KALYPSSO_PVPLUGIN_BUILD_TESTING AND BUILD_SHARED_LIBS)
  add_subdirectory(Testing)
endif()
