# ##################################################################################################
# VTK configuration tips, see /usr/lib/cmake/vtk-6.3/VTKConfig.cmake
# /usr/lib/cmake/vtk-6.3/UseVTK.cmake
# ##################################################################################################
if(KALYPSSO_CORE_USE_VTK)
  # look for VTK only if requested; VTK macro might even be not present on the target platform
  find_package(VTK)

  # the following add VTK to all targets
  # cmake-format: off
  # if(VTK_FOUND)
  #   include(${VTK_USE_FILE})
  # endif(VTK_FOUND)
  # cmake-format: on
  if(VTK_FOUND)
    message("***VTK FOUND ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
  else()
    message("*** VTK NOT FOUND")
  endif()
endif(KALYPSSO_CORE_USE_VTK)
