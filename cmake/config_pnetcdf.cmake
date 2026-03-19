# ##################################################################################################
# PNETCDF
# ##################################################################################################
if(KALYPSSO_CORE_USE_MPI)
  if(KALYPSSO_CORE_USE_PNETCDF)
    find_package(PNETCDF)
    if(PNETCDF_FOUND)
      add_compile_options(-DKALYPSSO_CORE_USE_PNETCDF)
      include_directories(${PNETCDF_INCLUDE_DIRS})
    endif(PNETCDF_FOUND)
  endif(KALYPSSO_CORE_USE_PNETCDF)
endif(KALYPSSO_CORE_USE_MPI)
