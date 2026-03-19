set(VTK_KALYPSSO_HDF5_READER_CLASSES vtkKalypssoHDF5Reader)

set(VTK_KALYPSSO_HDF5_READER_SOURCES vtkKalypssoHDF5Reader.cxx)

set(VTK_KALYPSSO_HDF5_READER_HEADERS vtkKalypssoHDF5Reader.h)

vtk_module_add_module(
  pvKalypssoHDF5Reader::KalypssoHDF5Reader
  CLASSES
  ${VTK_KALYPSSO_HDF5_READER_CLASSES}
  SOURCES
  ${VTK_KALYPSSO_HDF5_READER_SOURCES}
  HEADERS
  ${VTK_KALYPSSO_HDF5_READER_HEADERS})

# vtk_module_third_party_external( PACKAGE Kokkos TARGETS Kokkos::kokkos)

# vtk_module_find_package( PACKAGE Kokkos )

vtk_module_link(pvKalypssoHDF5Reader::KalypssoHDF5Reader PUBLIC HighFive::HighFive kalypsso::core)

paraview_add_server_manager_xmls(XMLS KalypssoHDF5_Server.xml)
