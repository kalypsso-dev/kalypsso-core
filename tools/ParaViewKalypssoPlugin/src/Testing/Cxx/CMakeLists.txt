add_executable(TestKalypssoHDF5Reader TestKalypssoHDF5Reader.cxx)
target_link_libraries(
  TestKalypssoHDF5Reader
  PUBLIC KalypssoHDF5Reader
  PRIVATE VTK::CommonCore VTK::CommonDataModel VTK::IOCore VTK::IOGeometry VTK::IOXML
          VTK::IOXMLParser)
