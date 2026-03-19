if(KALYPSSO_CORE_USE_MPI)
  # =======================================================================
  # =======================================================================
  add_executable(parallel_hdf5_collective_io_write parallel_hdf5_collective_io_write.cpp)
  target_compile_options(parallel_hdf5_collective_io_write PRIVATE -DKALYPSSO_IO_HIGH_FIVE)
  target_link_libraries(parallel_hdf5_collective_io_write
                        PUBLIC kalypsso::mpiUtils kalypsso::core_config HighFive::HighFive)

  # =======================================================================
  # =======================================================================
  add_executable(parallel_hdf5_collective_io_read parallel_hdf5_collective_io_read.cpp)
  target_compile_options(parallel_hdf5_collective_io_read PRIVATE -DKALYPSSO_IO_HIGH_FIVE)
  target_link_libraries(parallel_hdf5_collective_io_read
                        PUBLIC kalypsso::mpiUtils kalypsso::core_config HighFive::HighFive)
endif(KALYPSSO_CORE_USE_MPI)
