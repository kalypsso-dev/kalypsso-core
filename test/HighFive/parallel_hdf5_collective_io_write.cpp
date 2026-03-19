// Copyright (c), 2017, Adrien Devresse
// Copyright (c), 2022, Blue Brain Project
// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <highfive/highfive.hpp>

const std::string file_name("parallel_collective_example.h5");
const std::string dataset_name("dset");

// Currently, HighFive doesn't wrap retrieving information from property lists.
// Therefore, one needs to use HDF5 directly. For example, to see if collective
// MPI-IO operations were used, one may. Conveniently, this also provides identifiers
// of the cause for not using collective MPI calls.
void
check_collective_io(const HighFive::DataTransferProps & xfer_props)
{
  auto mnccp = HighFive::MpioNoCollectiveCause(xfer_props);
  if (mnccp.getLocalCause() || mnccp.getGlobalCause())
  {
    std::cout << "The operation was successful, but couldn't use collective MPI-IO. local cause: "
              << mnccp.getLocalCause() << " global cause:" << mnccp.getGlobalCause() << std::endl;
  }
}

// This is an example of how to write HDF5 files when all
// operations are collective, i.e. all MPI ranks participate in
// all HDF5 related function calls.
//
// If this assumption is met then one can ask HDF5 to use
// collective MPI-IO operations. This enables MPI-IO to optimize
// reads and writes.
//
// In this example we will create groups, and let every MPI rank
// write part of a 2D array; and then have all MPI ranks read back
// a different part of the array.
int
main(int argc, char ** argv)
{
  // initialize mpi, kokkos and p4est, use default MPI communicator : MPI_COMM_WORLD
  kalypsso::ParallelEnv par_env(argc, argv);

  try
  {
    // MPI-IO requires informing HDF5 that we want something other than
    // the default behaviour. This is done through property lists. We
    // need a file access property list.
    HighFive::FileAccessProps fapl;
    // We tell HDF5 to use MPI-IO
    fapl.add(HighFive::MPIOFileAccess{ par_env.mpi_comm(), MPI_INFO_NULL });
    // We also specify that we want all meta-data related operations
    // to use MPI collective operations. This implies that all MPI ranks
    // in the communicator must participate in any HDF5 operation that
    // reads or writes metadata. Essentially, this is safe if all MPI ranks
    // participate in all HDF5 operations.
    fapl.add(HighFive::MPIOCollectiveMetadata{});

    // Now we can create the file as usual.
    HighFive::File file(file_name, HighFive::File::Truncate, fapl);

    // We can create a group as usual, but all MPI ranks must participate.
    auto group = file.createGroup("grp");

    // We define the dataset have one row per MPI rank and two columns.
    std::vector<size_t> dims(2);
    dims[0] = std::size_t(par_env.size());
    dims[1] = 2ul;

    // We follow the path for
    HighFive::DataSet dataset =
      group.createDataSet<double>(dataset_name, HighFive::DataSpace(dims));

    // Each node want to write its own rank two time in
    // its associated row
    auto data = std::array<double, 2>{ par_env.rank() * 1.0, par_env.rank() * 2.0 };

    auto xfer_props = HighFive::DataTransferProps{};
    xfer_props.add(HighFive::UseCollectiveIO{});

    // Each MPI rank writes a non-overlapping part of the array.
    std::vector<size_t> offset{ std::size_t(par_env.rank()), 0ul };
    std::vector<size_t> count{ 1ul, 2ul };

    dataset.select(offset, count).write(data, xfer_props);
    check_collective_io(xfer_props);

    // Let's ensure that everything has been written do disk.
    file.flush();

    // We'd like to read back some data. For simplicity, we'll read the
    // row from the MPI above us (wrapping)
    offset[0] = (offset[0] + 1ul) % dims[0];

    // MPI ranks don't have to read non-overlapping parts, but in this
    // example they happen to. Again all rank participate in this call.
    dataset.select(offset, count).read(data, xfer_props);
    check_collective_io(xfer_props);
  }
  catch (HighFive::Exception & err)
  {
    // catch and print any HDF5 error
    std::cerr << err.what() << std::endl;
    MPI_Abort(par_env.mpi_comm(), 1);
  }

  return EXIT_SUCCESS;
}
