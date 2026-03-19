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

// This is an example of how to read HDF5 files when all
// operations are collective, i.e. all MPI ranks participate in
// all HDF5 related function calls.
//
int
main(int argc, char ** argv)
{
  // initialize mpi, kokkos and p4est, use default MPI communicator : MPI_COMM_WORLD
  kalypsso::ParallelEnv par_env(argc, argv);

  try
  {
    //  MPI-IO requires informing HDF5 that we want something other than
    //  the default behaviour. This is done through property lists. We
    //  need a file access property list.
    auto fapl = HighFive::FileAccessProps();

    // We tell HDF5 to use MPI-IO
    fapl.add(HighFive::MPIOFileAccess{ par_env.mpi_comm(), MPI_INFO_NULL });

    // We also specify that we want all meta-data related operations
    // to use MPI collective operations. This implies that all MPI ranks
    // in the communicator must participate in any HDF5 operation that
    // reads or writes metadata. Essentially, this is safe if all MPI ranks
    // participate in all HDF5 operations.
    fapl.add(HighFive::MPIOCollectiveMetadata{});

    // Now we can create the file as usual.
    HighFive::File file(file_name, HighFive::File::ReadOnly, fapl);

    auto dataset = file.getDataSet("grp/dset");

    auto dims = dataset.getDimensions();

    if (par_env.rank() == 0)
    {
      std::cout << "dimensions of dataset [grp/dset]\n";
      for (size_t i = 0; i < dims.size(); ++i)
      {
        std::cout << "dataset dims[" << i << "] = " << dims[i] << "\n";
      }
    }

    // each MPI rank will read a different piece, split dims[0] among all MPI ranks
    // if division is not exact, the last rank receive a smaller amount of data
    std::size_t         pieceSize = (dims[0] + par_env.size() - 1) / par_env.size();
    std::vector<size_t> offset{ std::size_t(pieceSize * par_env.rank()), 0ul };
    std::vector<size_t> count{ pieceSize, 2ul };

    // the last rank maybe read a smaller amount of data
    if (par_env.rank() == par_env.size() - 1)
    {
      pieceSize = dims[0] - pieceSize * (par_env.size() - 1);
      count[0] = pieceSize;
    }

    // Each node want to read its own piece
    std::vector<double> data(dataset.getElementCount());

    auto xfer_props = HighFive::DataTransferProps{};
    xfer_props.add(HighFive::UseCollectiveIO{});

    dataset.select(offset, count).read_raw(data.data(), xfer_props);
    check_collective_io(xfer_props);

    // std::cout << "[rank=" << par_env.rank() << "] "
    //           << "count[0] = " << count[0] << "\n";

    for (size_t i = 0; i < count[0]; ++i)
    {
      for (size_t j = 0; j < count[1]; ++j)
      {
        printf("[rank=%d] data[%ld][%ld]=%f\n",
               par_env.rank(),
               i + offset[0],
               j,
               data[j + count[1] * i]);
      }
    }
  }
  catch (HighFive::Exception & err)
  {
    // catch and print any HDF5 error
    std::cerr << err.what() << std::endl;
    MPI_Abort(par_env.mpi_comm(), 1);
  }

  return EXIT_SUCCESS;
}
