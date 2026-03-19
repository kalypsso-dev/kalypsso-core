// Copyright (2014) Sandia Corporation
// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <sstream>

#include <kalypsso/core/kalypsso_core_config.h>
#include <Kokkos_Macros.hpp>

#if defined(KALYPSSO_CORE_USE_MPI)
#  include <mpi.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <Kokkos_Core.hpp>

#ifndef UNUSED
#  define UNUSED(x) ((void)(x))
#endif

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

int
main(int argc, char ** argv)
{

  UNUSED(argc);
  UNUSED(argv);

  std::ostringstream msg;

  [[maybe_unused]] int mpi_rank = 0;
  [[maybe_unused]] int nRanks = 1;

#if defined(KALYPSSO_CORE_USE_MPI)

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

  msg << "MPI rank(" << mpi_rank << ") ";

#endif // KALYPSSO_CORE_USE_MPI

  Kokkos::initialize(argc, argv);

#ifdef KOKKOS_ENABLE_CUDA
  {

    // // get device count
    // int devCount;
    // cudaGetDeviceCount(&devCount);

    // int devId = mpi_rank % devCount;
    // cudaSetDevice(devId);

    // To enable kokkos accessing multiple GPUs don't forget to
    // add option "--ndevices=X" where X is the number of GPUs
    // you want to use per node.

    // on a large cluster, the scheduler should assign resources
    // in a way that each MPI task is mapped to a different GPU
    // let's cross-checked that:

    int cudaDeviceId;
    cudaGetDevice(&cudaDeviceId);
    std::cout << "I'm MPI task #" << mpi_rank << " (out of " << nRanks << ")" << " pinned to GPU #"
              << cudaDeviceId << "\n";
  }
#endif // KOKKOS_ENABLE_CUDA

  msg << "{" << std::endl;

  if (Kokkos::hwloc::available())
  {
    msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
        << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
        << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
  }

  Kokkos::print_configuration(msg);

  msg << "}" << std::endl;

  std::cout << msg.str();

  Kokkos::finalize();

#if defined(KALYPSSO_CORE_USE_MPI)

  MPI_Finalize();

#endif

  return 0;
}
