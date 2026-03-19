// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ParallelEnv.cpp
 */
#include "ParallelEnv.h"

#include <kalypsso/core/kalypsso_core_config.h>

#include <cstring>
#include <iostream>

namespace kalypsso
{

// ===================================================================================
// ===================================================================================
void
set_p4est_log_priority_from_env()
{

  if ([[maybe_unused]] const char * env_value = std::getenv("P4EST_LOG_PRIORITY"))
  {

    printf("P4EST_LOG_PRIORITY set from environment to value %s\n", env_value);

    if (!std::strcmp(env_value, "SC_LP_ALWAYS"))
    {
      sc_package_set_verbosity(sc_package_id, SC_LP_ALWAYS);
      sc_package_set_verbosity(p4est_package_id, SC_LP_ALWAYS);
    }
    else if (!std::strcmp(env_value, "SC_LP_TRACE"))
    {
      sc_package_set_verbosity(sc_package_id, SC_LP_TRACE);
      sc_package_set_verbosity(p4est_package_id, SC_LP_TRACE);
    }
    else if (!std::strcmp(env_value, "SC_LP_INFO"))
    {
      sc_package_set_verbosity(sc_package_id, SC_LP_INFO);
      sc_package_set_verbosity(p4est_package_id, SC_LP_INFO);
    }
    else if (!std::strcmp(env_value, "SC_LP_STATISTICS"))
    {
      sc_package_set_verbosity(sc_package_id, SC_LP_STATISTICS);
      sc_package_set_verbosity(p4est_package_id, SC_LP_STATISTICS);
    }
    else if (!std::strcmp(env_value, "SC_LP_PRODUCTION"))
    {
      sc_package_set_verbosity(sc_package_id, SC_LP_PRODUCTION);
      sc_package_set_verbosity(p4est_package_id, SC_LP_PRODUCTION);
    }
    else if (!std::strcmp(env_value, "SC_LP_ESSENTIAL"))
    {
      sc_package_set_verbosity(sc_package_id, SC_LP_ESSENTIAL);
      sc_package_set_verbosity(p4est_package_id, SC_LP_ESSENTIAL);
    }
    else if (!std::strcmp(env_value, "SC_LP_ERROR"))
    {
      sc_package_set_verbosity(sc_package_id, SC_LP_ERROR);
      sc_package_set_verbosity(p4est_package_id, SC_LP_ERROR);
    }
    else if (!std::strcmp(env_value, "SC_LP_SILENT"))
    {
      sc_package_set_verbosity(sc_package_id, SC_LP_SILENT);
      sc_package_set_verbosity(p4est_package_id, SC_LP_SILENT);
    }
  }

} // set_p4est_log_priority_from_env

// ===================================================================================
// ===================================================================================
ParallelEnv::ParallelEnv(int & argc, char **& argv)
{
  bool m_initialize_kokkos_before_mpi = false;

#ifdef KOKKOS_ENABLE_CUDA
  // when running on platform with Intel OmniPath interconnect and Nvidia GPUs, we may
  // need to initialize Kokkos (and Cuda context first)
  if ([[maybe_unused]] const char * env_value = std::getenv("PSM2_CUDA"))
  {
    std::cout << "PSM2_CUDA detected : Initializing Kokkos before MPI" << std::endl;
    m_initialize_kokkos_before_mpi = true;
  }
#endif

  if ([[maybe_unused]] const char * env_value = std::getenv("KALYPSSO_INIT_KOKKOS_BEFORE_MPI"))
  {
    std::cout << "KALYPSSO_INIT_KOKKOS_BEFORE_MPI detected : Initializing Kokkos before MPI"
              << std::endl;
    m_initialize_kokkos_before_mpi = true;
  }

  if (m_initialize_kokkos_before_mpi)
    Kokkos::initialize(argc, argv);

#ifdef KALYPSSO_CORE_USE_MPI
  // Create MPI session if MPI enabled
  m_mpiSession = std::make_unique<GlobalMpiSession>(argc, argv);

  // create a communicator for MPI_COMM_WORLD
  m_comm_ptr = std::make_unique<MpiComm>();
#endif // KALYPSSO_CORE_USE_MPI

  if (!m_initialize_kokkos_before_mpi)
    Kokkos::initialize(argc, argv);

  print_kokkos_config();

  // initialize sc
#ifdef KALYPSSO_CORE_USE_MPI
  sc_init(m_comm_ptr->get_MPI_Comm(), 1, 1, nullptr, SC_LP_DEFAULT);
#else
  sc_init(sc_MPI_COMM_WORLD, 1, 1, nullptr, SC_LP_DEFAULT);
#endif // KALYPSSO_CORE_USE_MPI

  // initialize p4est
  p4est_init(nullptr, SC_LP_DEFAULT);

  set_p4est_log_priority_from_env();

} // ParallelEnv::ParallelEnv

#ifdef KALYPSSO_CORE_USE_MPI
// ===================================================================================
// ===================================================================================
ParallelEnv::ParallelEnv(int argc, char * argv[], const MPI_Comm & comm)
{
  // Create MPI session (check if MPI is already initialized, it should be)
  m_mpiSession = std::make_unique<GlobalMpiSession>(argc, argv);

  // create a communicator wrapping comm
  m_comm_ptr = std::make_unique<MpiComm>(comm, MpiComm::COMM_DUPLICATE);

  // initialize kokkos
  Kokkos::initialize(argc, argv);
  print_kokkos_config();

  // initialize p4est
  sc_init(m_comm_ptr->get_MPI_Comm(), 1, 1, nullptr, SC_LP_DEFAULT);
  p4est_init(nullptr, SC_LP_DEFAULT);

  set_p4est_log_priority_from_env();

} // ParallelEnv::ParallelEnv
#endif // KALYPSSO_CORE_USE_MPI

// ===================================================================================
// ===================================================================================
ParallelEnv::~ParallelEnv()
{
  // clean up p4est and sc then exit
  sc_finalize();

  // cleanup kokkos
  Kokkos::finalize();

} // ParallelEnv::~ParallelEnv

// ===================================================================================
// ===================================================================================
bool
ParallelEnv::MPI_enabled()
{
#ifdef KALYPSSO_CORE_USE_MPI
  return true;
#else  // KALYPSSO_CORE_USE_MPI
  return false;
#endif // KALYPSSO_CORE_USE_MPI
} // ParallelEnv::MPI_enabled

// ===================================================================================
// ===================================================================================
void
ParallelEnv::print_kokkos_config()
{
  // only master MPI task print Kokkos config information
  if (rank() == 0)
  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";
    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if (Kokkos::hwloc::available())
    {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
          << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
          << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";
    std::cout << "END KOKKOS CONFIG         \n";
    std::cout << "##########################\n";
  }

#ifdef KOKKOS_ENABLE_CUDA
  if ([[maybe_unused]] const char * env_value = std::getenv("CUDA_VISIBLE_DEVICES"))
  {
    std::cout << "I'm MPI task #" << this->rank() << " CUDA_VISIBLE_DEVICES was set to "
              << std::string(env_value) << "\n";
  }
  else
  {
    std::cout << "I'm MPI task #" << this->rank() << " CUDA_VISIBLE_DEVICES was not set" << "\n";
  }

  if (MPI_enabled())
  {

    // To enable kokkos accessing multiple GPUs don't forget to
    // add option "--ndevices=X" where X is the number of GPUs
    // you want to use per node.

    // on a large cluster, the scheduler should assign resources
    // in a way that each MPI task is mapped to a different GPU
    // let's cross-checked that:

    int cudaDeviceId;
    cudaGetDevice(&cudaDeviceId);
    std::cout << "I'm MPI task #" << this->rank() << " (out of " << this->nRanks() << ")"
              << " pinned to GPU #" << cudaDeviceId << "\n";
  }
#endif // KOKKOS_ENABLE_CUDA

} // ParallelEnv::print_kokkos_config

} // namespace kalypsso
