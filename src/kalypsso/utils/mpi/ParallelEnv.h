// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ParallelEnv.h
 * \brief Provide a class for initializing, finalizing environment for parallel
 * computation (MPI, Kokkos, p4est, ...)
 *
 * MPI is optional
 * Kokkos and p4est are mandatory
 *
 */
#ifndef KALYPSSO_UTILS_MPI_PARALLEL_ENV_H_
#define KALYPSSO_UTILS_MPI_PARALLEL_ENV_H_

#include <kalypsso/core/kalypsso_core_config.h>

#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/utils/mpi/GlobalMpiSession.h>
#  include <kalypsso/utils/mpi/MpiComm.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <cstdlib> // for std::getenv
#include <memory>  // for std::make_unique
#include <p4est.h> // for p4est_init / sc_init

namespace kalypsso
{

/**
 * If env variable P4EST_LOG_PRIORITY is defined, use it to modify p4est log priority level.
 *
 * \note if this env varriable is not defined, SC_LP_DEFAULT is used.
 */
void
set_p4est_log_priority_from_env();

// forward declaration
class ParallelEnv;

/**
 * \class ParallelEnvDevice
 *
 * A simplified version of ParallelEnv to have access to MPI rank and size on device (i.e. inside a
 * parallel for loop)
 */
class ParallelEnvDevice
{
public:
  ParallelEnvDevice(int rank_, int size_)
    : m_rank(rank_)
    , m_size(size_)
  {}

  ParallelEnvDevice(ParallelEnv const & par_env);

  ~ParallelEnvDevice() = default;

  KOKKOS_INLINE_FUNCTION
  auto
  rank() const
  {
    return m_rank;
  }

  KOKKOS_INLINE_FUNCTION
  auto
  size() const
  {
    return m_size;
  }

private:
  int m_rank;
  int m_size;
};

/**
 * \class ParallelEnv
 * Should initialize MPI, Kokkos and p4est.
 */
class ParallelEnv
{
private:
#ifdef KALYPSSO_CORE_USE_MPI
  std::unique_ptr<kalypsso::GlobalMpiSession> m_mpiSession;
  std::unique_ptr<kalypsso::MpiComm>          m_comm_ptr;
#endif // KALYPSSO_CORE_USE_MPI

public:
  /**
   * Default constructor.
   *
   * To be used in a standalone app.
   * MPI communicator will default to MPI_COMM_WORLD
   */
  ParallelEnv(int & argc, char **& argv);

#ifdef KALYPSSO_CORE_USE_MPI
  /**
   * Additional constructor to be used when kalypsso is used as library.
   *
   * The calling code should provide a MPI communicator.
   * Here we either attach to the provided MPI communicator or duplicate it.
   *
   * In this case, we take responsibility to initialize kokkos and p4est unconditionally.
   */
  ParallelEnv(int argc, char * argv[], const MPI_Comm & comm);
#endif // KALYPSSO_CORE_USE_MPI

  // Destructor.
  ~ParallelEnv();

  //! \return MPI rank
  inline int
  rank() const
  {
#ifdef KALYPSSO_CORE_USE_MPI
    return m_comm_ptr->rank();
#else
    return 0;
#endif // KALYPSSO_CORE_USE_MPI
  }

  //! \return MPI size
  inline int
  nRanks() const
  {
#ifdef KALYPSSO_CORE_USE_MPI
    return m_comm_ptr->size();
#else
    return 1;
#endif //  KALYPSSO_CORE_USE_MPI
  }

  //! \return MPI size
  inline int
  size() const
  {
#ifdef KALYPSSO_CORE_USE_MPI
    return m_comm_ptr->size();
#else
    return 1;
#endif // KALYPSSO_CORE_USE_MPI
  }

#ifdef KALYPSSO_CORE_USE_MPI

  //! \return MPI communicator (see MPIComm)
  const MpiComm &
  comm() const
  {
    return *m_comm_ptr;
  }

  //! return raw MPI communicator
  MPI_Comm
  mpi_comm() const
  {
    return m_comm_ptr->get_MPI_Comm();
  }
#endif // KALYPSSO_CORE_USE_MPI

  //! \return boolean to indicated if MPI is enabled
  static bool
  MPI_enabled();

  //! return a ParallelEnvDevice
  auto inline to_dev() { return ParallelEnvDevice(this->rank(), this->size()); }

private:
  //! print Kokkos configuration (backend enabled, etc...)
  void
  print_kokkos_config();

}; // class ParallelEnv

} // namespace kalypsso

#endif // KALYPSSO_UTILS_MPI_PARALLEL_ENV_H_
