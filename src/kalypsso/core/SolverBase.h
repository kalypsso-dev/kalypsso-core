// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file SolverBase.h
 */
#ifndef KALYPSSO_CORE_SOLVER_BASE_H_
#define KALYPSSO_CORE_SOLVER_BASE_H_

#include <map>
#include <memory> // for std::unique_ptr / std::shared_ptr

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_HDF5, ...
#include <kalypsso/core/HydroParams.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/utils/monitoring/ProfilingManager.h>
#include <kalypsso/utils/p4est/p4est_wrapper.h>

#ifdef KALYPSSO_CORE_USE_HDF5
#  include <kalypsso/core/HDF5_Xdmf_Writer.h>
#endif // KALYPSSO_CORE_USE_HDF5

namespace kalypsso
{

// ==========================================================================
// ==========================================================================
// ==========================================================================
/**
 * Abstract base class for all our actual solvers.
 */
class SolverBase
{
public:
  SolverBase(ParallelEnv const & par_env, HydroParams const & params, ConfigMap const & config_map);
  virtual ~SolverBase();

protected:
  //! parallel environment (MPI, Kokkos, p4est)
  ParallelEnv const & m_par_env;

  //! hydrodynamics parameters settings
  HydroParams const & m_params;

  //! unordered map of parameters read from input ini file
  ConfigMap const & m_config_map;

  //! Profiling manager
  ProfilingManager m_profiling_mgr;

  //! is this a restart run ?
  int m_restart_run_enabled;

  //! filename containing data from a previous run.
  std::string m_restart_run_filename;

  // iteration info
  real_t m_t;              //!< the time at the current iteration
  real_t m_dt;             //!< the time step at the current iteration
  int    m_iteration;      //!< the current iteration (integer)
  int    m_max_iterations; //!< user defined maximum iteration count
  real_t m_tBegin;         //!< begin time (might be non-zero upon restart)
  real_t m_tEnd;           //!< maximum time
  real_t m_cfl;            //!< Courant number
  int    m_nlog;           //!< number of steps between two monitoring print on screen

  int  m_max_save_count;         //!< max number of output written
  int  m_max_checkpoint_count;   //!< max number of checkpoint file written
  bool m_save_initial_condition; //!< true if you want to save initial condition

  //! control how often AMR cycle is done (how many time steps between two AMR cycles)
  int m_amr_cycle_frequency;

  //! number of AMR cycle done
  int m_amr_cycle_counter;

  /**
   * specify how often load balancing must be done.
   * Load balancing is performed once every amr_load_balancing_frequency
   * This value must be strictly larger than 0.
   */
  int m_amr_load_balancing_frequency;

  long long int m_nCells;       //!< number of cells
  long long int m_nDofsPerCell; //!< number of degrees of freedom per cell

  // statistics
  //! total number of quadrant update
  uint64_t m_total_num_cell_updates;

  //! init condition name (or problem)
  std::string m_problem_name;

  //! solver name (use in output file).
  std::string m_solver_name;

  //! dimension (2 or 3)
  int m_dim;

public:
  ConfigMap const &
  config_map() const
  {
    return m_config_map;
  }

  ParallelEnv const &
  par_env() const
  {
    return m_par_env;
  }

  HydroParams const &
  hydro_params() const
  {
    return m_params;
  }

  auto
  begin_time() const
  {
    return m_tBegin;
  }

  auto &
  begin_time()
  {
    return m_tBegin;
  }

  auto
  current_time() const
  {
    return m_t;
  }

  auto &
  current_time()
  {
    return m_t;
  }

  auto
  iteration() const
  {
    return m_iteration;
  }

  auto &
  iteration()
  {
    return m_iteration;
  }

  auto
  problem_name() const
  {
    return m_problem_name;
  }

  auto &
  profiling_mgr()
  {
    return m_profiling_mgr;
  }

  /*
   *
   * Computation interface that may be overridden in a derived
   * concrete implementation.
   *
   */
  //! solver name
  virtual std::string
  solver_name() const
  {
    return "unknown_solver";
  };

  //! perform AMR cycle (mark cells, adapt = refine/coarsen, remap user data)
  virtual void
  do_amr_cycle();

  //! do MPI load balancing
  virtual void
  do_load_balancing();

  //! Read and parse the configuration file (ini format).
  virtual void
  read_config();

  //! Compute CFL condition (allowed time step), over all MPI process.
  virtual void
  compute_dt();

  //! Compute CFL condition local to current MPI process
  virtual real_t
  compute_dt_local();

  //! Check if current time is larger than end time.
  virtual int
  finished();

  //! Check if AMR cycle is required
  virtual bool
  should_do_amr_cycle();

  //! Check if printing monitoring info is required
  virtual bool
  should_print_monitoring_info();

  //! Check if Load Balancing is required
  virtual bool
  should_do_load_balancing();

  //! This is where action takes place. Wrapper around next_iteration_impl.
  virtual void
  next_iteration();

  //! This is the next iteration computation (application specific).
  virtual void
  next_iteration_impl();

  //! This is were the time loop is done
  virtual void
  run();

  //! Decides if the current time step is eligible for saving data to file
  virtual bool
  should_save_solution();

  //! Decides if the current time step is eligible for checkpoint, ie.
  //! - saving p4est mesh state to file,
  //! - saving all required variables (e.g. conservative variables).
  //!
  //! This can be evaluated at any time step.
  //!
  //! If a regular output is also requested, i.e. should_save_solution() is true, then we make sure
  //! that all required variable will be save
  //! If a regular outpur is not requested, we are doing a "pure" checkpoint.
  //!
  //! The default behavior is to return false. The user has to explicitly request a checkpoint.
  //
  //! Note: By setting parameter checkpoint/all_outputs_are_checkpoint to true in the ini parameter
  //! file every output will be turned to a checkpoint file. default behavior is that only the last
  //! output is always a checkpoint.
  virtual bool
  should_do_checkpoint();

  //! return suffix for output files name containing time iteration format
  virtual std::string
  output_time_suffix();

  //! return base name for output files
  virtual std::string
  output_basename();

  //! main routine to dump solution (and additional derived quantities) to file.
  //!
  //! param[in] pure_checkpoint boolean indicating if the output should be considerate as a pure
  //! checkpoint file
  //!
  //! When doing a pure checkpoint, only the required field are dump to file
  virtual void
  save_solution(bool pure_checkpoint);

  //! main routine to dump solution to file.
  //! \sa save_solution
  virtual void
  save_solution_impl(bool pure_checkpoint);

  //! main routine to dump p4est mesh to file (for checkpoint / restart)
  //!
  //! \param[in] filename name of the p4est mesh file
  //! \param[in] forest p4est main data structure
  template <size_t dim_>
  void
  save_p4est_mesh(std::string filename, typename kalypsso::p4est::Wrapper<dim_>::forest_t * forest)
  {
    static const int save_data = 0;
    static const int save_partition = 0;

    if constexpr (dim_ == 2)
    {
      p4est::Wrapper<2>::save_ext(filename.c_str(), forest, save_data, save_partition);
    }
    else if constexpr (dim_ == 3)
    {
      p4est::Wrapper<3>::save_ext(filename.c_str(), forest, save_data, save_partition);
    }
  }

  //! print monitoring information
  virtual void
  print_monitoring_info();

  //! print monitoring information after final timep step
  virtual void
  print_monitoring_info_final();

  //! helper to register volume integrals
  virtual void
  register_volume_integrals(bool is_reference);

  //! print conservativity check report
  virtual void
  print_conservativity_check_report() const;


  /* IO related */

  //! counter incremented each time an output is written
  int m_times_saved;

  //! counter incremented each time a restart file is written
  int m_times_saved_restart;

}; // class SolverBase

} // namespace kalypsso

#endif // KALYPSSO_CORE_SOLVER_BASE_H_
