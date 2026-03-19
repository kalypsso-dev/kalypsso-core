// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file SolverBase.cpp
 */
#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_MPI
#include <kalypsso/core/SolverBase.h>
#include <kalypsso/core/misc_utils.h>

#include <memory>
#include <sstream>

#ifdef KALYPSSO_CORE_USE_MPI
// #include "kalypsso/core/mpiBorderUtils.h"
#endif // KALYPSSO_CORE_USE_MPI

namespace kalypsso
{

// =======================================================
// ==== CLASS SolverBase IMPL ============================
// =======================================================

// =======================================================
// =======================================================
SolverBase::SolverBase(ParallelEnv const & par_env,
                       HydroParams const & params,
                       ConfigMap const &   config_map)
  : m_par_env(par_env)
  , m_params(params)
  , m_config_map(config_map)
  , m_profiling_mgr(par_env)
  , m_restart_run_enabled(false)
  , m_restart_run_filename("")
  , m_amr_cycle_counter(0)

{

  /*
   * init some variables by reading the ini parameter file.
   */
  read_config();

  /*
   * other variables initialization.
   */
  m_times_saved = 0;
  m_times_saved_restart = 0;

  m_nCells = -1;
  m_nDofsPerCell = -1;

  // statistics
  m_total_num_cell_updates = 0;

#ifdef KALYPSSO_CORE_USE_MPI
  // const int nbvar = m_params.nbvar;

  // TODO

#endif // KALYPSSO_CORE_USE_MPI

  // initialize memory allocation growth rate
  DataArrayUtils::set_growth_rate(config_map);

} // SolverBase::SolverBase

// =======================================================
// =======================================================
SolverBase::~SolverBase() {} // SolverBase::~SolverBase

// =======================================================
// =======================================================
void
SolverBase::do_amr_cycle()
{

  // Example of what must be implemented in derived class

  // 1. User data comm to update ghost cell values

  // 2. mark cell for refinement / coarsening

  // 3. adapt mesh

  // 4. user data remapping

  // increase number of AMR cycle
  m_amr_cycle_counter++;

} // SolverBase::do_amr_cycle

// =======================================================
// =======================================================
void
SolverBase::do_load_balancing()
{

  // perform MPI load balancing (mesh + user data)

} // SolverBase::do_load_balancing

// =======================================================
// =======================================================
void
SolverBase::read_config()
{

  // restart run : default is no
  m_restart_run_enabled = m_config_map.getInteger("run", "restart_enabled", 0);
  m_restart_run_filename = m_config_map.getString("run", "restart_filename", "");

  m_tBegin = m_config_map.getReal("run", "tBegin", KALYPSSO_NUM(0.0));
  m_tEnd = m_config_map.getReal("run", "tEnd", KALYPSSO_NUM(0.0));

  // note: when doing a restart run, m_t / m_tBegin will be read from the restart data file
  m_t = m_tBegin;

  m_max_iterations = m_params.nStepmax;

  // maximum number of output written
  m_max_save_count = m_params.nOutput;

  // save initial condition ?
  m_save_initial_condition = m_config_map.getBool("run", "save_initial_condition", true);

  // maximum number of checkpoint output written
  m_max_checkpoint_count = m_config_map.getInteger("checkpoint", "count", 1);

  m_amr_cycle_frequency = m_config_map.getInteger("amr", "cycle_frequency", 10);

  m_amr_load_balancing_frequency = m_config_map.getInteger("amr", "load_balancing_frequency", 20);

  m_dt = m_tEnd;
  m_cfl = m_config_map.getReal("hydro", "cfl", KALYPSSO_NUM(1.0));
  m_nlog = m_config_map.getInteger("run", "nlog", 10);
  m_iteration = 0;

  m_problem_name = m_config_map.getString("hydro", "problem", "unknown");

  m_solver_name = m_config_map.getString("run", "solver_name", "unknown");

} // SolverBase::read_config

// =======================================================
// =======================================================
void
SolverBase::compute_dt()
{

#ifdef KALYPSSO_CORE_USE_MPI

  // get local time step
  real_t dt_local = compute_dt_local();

  // perform MPI_Allreduce to get global time step
  real_t dt_global;
  m_par_env.comm().MPI_Allreduce<MpiComm::MIN>(&dt_local, &dt_global, 1);

  m_dt = dt_global;

#else

  m_dt = compute_dt_local();

#endif // KALYPSSO_CORE_USE_MPI

  // correct m_dt if necessary
  if (m_t + m_dt > m_tEnd)
  {
    m_dt = m_tEnd - m_t;
  }

} // SolverBase::compute_dt

// =======================================================
// =======================================================
real_t
SolverBase::compute_dt_local()
{

  // the actual numerical scheme must provide it a genuine implementation

  return m_tEnd;

} // SolverBase::compute_dt_local

// =======================================================
// =======================================================
int
SolverBase::finished()
{

  return m_t >= (m_tEnd - KALYPSSO_NUM(1e-14)) || m_iteration >= m_max_iterations;

} // SolverBase::finished

// =======================================================
// =======================================================
// TODO: better strategy to decide when to adapt ?
bool
SolverBase::should_do_amr_cycle()
{

  // default behavior : once every amr_cycle_frequency time steps
  const bool do_amr_cycle = m_params.amr_cycle_enabled and m_amr_cycle_frequency > 0 and
                            m_iteration > 0 and ((m_iteration % m_amr_cycle_frequency) == 0);

  const auto disable_first_amr_cycle = m_config_map.getBool("amr", "disable_first_amr_cycle", true);

  return do_amr_cycle or (m_iteration == 0 and disable_first_amr_cycle == false);

} // SolverBase::should_do_amr_cycle

// =======================================================
// =======================================================
// TODO: design a better strategy when print AMR info during a simulation run
bool
SolverBase::should_print_monitoring_info()
{
  const auto enable_amr_cycle_monitoring =
    m_config_map.getBool("amr", "amy_cycle_monitoring_enable", false);

  // strategy once every X amr cycle
  constexpr int default_value = 10;

  auto num_amr_cycles_monitoring =
    m_config_map.getInteger("amr", "num_amr_cycle_between_monitoring_print", default_value);

  // check for invalid values
  if (num_amr_cycles_monitoring < 1)
    num_amr_cycles_monitoring = default_value;

  return enable_amr_cycle_monitoring and (m_amr_cycle_counter % num_amr_cycles_monitoring == 0);

} // SolverBase::should_print_monitoring_info

// =======================================================
// =======================================================
bool
SolverBase::should_do_load_balancing()
{

  // default behavior : true once every amr_load_balancing_frequency
  return m_iteration != 0 and (m_iteration % m_amr_load_balancing_frequency) == 0;

} // SolverBase::should_do_load_balancing

// =======================================================
// =======================================================
void
SolverBase::next_iteration()
{

  // setup a timer here (?)

  // genuine implementation called here
  next_iteration_impl();

  // increment time
  ++m_iteration;
  m_t += m_dt;

} // SolverBase::next_iteration

// =======================================================
// =======================================================
void
SolverBase::next_iteration_impl()
{

  // This is application dependent

} // SolverBase::next_iteration_impl

// =======================================================
// =======================================================
void
SolverBase::run()
{

  /*
   * Default implementation for the time loop
   */
  while (!finished())
  {
    next_iteration();
  } // end time loop

} // SolverBase::run

// =======================================================
// =======================================================
void
SolverBase::save_solution(bool pure_checkpoint)
{

  // save solution to output file
  save_solution_impl(pure_checkpoint);

  // increment output file number
  // do not increment when dumping initial condition
  if (m_iteration > 0)
    ++m_times_saved;

} // SolverBase::save_solution

// =======================================================
// =======================================================
void
SolverBase::save_solution_impl([[maybe_unused]] bool pure_checkpoint)
{} // SolverBase::save_solution_impl

// =======================================================
// =======================================================
void
SolverBase::print_monitoring_info()
{} // SolverBase::print_monitoring_info

// =======================================================
// =======================================================
void
SolverBase::print_monitoring_info_final()
{} // SolverBase::print_monitoring_info_final

// =======================================================
// =======================================================
void
SolverBase::register_volume_integrals([[maybe_unused]] bool is_reference)
{} // SolverBase::register_volume_integrals

// =======================================================
// =======================================================
void
SolverBase::print_conservativity_check_report() const
{} // SolverBase::print_conservativity_check_report

// =======================================================
// =======================================================
bool
SolverBase::should_save_solution()
{

  real_t interval = (m_tEnd - m_tBegin) / static_cast<real_t>(m_params.nOutput);

  // m_params.nOutput == 0 means no output at all
  if (m_max_save_count == 0)
  {
    return false;
  }

  // m_params.nOutput < 0  means always output
  if (m_max_save_count < 0)
  {
    return true;
  }

  if (m_iteration == 0 and m_save_initial_condition)
  {
    return true;
  }
  else if ((m_t - m_tBegin) > (static_cast<real_t>(m_times_saved + 1) * interval))
  {
    return true;
  }

  // always write the last time step
  if (ISFUZZYNULL(m_t - m_tEnd))
  {
    return true;
  }

  return false;

} // SolverBase::should_save_solution

// =======================================================
// =======================================================
bool
SolverBase::should_do_checkpoint()
{

  // all outputs are turned into a checkpoint file
  if (m_config_map.getBool("checkpoint", "all_outputs_are_checkpoint", false))
  {
    return should_save_solution();
  }

  // doing checkpoint at regular intervals
  if (m_max_checkpoint_count > 0)
  {

    real_t interval = (m_tEnd - m_tBegin) / static_cast<real_t>(m_max_checkpoint_count);

    // never write restart file at t = m_tBegin
    if (((m_t - m_tBegin) - static_cast<real_t>(m_times_saved_restart) * interval) > interval)
    {
      return true;
    }

    // always write the restart at the last time step
    if (ISFUZZYNULL(m_t - m_tEnd))
    {
      return true;
    }
  }

  return false;

} // SolverBase::should_do_checkpoint

// =======================================================
// =======================================================
std::string
SolverBase::output_time_suffix()
{
  // prepare suffix string
  std::ostringstream strsuffix;
  strsuffix << "iter";
  strsuffix.width(7);
  strsuffix.fill('0');
  strsuffix << m_iteration;

  return strsuffix.str();

} // SolverBase::output_time_suffix

// =======================================================
// =======================================================
std::string
SolverBase::output_basename()
{
  std::string outputPrefix = m_config_map.getString("output", "outputPrefix", "output");

  return outputPrefix + "_" + output_time_suffix();

} // SolverBase::output_basename

} // namespace kalypsso
