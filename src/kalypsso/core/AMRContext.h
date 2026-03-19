// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AMRContext.h
 *
 * Define data structure for applying AMR cycle (refine, coarsen, balance) using p4est API.
 */
#ifndef KALYPSSO_CORE_AMRCONTEXT_H_
#define KALYPSSO_CORE_AMRCONTEXT_H_

#include <kalypsso/core/kalypsso_core_base.h> // for KALYPSSO_ASSERT
#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/core/kokkos_shared.h>

#include <cstdint> // for uint8_t

namespace kalypsso
{

// forward declaration
template <size_t dim>
int
refine_callback(typename p4est::Wrapper<dim>::forest_t *   forest,
                p4est::topidx_t                            which_tree,
                typename p4est::Wrapper<dim>::quadrant_t * q);

template <size_t dim>
void
replace_on_refine_callback(typename p4est::Wrapper<dim>::forest_t *   forest,
                           p4est::topidx_t                            which_tree,
                           int                                        num_outgoing,
                           typename p4est::Wrapper<dim>::quadrant_t * outgoing[],
                           int                                        num_incoming,
                           typename p4est::Wrapper<dim>::quadrant_t * incoming[]);

template <size_t dim>
int
coarsen_callback(typename p4est::Wrapper<dim>::forest_t *             forest,
                 p4est::topidx_t                                      which_tree,
                 typename kalypsso::p4est::Wrapper<dim>::quadrant_t * q[]);

template <size_t dim>
void
replace_on_coarsen_callback(typename p4est::Wrapper<dim>::forest_t *   forest,
                            p4est::topidx_t                            which_tree,
                            int                                        num_outgoing,
                            typename p4est::Wrapper<dim>::quadrant_t * outgoing[],
                            int                                        num_incoming,
                            typename p4est::Wrapper<dim>::quadrant_t * incoming[]);

template <size_t dim>
void
replace_on_balance_callback(typename p4est::Wrapper<dim>::forest_t *   forest,
                            p4est::topidx_t                            which_tree,
                            int                                        num_outgoing,
                            typename p4est::Wrapper<dim>::quadrant_t * outgoing[],
                            int                                        num_incoming,
                            typename p4est::Wrapper<dim>::quadrant_t * incoming[]);


// =======================================================
// =======================================================
/**
 * A base class for AMRContext structure.
 */
struct AMRContextBase
{

  //! amr flags type
  using amrflag_t = int32_t;

  //!@{
  //! flags used in refine / coarsen callback functions
  static constexpr amrflag_t KALYPSSO_FLAG_INIT = -2;
  static constexpr amrflag_t KALYPSSO_DO_COARSEN = -1;
  static constexpr amrflag_t KALYPSSO_DO_NOTHING = 0;
  static constexpr amrflag_t KALYPSSO_DO_REFINE = 1;
  //!@}

}; // struct AMRContextBase

// =======================================================
// =======================================================
struct AMRCycleData
{
  using amrflag_t = AMRContextBase::amrflag_t;

  //! raw pointer to AMR flags for refine (member of AMRContext, allocated in AMRContext)
  //! must be allocated by AMRCycleData owner, size is p4est local number of quadrants
  amrflag_t * flags;

  //! raw pointer to AMR flags for coarsen (member of AMRContext, allocated in AMRContext)
  //! We strongly assumed coarsening happens AFTER refining.
  //! must be allocated by AMRCycleData owner, size is p4est local number of quadrant, pls
  //! additional newly added quadrants (from refining operation).
  amrflag_t * tmp_flags;

  /**
   * counter, determine the number of quadrants/octants to be refined.
   * It is incremented in mark_refine, and decremented in mark_coarsen (if quadrant/octant
   * has already been marked previously for refinement).
   */
  p4est::locidx_t num_refine_flags;

  /**
   * counter of visited quadrant/octant.
   *
   * incremented inside refine_callback (p4est_refine_ext), used for
   * cross-checking afterwards, that the callback was actually called
   * the right number of times.
   */
  p4est::locidx_t inside_counter;

  /**
   * actual number of coarsened quadrants; incremented inside coarsen_callback.
   */
  p4est::locidx_t num_replaced;

  /**
   * parameter that delays coarsening after adaptation.
   * \sa https://github.com/cburstedde/p4est/blob/master/src/p4est_wrap.h#L60
   */
  int coarsen_delay;

  /**
   * parameter that change the behavior of coarsening. Currently not used in kalypsso.
   * \sa https://github.com/cburstedde/p4est/blob/master/src/p4est_wrap.h#L64
   */
  int coarsen_affect = 0;

  AMRCycleData(amrflag_t * _flags, amrflag_t * _tmp_flags)
    : flags(_flags)
    , tmp_flags(_tmp_flags)
    , num_refine_flags(0)
    , inside_counter(0)
    , num_replaced(0)
    , coarsen_delay(0)
  {}

  void
  reset()
  {
    num_refine_flags = 0;
    inside_counter = 0;
    num_replaced = 0;
  } // reset

  void
  reset_internal()
  {
    inside_counter = 0;
    num_replaced = 0;
  } // reset_internal

  void
  set_coarsen_delay(int new_coarsen_delay)
  {
    coarsen_delay = new_coarsen_delay;
  }

}; // struct AMRCycleData

// ======================================================================================
// ======================================================================================
/**
 * A simple data structure holding minimal data needed to perform mesh refine/coarsen/balanced on
 * host using wrapper p4est API.
 *
 * Purpose:
 * - RAII for host and device AMR flags; the AMR flags are exposed to device; the user physics is
 *   responsible to compute the AMR flags (one per quadrant) to instruct p4est which
 *   quadrant must be refined, coarsened or stay untouched.
 * - the main member is adapt_mesh: use the AMR flags to commit the mesh changes; in other words,
 *   make a call to three p4est API : p4est_refine, p4est_coarsen, and p4est_balance.
 *   Just as a reminder refine and coarsen are purely local to current MPI process, while
 *   p4est_balance involves MPI communications (internal to p4est).
 *
 * \note
 * User data remapping (refine/coarsen/balance) will be done on device later. (TODO add here a link
 * to the code where this is done).
 *
 * This class is a close companion to ARMmesh (p4est RAII) and MeshMap (for converting AMR key into
 * a hash map).
 *
 * \TODO clarify if this must be a struct or a class
 */
template <size_t dim, typename device_t>
struct AMRContext : public AMRContextBase
{

  //! Kokkos execution space alias
  using exec_space = typename device_t::execution_space;

  //! type alias for a kokkos view holding amr flags on device
  using amrflags_view_t = typename Kokkos::View<amrflag_t *, device_t>;

  //! type alias for a kokkos view holding amr flags on host
  using amrflags_view_host_t = typename amrflags_view_t::HostMirror;

  using p4est_t = typename p4est::Wrapper<dim>;
  using forest_t = typename p4est_t::forest_t;


  // =======================================================
  /**
   * \param[in] local_num_octant local number of octant in current MPI process
   */
  AMRContext(int32_t local_num_octants)
    : m_amrflags_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "AMRContext::amrflags"),
                   static_cast<size_t>(local_num_octants))
    , m_amrflags_h(Kokkos::create_mirror_view(m_amrflags_d))
  {}

  // =======================================================
  //! resize amrflags view
  void
  resize_amrflags(int32_t new_size)
  {

    // resize (actually just reallocate memory without initializing)
    Kokkos::realloc(
      Kokkos::view_alloc(Kokkos::WithoutInitializing), m_amrflags_d, static_cast<size_t>(new_size));
    m_amrflags_h = Kokkos::create_mirror_view(m_amrflags_d);

  } // resize_amrflags

  // =======================================================
  //! reset amrflags view
  void
  reset_amrflags()
  {
    Kokkos::deep_copy(m_amrflags_d, AMRContextBase::KALYPSSO_FLAG_INIT);
  } // reset_amrflags

  // =======================================================
  /**
   * Apply refine / coarsen / 2:1 balance on the p4est mesh.
   *
   * Inside this routine, we assume the numerical scheme already filled the amrflags array using
   * physics specific criterion, and deep copying the flags on the host mirror is up to date.
   *
   * \note heavy data / physics remap is done elsewhere.
   *
   * \return boolean indicating if mesh has changed
   */
  bool
  adapt_mesh(typename p4est_t::forest_t * forest)
  {

    constexpr auto NB_CHILDREN = p4est_t::NB_CHILDREN;

    // store local number of quad before AMR cycle (will be used in assertions)
    [[maybe_unused]] p4est::locidx_t local_num_quadrants = forest->local_num_quadrants;

    // compute the number of possible refined quadrants (used to allocate temp_flags host array)
    int32_t              num_refine_flags = 0;
    Kokkos::Sum<int32_t> reducer(num_refine_flags);

    {
      // view copy to avoid "Implicit capture of 'this' in extended lambda expression"
      auto amrflags_d = m_amrflags_d;

      Kokkos::parallel_reduce(
        "Compute_num_refined_flags",
        Kokkos::RangePolicy<exec_space>(0, forest->local_num_quadrants),
        KOKKOS_LAMBDA(const int32_t index, int32_t & local_sum) {
          if (amrflags_d(index) == AMRContextBase::KALYPSSO_DO_REFINE)
            local_sum++;
        },
        reducer);
    }

    KALYPSSO_ASSERT(num_refine_flags >= 0 &&
                    num_refine_flags <= static_cast<int32_t>(forest->local_num_quadrants));

    // This allocation is optimistic when not all refine requests are honored
    amrflags_view_host_t temp_flags(
      "temp_flags",
      static_cast<size_t>(forest->local_num_quadrants + (NB_CHILDREN - 1) * num_refine_flags));
    Kokkos::deep_copy(temp_flags, AMRContextBase::KALYPSSO_DO_REFINE);

    // Create and initialize context data for running AMR cycle (p4est mesh adapt) on host
    AMRCycleData amr_cycle_data(m_amrflags_h.data(), temp_flags.data());
    amr_cycle_data.num_refine_flags = num_refine_flags;

    // change forest user_pointer
    void * tmp_user_pointer = forest->user_pointer;
    forest->user_pointer = static_cast<void *>(&amr_cycle_data);

    // store the number of quadrants before refinement
    p4est::gloidx_t old_global_num_quadrants = forest->global_num_quadrants;

    bool mesh_changed = false;

    //
    // Execute refinement
    //
    {
      // TODO : design a class containing the ScopedRegion + some optional internal monitoring
      // (Timer)
      Kokkos::Profiling::ScopedRegion myprof("AMR_MESH::refine");

      amr_cycle_data.reset_internal();

      // let p4est perform the actual refinement
      // - non-recursive
      // - allowed level is the maxlevel (P4EST_QMAXLEVEL)
      // - refine_callback is a function pointer (can't be nullptr), this is where the flag are
      // checked,
      //   the callback must return true if the quadrant is to be refined. When non-recursive, only
      //   existing quadrant are checked, when recursive is activated then all quadrant are checked,
      //   existing ones and newly created ones.
      // - init_fn will be non-nullptr only in the initial refinement (when computing the init
      // condition)
      // - replace_on_refine is specified in Solver::adapt to be
      //   wrap_replace_flags (initial condition) or quadrant_replace_fn (otherwise)
      //   note that wrap_replace_flags is defined here in SolverWrap
      //   while quadrant_replace_fn is define in Solver.
      //
      //   Please note that quadrant_replace_fn is used to change quadrant's user data
      //   when we will refactor for external memory storage (e.g. using Kokkos library)
      //   this operation will be done is a separate call.
      //

      constexpr int RECURSIVE_DISABLED = 0;
      constexpr int ALL_LEVEL_ALLOWED = -1; // amrflags MUST check we stay on valid level range

      p4est_t::refine_ext(forest,
                          RECURSIVE_DISABLED,
                          ALL_LEVEL_ALLOWED,
                          refine_callback<dim>,
                          nullptr,
                          replace_on_refine_callback<dim>);

      KALYPSSO_ASSERT(amr_cycle_data.inside_counter == local_num_quadrants);

      KALYPSSO_ASSERT(((forest->local_num_quadrants - local_num_quadrants) ==
                       amr_cycle_data.num_replaced * (NB_CHILDREN - 1)));


      mesh_changed = old_global_num_quadrants != forest->global_num_quadrants;
    }

    //
    // Execute coarsening
    //
    {
      Kokkos::Profiling::ScopedRegion myprof("AMR_MESH::coarsen");

      amr_cycle_data.reset_internal();

      // store the number of quadrants before coarsening
      old_global_num_quadrants = forest->global_num_quadrants;

      // only used in debug (for assertion)
      [[maybe_unused]] auto local_num_quadrants_before_coarsening = forest->local_num_quadrants;

      // let p4est perform the actual coarsening
      // - non-recursive
      // - callback_orphans=1 means  coarsen_callback is also called when siblings are not
      // "available" (i.e. ghost quadrants)
      // - coarsen callback is a function pointer (that can't be nullptr), this is WHERE action is,
      // i.e.
      //   where the flags of current quad and siblings are checked; if they all agree on
      //   coarsenning then they will disappear and be replaced by the coarse quadrant
      // - init_fn will be non-nullptr only in the initial refinement (when computing the init
      // condition)
      // - replace_on_coarsen is where quadrant's user_data is changed

      constexpr int RECURSIVE_DISABLED = 0;
      constexpr int CALLBACK_ORPHANS = 1;
      p4est_t::coarsen_ext(forest,
                           RECURSIVE_DISABLED,
                           CALLBACK_ORPHANS,
                           coarsen_callback<dim>,
                           nullptr,
                           amr_cycle_data.coarsen_delay ? replace_on_coarsen_callback<dim>
                                                        : nullptr);

      KALYPSSO_ASSERT(amr_cycle_data.inside_counter == local_num_quadrants_before_coarsening);
      KALYPSSO_ASSERT(local_num_quadrants_before_coarsening - forest->local_num_quadrants ==
                      amr_cycle_data.num_replaced * (NB_CHILDREN - 1));
      mesh_changed = mesh_changed || old_global_num_quadrants != forest->global_num_quadrants;
    }

    //
    // Execute 2:1 balance
    //

    // if mesh changed (refinement and/or coarsening actually happened), do also need to do 2:1
    // balance
    if (mesh_changed)
    {
      Kokkos::Profiling::ScopedRegion myprof("AMR_MESH::2to1_balance");

      p4est_t::balance_ext(forest,
                           p4est_t::CONNECT_FULL,
                           nullptr,
                           amr_cycle_data.coarsen_delay ? replace_on_balance_callback<dim>
                                                        : nullptr);
    }

    // restore forest user_pointer
    forest->user_pointer = tmp_user_pointer;

    return mesh_changed;

  } // adapt_mesh

  //! device amr flags.
  //! array of flags (one per local quadrant, dynamic allocation) to indicate a cell
  //! to refine or coarsen
  amrflags_view_t m_amrflags_d;

  //! host amr flags.
  //! host mirror of m_amrflags_h
  amrflags_view_host_t m_amrflags_h;

}; // struct AMRContext

// =======================================================
// =======================================================
/**
 * p4est refine callback used internally by p4est_refine_ext.
 */
template <size_t dim>
int
refine_callback(typename p4est::Wrapper<dim>::forest_t *                    forest,
                [[maybe_unused]] p4est::topidx_t                            which_tree,
                [[maybe_unused]] typename p4est::Wrapper<dim>::quadrant_t * q)
{

  using amrflag_t = AMRContextBase::amrflag_t;

  constexpr auto NB_CHILDREN = p4est::Wrapper<dim>::NB_CHILDREN;

  // get the AMR cycle internal data
  AMRCycleData * amr_cycle_data = static_cast<AMRCycleData *>(forest->user_pointer);

  const p4est::locidx_t old_counter = amr_cycle_data->inside_counter++;
  const amrflag_t       flag = amr_cycle_data->flags[old_counter];

  // check for invalid flag values
  KALYPSSO_ASSERT(flag >= -1 && flag <= 1);

  KALYPSSO_ASSERT(amr_cycle_data->coarsen_delay >= 0);
  KALYPSSO_ASSERT(0 <= old_counter);
  KALYPSSO_ASSERT(0 <= amr_cycle_data->num_replaced &&
                  amr_cycle_data->num_replaced <= amr_cycle_data->num_refine_flags);

  // refining WILL happen; the functor that created refine flags is responsible for checking
  // if refinement if possible (i.e. max level allowed not reached yet)
  amr_cycle_data->flags[old_counter] = AMRContextBase::KALYPSSO_DO_NOTHING;
  amr_cycle_data->tmp_flags[old_counter + (NB_CHILDREN - 1) * amr_cycle_data->num_replaced] =
    (flag == AMRContextBase::KALYPSSO_DO_REFINE) ? AMRContextBase::KALYPSSO_DO_NOTHING : flag;

  // increase quadrant's counter of most recent adaptation
  // if refinement actually occurs, it will be reset to zero in all children
  if (amr_cycle_data->coarsen_delay && q->p.user_int >= 0)
  {
    ++q->p.user_int;
  }

  return flag == AMRContextBase::KALYPSSO_DO_REFINE ? 1 : 0;

} // refine_callback

// =======================================================
// =======================================================
/**
 * p4est callback used when refinement actually happens.
 */
template <size_t dim>
void
replace_on_refine_callback(typename p4est::Wrapper<dim>::forest_t *                    forest,
                           [[maybe_unused]] p4est::topidx_t                            which_tree,
                           [[maybe_unused]] int                                        num_outgoing,
                           [[maybe_unused]] typename p4est::Wrapper<dim>::quadrant_t * outgoing[],
                           [[maybe_unused]] int                                        num_incoming,
                           [[maybe_unused]] typename p4est::Wrapper<dim>::quadrant_t * incoming[])
{

  constexpr auto NB_CHILDREN = p4est::Wrapper<dim>::NB_CHILDREN;

  // get the AMR cycle internal data
  AMRCycleData * amr_cycle_data = static_cast<AMRCycleData *>(forest->user_pointer);

  const p4est::locidx_t new_counter =
    amr_cycle_data->inside_counter - 1 + (NB_CHILDREN - 1) * amr_cycle_data->num_replaced++;

  const auto flag = amr_cycle_data->tmp_flags[new_counter];

  // this function is only called when refinement actually happens
  KALYPSSO_ASSERT(num_outgoing == 1 && num_incoming == NB_CHILDREN);
  KALYPSSO_ASSERT(flag == AMRContextBase::KALYPSSO_DO_NOTHING);

  // we have set the first flag in the refinement callback, do the others
  for (int k = 1; k < NB_CHILDREN; ++k)
  {
    amr_cycle_data->tmp_flags[new_counter + k] = flag;
  }

  // reset the counter for most recent adaptation
  KALYPSSO_ASSERT(amr_cycle_data->coarsen_delay >= 0);
  if (amr_cycle_data->coarsen_delay)
  {
    for (int k = 0; k < NB_CHILDREN; ++k)
    {
      incoming[k]->p.user_int = 0;
    }
  }

} // replace_on_refine_callback

// =======================================================
// =======================================================
/**
 * p4est coarsen callback used internally by p4est_coarsen_ext.
 */
template <size_t dim>
int
coarsen_callback(typename p4est::Wrapper<dim>::forest_t *                              forest,
                 [[maybe_unused]] p4est::topidx_t                                      which_tree,
                 [[maybe_unused]] typename kalypsso::p4est::Wrapper<dim>::quadrant_t * q[])
{
  constexpr auto NB_CHILDREN = p4est::Wrapper<dim>::NB_CHILDREN;

  // get the AMR cycle internal data
  AMRCycleData * amr_cycle_data = static_cast<AMRCycleData *>(forest->user_pointer);

  KALYPSSO_ASSERT(amr_cycle_data->coarsen_delay >= 0);

  const p4est::locidx_t old_counter = amr_cycle_data->inside_counter++;

  // are we not coarsening at all, just counting ? TO BE CLARIFIED IF REALLY NEEDED
  if (q[1] == nullptr)
  {
    return 0;
  }

  // now we are possibly coarsening, check that all quadrant in family agree on coarsening
  // if only one disagree, do not coarsen
  for (int k = 0; k < NB_CHILDREN; ++k)
  {
    if (amr_cycle_data->tmp_flags[old_counter + k] != AMRContextBase::KALYPSSO_DO_COARSEN)
    {
      return 0;
    }
    if (amr_cycle_data->coarsen_delay && q[k]->p.user_int >= 0 &&
        q[k]->p.user_int <= amr_cycle_data->coarsen_delay)
    {
      // most recent adaptation has been too recent, do not coarsen
      return 0;
    }
  }

  // we are definitely coarsening
  amr_cycle_data->inside_counter += NB_CHILDREN - 1;
  ++amr_cycle_data->num_replaced;
  return 1;

} // coarsen_callback

// =======================================================
// =======================================================
/**
 * p4est callback used when coarsening actually happens.
 */
template <size_t dim>
void
replace_on_coarsen_callback(typename p4est::Wrapper<dim>::forest_t * forest,
                            [[maybe_unused]] p4est::topidx_t         which_tree,
                            [[maybe_unused]] int                     num_outgoing,
                            [[maybe_unused]] typename p4est::Wrapper<dim>::quadrant_t * outgoing[],
                            [[maybe_unused]] int num_incoming,
                            [[maybe_unused]] typename p4est::Wrapper<dim>::quadrant_t * incoming[])
{
  // using amrflag_t = AMRContextBase::amrflag_t;
  [[maybe_unused]] constexpr auto NB_CHILDREN = p4est::Wrapper<dim>::NB_CHILDREN;

  // get the AMR cycle internal data
  [[maybe_unused]] AMRCycleData * amr_cycle_data =
    static_cast<AMRCycleData *>(forest->user_pointer);

  KALYPSSO_ASSERT(num_incoming == 1 && num_outgoing == NB_CHILDREN);
  KALYPSSO_ASSERT(amr_cycle_data->coarsen_delay > 0);

  // reset most recent adaptation timer
  incoming[0]->p.user_int = amr_cycle_data->coarsen_affect ? 0 : -1;

} // replace_on_coarsen_callback

// =======================================================
// =======================================================
/**
 * p4est callback used when restoring 2:1 level balance.
 */
template <size_t dim>
void
replace_on_balance_callback(typename p4est::Wrapper<dim>::forest_t * forest,
                            [[maybe_unused]] p4est::topidx_t         which_tree,
                            [[maybe_unused]] int                     num_outgoing,
                            [[maybe_unused]] typename p4est::Wrapper<dim>::quadrant_t * outgoing[],
                            [[maybe_unused]] int num_incoming,
                            [[maybe_unused]] typename p4est::Wrapper<dim>::quadrant_t * incoming[])
{

  [[maybe_unused]] constexpr auto NB_CHILDREN = p4est::Wrapper<dim>::NB_CHILDREN;

  // get the AMR cycle internal data
  [[maybe_unused]] AMRCycleData * amr_cycle_data =
    static_cast<AMRCycleData *>(forest->user_pointer);

  KALYPSSO_ASSERT(num_outgoing == 1 && num_incoming == NB_CHILDREN);
  KALYPSSO_ASSERT(amr_cycle_data->coarsen_delay > 0);

  // negative value means coarsening is allowed next time
  for (int k = 0; k < NB_CHILDREN; ++k)
  {
    incoming[k]->p.user_int = -1;
  }

} // replace_on_balance_callback

} // namespace kalypsso

#endif // KALYPSSO_CORE_AMRCONTEXT_H
