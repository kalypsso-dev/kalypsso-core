// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitialAMRSetup.h
 *
 * Define a common base class for doing actual tests.
 *
 * - predefine a p4est mesh with some initial refined mesh
 */

#ifndef KALYPSSO_TEST_AMRMESH_INITIALAMRSETUP_H_
#define KALYPSSO_TEST_AMRMESH_INITIALAMRSETUP_H_

#include <Kokkos_Macros.hpp>

#include <kalypsso/core/kalypsso_core_base.h> // for KALYPSSO_ASSERT
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h>
#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/enums.h>
#include <kalypsso/core/init_func.h> // for InitFunc1, InitFunc2, ...
#include <kalypsso/core/models/HydroState.h>

#include "test_func.h"

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/p4est/connectivity.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <cstdint> // for uint8_t, uint32_t, etc...
#include <string>

namespace kalypsso
{

// =============================================================================
// =============================================================================
struct InitialAMRSetupBase
{
  static int refine_level;
};

// =============================================================================
// =============================================================================
template <size_t dim>
int
refine_normal_fn(typename p4est::Wrapper<dim>::forest_t *   forest,
                 typename p4est::topidx_t                   which_tree,
                 typename p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  using Wrapper = typename p4est::Wrapper<dim>;

  using p4est_userdata_t = typename AMRmesh<dim>::p4est_userdata_t;

  p4est_userdata_t * p4est_userdata = static_cast<p4est_userdata_t *>(forest->user_pointer);
  [[maybe_unused]] const auto level_min = p4est_userdata->level_min;
  const auto                  level_max = p4est_userdata->level_max;

  // when reaching max level, do not refine anymore
  if (quadrant->level == level_max)
    return 0;

  if constexpr (dim == 3)
  {
    if (quadrant->level >= 1 &&
        (Wrapper::quadrant_child_id(quadrant) == 4 or Wrapper::quadrant_child_id(quadrant) == 6))
    {
      return 1;
    }
  }

  if constexpr (dim == 2)
  {
    if (which_tree == 0 and static_cast<int>(quadrant->level) < 4)
    {
      double x = static_cast<double>(p4est::get_x(quadrant)) / Wrapper::ROOT_LEN;
      double y = static_cast<double>(p4est::get_y(quadrant)) / Wrapper::ROOT_LEN;
      if (x > 0.25 and x < 0.35 and y > 0.85 and y < 0.95)
        return 1;
    }
  }

  if (which_tree == 1 and static_cast<int>(quadrant->level) < 4)
  {
    double x = static_cast<double>(p4est::get_x(quadrant)) / Wrapper::ROOT_LEN;
    double y = static_cast<double>(p4est::get_y(quadrant)) / Wrapper::ROOT_LEN;
    if (x > 0.45 and x < 0.55 and y > 0.45 and y < 0.55)
      return 1;
  }

  if (static_cast<int>(quadrant->level) >=
      (InitialAMRSetupBase::refine_level - static_cast<int>(which_tree) % 3))
  {
    return 0;
  }
  if (quadrant->level == 1 &&
      (Wrapper::quadrant_child_id(quadrant) == 3 or Wrapper::quadrant_child_id(quadrant) == 5))
  {
    return 1;
  }

  if constexpr (dim == 2)
  {
    if (quadrant->x == P4EST_LAST_OFFSET(2) && quadrant->y == P4EST_LAST_OFFSET(2))
    {
      return 1;
    }
  }
  else if constexpr (dim == 3)
  {
    if (quadrant->x == P4EST_LAST_OFFSET(2) && quadrant->y == P4EST_LAST_OFFSET(2) &&
        quadrant->z != P4EST_LAST_OFFSET(2))
    {
      return 1;
    }
  }

  if constexpr (dim == 2)
  {
    if (p4est::get_x(quadrant) >= static_cast<int>(Wrapper::QUADRANT_LEN(2)))
    {
      return 0;
    }
  }
  else
  {
    if (p4est::get_z(quadrant) >= static_cast<int>(Wrapper::QUADRANT_LEN(2)) or
        p4est::get_y(quadrant) <= static_cast<int>(Wrapper::QUADRANT_LEN(2)))
    {
      return 0;
    }
  }

  return 1;

} // refine_normal_fn

// =============================================================================
// =============================================================================
template <size_t dim>
int
refine_simple_fn(typename p4est::Wrapper<dim>::forest_t *   forest,
                 typename p4est::topidx_t                   which_tree,
                 typename p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  using Wrapper = typename p4est::Wrapper<dim>;

  using p4est_userdata_t = typename AMRmesh<dim>::p4est_userdata_t;

  p4est_userdata_t * p4est_userdata = static_cast<p4est_userdata_t *>(forest->user_pointer);
  [[maybe_unused]] const auto level_min = p4est_userdata->level_min;
  const auto                  level_max = p4est_userdata->level_max;

  // when reaching max level, do not refine anymore
  if (quadrant->level == level_max)
    return 0;

  if (which_tree == 0)
  {
    if (quadrant->level < 1)
      return 1;

    if (quadrant->level == 1)
    {
      if constexpr (dim == 2)
      {
        if ((p4est::get_x(quadrant) == Wrapper::ROOT_LEN / 2 and p4est::get_y(quadrant) == 0) or
            (p4est::get_y(quadrant) == Wrapper::ROOT_LEN / 2 and p4est::get_x(quadrant) == 0))
          return 1;
      }
      else if constexpr (dim == 3)
      {
        if ((p4est::get_x(quadrant) == Wrapper::ROOT_LEN / 2 and p4est::get_y(quadrant) == 0 and
             p4est::get_z(quadrant) == 0) or
            (p4est::get_y(quadrant) == Wrapper::ROOT_LEN / 2 and p4est::get_x(quadrant) == 0 and
             p4est::get_z(quadrant) == 0))
          return 1;
      }
    }
  }

  return 0;

} // refine_simple_fn

// =============================================================================
// =============================================================================
template <size_t dim>
int
norefine_fn([[maybe_unused]] typename p4est::Wrapper<dim>::forest_t *   forest,
            [[maybe_unused]] typename p4est::topidx_t                   which_tree,
            [[maybe_unused]] typename p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  return 0;
} // norefine_fn

// =============================================================================
// =============================================================================
template <size_t dim>
int
coarsen_normal_fn([[maybe_unused]] typename p4est::Wrapper<dim>::forest_t *   forest,
                  [[maybe_unused]] typename p4est::topidx_t                   which_tree,
                  [[maybe_unused]] typename p4est::Wrapper<dim>::quadrant_t * quadrant[])
{
  // currently, no coarsening

  return 0;
} // coarsen_normal_fn


// =============================================================================
// =============================================================================
template <size_t dim, typename device_t, typename Function>
class InitialAMRSetup : public InitialAMRSetupBase
{
public:
  using DataArrayLeaf_t = DataArrayLeaf<real_t, device_t>;
  // using DataArrayLeafHost_t = DataArrayLeafHost<real_t, device_t>;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  // using DataArrayBlockHost_t = DataArrayBlockHost<real_t, device_t>;

  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  using exec_space = typename device_t::execution_space;

  InitialAMRSetup() = delete;
  InitialAMRSetup(const ParallelEnv & par_env, const ConfigMap & config_map, const Function f);

  // =============================================================
  // =============================================================
  void
  setup_initial_mesh(bool norefine = false);

  // =============================================================
  // =============================================================
  auto
  setup_initial_data_leaf(typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device)
    -> DataArrayLeaf_t;

  // =============================================================
  // =============================================================
  auto
  setup_initial_data_block_new(
    typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device) -> DataArrayBlock_t;

  // =============================================================
  // =============================================================
  auto
  setup_initial_data_block_flux(
    typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device,
    int                                                 direction) -> DataArrayBlock_t;

  // =============================================================
  // =============================================================
  auto
  setup_initial_data_block_face(
    typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device,
    bool use_face_averaged_values) -> FaceDataArrayBlock_t;

  // =============================================================
  // =============================================================
  auto
  setup_initial_data_ghosted_block(
    typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device,
    bool                                                fill_ghosts) -> DataArrayGhostedBlock_t;

  // =============================================================
  // =============================================================
  auto
  compute_diff_ghosted_block(
    typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device,
    DataArrayGhostedBlock_t                             data1,
    DataArrayGhostedBlock_t                             data2) -> DataArrayGhostedBlock_t;

  AMRmesh<dim> &
  mesh()
  {
    return *m_amr_mesh;
  }

  auto
  mesh_map() const
  {
    return m_mesh_map;
  }

  brick_size_t<dim>
  brick_sizes() const
  {
    return m_brick_sizes;
  }

  // Kokkos::Array<uint8_t, dim>
  // brick_sizes_uint8() const
  // {
  //   if constexpr (dim == 2)
  //     return Kokkos::Array<uint8_t, 2>{ (uint8_t)m_brick_sizes[0], (uint8_t)m_brick_sizes[1] };
  //   else if constexpr (dim == 3)
  //     return Kokkos::Array<uint8_t, 3>{ (uint8_t)m_brick_sizes[0],
  //                                       (uint8_t)m_brick_sizes[1],
  //                                       (uint8_t)m_brick_sizes[2] };
  // }

  decltype(auto)
  is_brick_periodic() const
  {
    return m_is_brick_periodic;
  }

  decltype(auto)
  block_sizes() const
  {
    return m_block_sizes;
  }

  decltype(auto)
  ghost_sizes() const
  {
    return m_ghost_sizes;
  }

  decltype(auto)
  amr_mesh_info() const
  {
    return m_mesh_map->get_amr_mesh_info();
  }

private:
  //! parallel environment
  const ParallelEnv & m_par_env;

  //! config map (input parameter)
  ConfigMap m_config_map;

  std::shared_ptr<AMRmesh<dim>> m_amr_mesh;

  //! a MeshMap object so that we can extract the orchard keys as a kokkos view
  std::shared_ptr<MeshMap<dim, device_t>> m_mesh_map;

  //! p4est connectivity sizes.
  //! brick sizes are used to transform logical coordinates into vertex space (real space)
  // Kokkos::Array<int, 3> m_brick_sizes;
  brick_size_t<dim> m_brick_sizes;

  //! array of bool to tell if mesh is periodic or not
  Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! block sizes
  block_size_t<dim> m_block_sizes;

  //! block sizes
  block_size_t<dim> m_ghost_sizes;

  //! pointwise init functor
  const Function m_f;

}; // class InitialAMRSetup

extern template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitGaussian>;
extern template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitGaussian>;

extern template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitHat>;
extern template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitHat>;

extern template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitFuncParabola>;
extern template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitFuncParabola>;

extern template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitFuncSineWave>;
extern template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitFuncSineWave>;

extern template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitFunc1>;
extern template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitFunc1>;

extern template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitFunc2>;
extern template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitFunc2>;

extern template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitFunc3>;
extern template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitFunc3>;

// only instantiate those class when the default device is not on host
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) || \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS) ||  \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
#else
extern template class InitialAMRSetup<2, kalypsso::HostDevice, InitFuncSineWave>;
extern template class InitialAMRSetup<3, kalypsso::HostDevice, InitFuncSineWave>;
#endif

} // namespace kalypsso

#endif // KALYPSSO_TEST_AMRMESH_INITIALAMRSETUP_H_
