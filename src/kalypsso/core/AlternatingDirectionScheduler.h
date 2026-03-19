// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/**
 * \file AlternatingDirectionScheduler.h
 */
#ifndef KALYPSSO_CORE_ALTERNATINGDIRECTIONSCHEDULER_H_
#define KALYPSSO_CORE_ALTERNATINGDIRECTIONSCHEDULER_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/enums.h>

#include <../better-enums/enum.h>

namespace kalypsso
{
// clang-format off
/**
 * An enum type to represent all implemented time integration methods.
 *
 * - NONE        : XY - XY - XY - ....
 * - PERMUTATION : XY - YX - XY - ....
 * - RUTH_ORDER3 : should provide third order convergence (when reachable)
 */
BETTER_ENUM(AlternatingDirectionType, uint8_t,
            NONE = 0,
            PERMUTATION = 1,
            RUTH_ORDER3 = 2)
// clang-format on

/**
 * Read configuration map to initialize the alternating direction type.
 */
AlternatingDirectionType
get_alternating_direction_type(ConfigMap const & config_map);

// ===================================================================================
// ===================================================================================
// ===================================================================================
/**
 * Helper class for applying numerical scheme using an alternating direction method.
 */
class AlternatingDirectionScheduler
{
public:
  using direction_view_t = Kokkos::View<ComponentIndex3D **, HostDevice>;
  using weight_view_t = Kokkos::View<real_t *, HostDevice>;

private:
  //! alternating direction type
  AlternatingDirectionType m_alternating_direction_type;

  //! dimension
  size_t m_dim;

  //! number of phases.
  //! e.g when doing permutations,
  //! in 2d there are 2 phases: XY and YZ
  //! in 3d there are 6 phases: XYZ, ZXY, YZX, ZYX, XZY and YXZ
  size_t m_num_phases;

  //! number of sweeps per phase.
  //! currently there are dim steps per phases (but there could be more)
  size_t m_num_sweeps;

  //! directions
  direction_view_t m_directions;

  //! time integration weights
  weight_view_t m_weights;

public:
  AlternatingDirectionScheduler(ConfigMap const & config_map, size_t dim);

  //! alternating direction type
  auto
  type() const
  {
    return m_alternating_direction_type;
  }

  //! get number of phases
  auto
  num_phases() const
  {
    return m_num_phases;
  }

  //! get number of sweeps per phase
  auto
  num_sweeps() const
  {
    return m_num_sweeps;
  }

  //! get direction
  ComponentIndex3D
  get_direction(size_t iteration, size_t sweep)
  {
    const auto phase_id = iteration % m_num_phases;
    return m_directions(phase_id, sweep);
  }

  //! get weight
  real_t
  get_weight(size_t sweep)
  {
    return m_weights(sweep);
  }

private:
  void
  init_directions();

  void
  init_weights();

}; // class AlternatingDirectionScheduler

} // namespace kalypsso

#endif // KALYPSSO_CORE_ALTERNATINGDIRECTIONSCHEDULER_H_
