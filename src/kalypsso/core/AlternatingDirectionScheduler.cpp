// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/**
 * \file AlternatingDirectionScheduler.cpp
 */
#include <kalypsso/core/AlternatingDirectionScheduler.h>

namespace kalypsso
{

// ==================================================================
// ==================================================================
AlternatingDirectionType
get_alternating_direction_type(ConfigMap const & config_map)
{
  const auto alternating_direction_type_name =
    config_map.getString("amr", "alternating_direction_type", "PERMUTATION");
  auto maybe_value =
    AlternatingDirectionType::_from_string_nothrow(alternating_direction_type_name.c_str());
  if (maybe_value)
  {
    return *maybe_value;
  }
  else
  {
    Kokkos::abort("Wrong parameter for amr/alternating_direction_type.");
  }
  return AlternatingDirectionType::NONE;
}


// ==================================================================
// ==================================================================

// ==================================================================
// ==================================================================
AlternatingDirectionScheduler::AlternatingDirectionScheduler(ConfigMap const & config_map,
                                                             size_t            dim)
  : m_alternating_direction_type(get_alternating_direction_type(config_map))
  , m_dim(dim)
  , m_num_phases(0)
  , m_num_sweeps(0)
  , m_directions()
  , m_weights()
{
  if (m_alternating_direction_type == +AlternatingDirectionType::NONE)
  {
    m_num_phases = 1;
    m_num_sweeps = dim;
  }
  else if (m_alternating_direction_type == +AlternatingDirectionType::PERMUTATION)
  {
    m_num_phases = dim == 2 ? 2 : 6;
    m_num_sweeps = dim;
  }
  else if (m_alternating_direction_type == +AlternatingDirectionType::RUTH_ORDER3)
  {
    m_num_phases = dim == 2 ? 2 : 6;
    m_num_sweeps = dim == 2 ? 6 : 21;
  }

  m_directions = direction_view_t("alternating directions", m_num_phases, m_num_sweeps);
  m_weights = weight_view_t("integration weights", m_num_sweeps);

  init_directions();
  init_weights();
}

// ==================================================================
// ==================================================================
void
AlternatingDirectionScheduler::init_directions()
{

  if (m_alternating_direction_type == +AlternatingDirectionType::NONE)
  {
    m_directions(0, 0) = IX;
    m_directions(0, 1) = IY;
    if (m_dim == 3)
      m_directions(0, 2) = IZ;
  }
  else if (m_alternating_direction_type == +AlternatingDirectionType::PERMUTATION)
  {
    if (m_dim == 2)
    {
      m_directions(0, 0) = IX;
      m_directions(0, 1) = IY;

      m_directions(1, 0) = IY;
      m_directions(1, 1) = IX;
    }
    else if (m_dim == 3)
    {
      m_directions(0, 0) = IX;
      m_directions(0, 1) = IY;
      m_directions(0, 2) = IZ;

      m_directions(1, 0) = IZ;
      m_directions(1, 1) = IY;
      m_directions(1, 2) = IX;

      m_directions(2, 0) = IY;
      m_directions(2, 1) = IZ;
      m_directions(2, 2) = IX;

      m_directions(3, 0) = IX;
      m_directions(3, 1) = IZ;
      m_directions(3, 2) = IY;

      m_directions(4, 0) = IZ;
      m_directions(4, 1) = IX;
      m_directions(4, 2) = IY;

      m_directions(5, 0) = IY;
      m_directions(5, 1) = IX;
      m_directions(5, 2) = IZ;
    }
  }
  else if (m_alternating_direction_type == +AlternatingDirectionType::RUTH_ORDER3)
  {
    if (m_dim == 2)
    {
      m_directions(0, 0) = IX;
      m_directions(0, 1) = IY;
      m_directions(0, 2) = IX;
      m_directions(0, 3) = IY;
      m_directions(0, 4) = IX;
      m_directions(0, 5) = IY;

      m_directions(1, 0) = IY;
      m_directions(1, 1) = IX;
      m_directions(1, 2) = IY;
      m_directions(1, 3) = IX;
      m_directions(1, 4) = IY;
      m_directions(1, 5) = IX;
    }
    else if (m_dim == 3)
    {
      ComponentIndex3D dir1, dir2, dir3;
      for (size_t i_phase = 0; i_phase < m_num_phases; ++i_phase)
      {
        if (i_phase == 0)
        {
          dir1 = IX;
          dir2 = IY;
          dir3 = IZ;
        }
        else if (i_phase == 1)
        {
          dir1 = IX;
          dir2 = IZ;
          dir3 = IY;
        }
        else if (i_phase == 2)
        {
          dir1 = IY;
          dir2 = IX;
          dir3 = IZ;
        }
        else if (i_phase == 3)
        {
          dir1 = IY;
          dir2 = IZ;
          dir3 = IX;
        }
        else if (i_phase == 4)
        {
          dir1 = IZ;
          dir2 = IX;
          dir3 = IY;
        }
        else if (i_phase == 5)
        {
          dir1 = IZ;
          dir2 = IY;
          dir3 = IX;
        }
        m_directions(i_phase, 0) = dir1;
        m_directions(i_phase, 1) = dir2;
        m_directions(i_phase, 2) = dir1;
        m_directions(i_phase, 3) = dir2;
        m_directions(i_phase, 4) = dir1;
        m_directions(i_phase, 5) = dir2;
        m_directions(i_phase, 6) = dir3;
        m_directions(i_phase, 7) = dir1;
        m_directions(i_phase, 8) = dir2;
        m_directions(i_phase, 9) = dir1;
        m_directions(i_phase, 10) = dir2;
        m_directions(i_phase, 11) = dir1;
        m_directions(i_phase, 12) = dir2;
        m_directions(i_phase, 13) = dir3;
        m_directions(i_phase, 14) = dir1;
        m_directions(i_phase, 15) = dir2;
        m_directions(i_phase, 16) = dir1;
        m_directions(i_phase, 17) = dir2;
        m_directions(i_phase, 18) = dir1;
        m_directions(i_phase, 19) = dir2;
        m_directions(i_phase, 20) = dir3;
      }

    } // end dim==3

  } // end RUTH_ORDER3

} // AlternatingDirectionScheduler::init_directions

// ==================================================================
// ==================================================================
void
AlternatingDirectionScheduler::init_weights()
{
  if (m_alternating_direction_type == +AlternatingDirectionType::NONE)
  {
    m_weights(0) = ONE_F;
    m_weights(1) = ONE_F;
    if (m_dim == 3)
      m_weights(2) = ONE_F;
  }
  else if (m_alternating_direction_type == +AlternatingDirectionType::PERMUTATION)
  {
    m_weights(0) = ONE_F;
    m_weights(1) = ONE_F;
    if (m_dim == 3)
      m_weights(2) = ONE_F;
  }
  else if (m_alternating_direction_type == +AlternatingDirectionType::RUTH_ORDER3)
  {
    constexpr real_t a1 = KALYPSSO_NUM(7.0) / KALYPSSO_NUM(24.0);
    constexpr real_t a2 = KALYPSSO_NUM(3.0) / KALYPSSO_NUM(4.0);
    constexpr real_t a3 = KALYPSSO_NUM(-1.0) / KALYPSSO_NUM(24.0);

    constexpr real_t b1 = KALYPSSO_NUM(2.0) / KALYPSSO_NUM(3.0);
    constexpr real_t b2 = KALYPSSO_NUM(-2.0) / KALYPSSO_NUM(3.0);
    constexpr real_t b3 = KALYPSSO_NUM(1.0);

    if (m_dim == 2)
    {
      m_weights(0) = a1;
      m_weights(1) = b1;
      m_weights(2) = a2;
      m_weights(3) = b2;
      m_weights(4) = a3;
      m_weights(5) = b3;
    }
    else if (m_dim == 3)
    {
      m_weights(0) = a1 * a1;
      m_weights(1) = a1 * b1;
      m_weights(2) = a1 * a2;
      m_weights(3) = a1 * b2;
      m_weights(4) = a1 * a3;
      m_weights(5) = a1 * b3;
      m_weights(6) = b1;
      m_weights(7) = a2 * a1;
      m_weights(8) = a2 * b1;
      m_weights(9) = a2 * a2;
      m_weights(10) = a2 * b2;
      m_weights(11) = a2 * a3;
      m_weights(12) = a2 * b3;
      m_weights(13) = b2;
      m_weights(14) = a3 * a1;
      m_weights(15) = a3 * b1;
      m_weights(16) = a3 * a2;
      m_weights(17) = a3 * b2;
      m_weights(18) = a3 * a3;
      m_weights(19) = a3 * b3;
      m_weights(20) = b3;
    }
  }
} // AlternatingDirectionScheduler::init_weights

} // namespace kalypsso
