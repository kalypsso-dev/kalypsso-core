// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FourQuadrantParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_FOURQUADRANT_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_FOURQUADRANT_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/enums.h>

namespace kalypsso
{

namespace core
{

/**
 * Four-Quadrant problem test parameters.
 */
struct FourQuadrantParams
{

  Kokkos::Array<real_t, 3> pos;           //! Discontinuity locations
  int                      config_number; //! Config number

  FourQuadrantParams(ConfigMap const & config_map)
  {
    pos[0] = config_map.getReal("four_quadrant", "x", KALYPSSO_NUM(0.8));
    pos[1] = config_map.getReal("four_quadrant", "y", KALYPSSO_NUM(0.8));
    pos[2] = config_map.getReal("four_quadrant", "z", KALYPSSO_NUM(0.8));
    config_number = config_map.getInteger("four_quadrant", "config_number", 0);
  }

  /**
   * Get region id from a given location (2d version)
   *
   *
   * region labeling
   *
   *  1 | 0
   *  -----
   *  2 | 3
   */
  template <size_t dim>
  KOKKOS_INLINE_FUNCTION int
  get_region_id(Kokkos::Array<real_t, dim> const & xyz) const
  {
    int region_id = -1;

    if constexpr (dim == 2)
    {
      const real_t & x = xyz[IX];
      const real_t & y = xyz[IY];

      if (x < pos[IX])
      {
        if (y < pos[IY])
        {
          region_id = 2;
        }
        else
        {
          region_id = 1;
        }
      }
      else
      {
        if (y < pos[IY])
        {
          region_id = 3;
        }
        else
        {
          region_id = 0;
        }
      }
    }
    else if constexpr (dim == 3)
    {
      const real_t & x = xyz[IX];
      const real_t & y = xyz[IY];
      const real_t & z = xyz[IZ];

      if (x < pos[IX])
      {
        if (y < pos[IY])
        {
          if (z < pos[IZ])
          {
            region_id = 2;
          }
          else
          {
            region_id = 6;
          }
        }
        else
        {
          if (z < pos[IZ])
          {
            region_id = 1;
          }
          else
          {
            region_id = 5;
          }
        } // end y
      }
      else
      {
        if (y < pos[IY])
        {
          if (z < pos[IZ])
          {
            region_id = 3;
          }
          else
          {
            region_id = 7;
          }
        }
        else
        {
          if (z < pos[IZ])
          {
            region_id = 0;
          }
          else
          {
            region_id = 4;
          }
        } // end y
      } // end x
    } // end dim == 3

    return region_id;
  }

}; // struct FourQuadrantParameters

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_FOURQUADRANT_PARAMS_H_
