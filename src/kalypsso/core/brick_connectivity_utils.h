// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file brick_connectivity_utils.h
 */
#ifndef KALYPSSO_CORE_BRICK_CONNECTIVITY_UTILS_H_
#define KALYPSSO_CORE_BRICK_CONNECTIVITY_UTILS_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/brick_utils.h>
#include <kalypsso/utils/p4est/p4est_wrapper.h> // for SC_LOG2_32

#include <cstdint>

namespace kalypsso
{

/**
 * \class BrickLinearToXYZ
 *
 * \brief a utility class to convert a tree linear id number (morton tree id, as enumerated by
 * p4est) to x,y,z coordinates in the brick tree grid.
 *
 *
 */
template <size_t dim>
struct BrickLinearToXYZ
{
  Kokkos::Array<int, 3> m_brick_sizes;
  Kokkos::Array<int, 3> m_rankx;
  Kokkos::Array<int, 3> m_logx;
  int                   m_n_iter;

  BrickLinearToXYZ(brick_size_t<dim> brick_sizes)
  {
    m_brick_sizes[0] = brick_sizes[0];
    m_brick_sizes[1] = brick_sizes[1];
    if constexpr (dim == 3)
      m_brick_sizes[2] = brick_sizes[2];
    else
      m_brick_sizes[2] = 1;

    // compute log2 of each brick sizes, rounded to next integer
    // and compute the smallest embedding morton space containing the brick connectivity
    //
    // initializes m_rankx, and m_logx
    m_logx[0] = SC_LOG2_32(brick_sizes[0] - 1) + 1;
    m_logx[1] = SC_LOG2_32(brick_sizes[1] - 1) + 1;
    m_n_iter = (1 << m_logx[0]) * (1 << m_logx[1]);
    if (m_logx[0] <= m_logx[1])
    {
      m_rankx[0] = 0;
      m_rankx[1] = 1;
    }
    else
    {
      m_rankx[0] = 1;
      m_rankx[1] = 0;
    }

    if constexpr (dim == 3)
    {
      m_logx[2] = SC_LOG2_32(brick_sizes[2] - 1) + 1;
      m_n_iter *= (1 << m_logx[2]);
      if (m_logx[2] < m_logx[static_cast<size_t>(m_rankx[0])])
      {
        m_rankx[2] = m_rankx[1];
        m_rankx[1] = m_rankx[0];
        m_rankx[0] = 2;
      }
      else if (m_logx[static_cast<size_t>(m_rankx[1])] <= m_logx[2])
      {
        m_rankx[2] = 2;
      }
      else
      {
        m_rankx[2] = m_rankx[1];
        m_rankx[1] = 2;
      }
    }
  } // end BrickLinearToXYZ

  /**
   * convert a tree number into its x,y,z coordinates in the brick connectivity.
   *
   * adapted from the utility function \"brick_linear_to_xyz\" in p4est_connectivity.c
   *
   * \param[in] treeId is the tree id as used by p4est in the brick connectivity
   *
   * \return vector of cartesian coordinates of the tree in the brick grid
   */
  Kokkos::Array<uint16_t, dim>
  toXYZ(int treeId) const
  {
    Kokkos::Array<uint16_t, dim> tx;

    int i, j;
    int lastlog = 0;

    auto sdim = static_cast<int>(dim);

    for (size_t ii = 0; ii < dim; ii++)
    {
      tx[ii] = 0;
    }

    for (i = 0; i < sdim - 1; i++)
    {
      auto ii = static_cast<size_t>(i);

      p4est_topidx_t tempx[3] = { 0, 0, 0 };
      int            logi = m_logx[static_cast<size_t>(m_rankx[ii])] - lastlog;
      int            idx[3] = { -1, -1, -1 };
      int            c = 0;

      for (size_t k = 0; k < dim - ii; k++)
      {
        size_t d = static_cast<size_t>(m_rankx[ii + k]);

        idx[d] = 0;
      }
      for (size_t k = 0; k < dim; k++)
      {
        if (idx[k] == 0)
        {
          idx[k] = c++;
        }
      }

      for (j = 0; j < logi; j++)
      {
        int base = (sdim - i) * j;
        int shift = (sdim - i - 1) * j;

        for (size_t k = 0; k < dim; k++)
        {
          int id = idx[k];

          if (id >= 0)
          {
            tempx[k] |= (treeId & (1 << (base + id))) >> (shift + id);
          }
        }
      }
      for (size_t k = 0; k < dim; k++)
      {
        tx[k] += static_cast<uint16_t>(tempx[k] << lastlog);
      }
      lastlog += logi;
      treeId >>= (sdim - i) * logi;
    }
    tx[static_cast<size_t>(m_rankx[dim - 1])] += static_cast<uint16_t>(treeId << lastlog);

    return tx;

  } // end toXYZ

}; // BrickLinearToXYZ

/**
 * p4est brick connectivity geometrical data.
 *
 *
 */
template <size_t dim>
struct BrickConnectivityData
{
  BrickConnectivityData(brick_size_t<dim> brick_sizes)
    : m_brick_sizes(brick_sizes)
    , m_brick_coords("brick coords",
                     dim == 2
                       ? static_cast<size_t>(brick_sizes[0] * brick_sizes[1])
                       : static_cast<size_t>(brick_sizes[0] * brick_sizes[1] * brick_sizes[2]))
    , m_treeIds("treeIds",
                dim == 2 ? static_cast<size_t>(brick_sizes[0] * brick_sizes[1])
                         : static_cast<size_t>(brick_sizes[0] * brick_sizes[1] * brick_sizes[2]))
    , m_num_trees(dim == 2 ? brick_sizes[0] * brick_sizes[1]
                           : brick_sizes[0] * brick_sizes[1] * brick_sizes[2])

  {
    init();
  };

  /**
   * Brick connectivity data for device use.
   */
  template <typename device_t>
  struct Dev
  {
    Dev(brick_size_t<3> _brick_sizes)
      : brick_sizes(_brick_sizes)
      , brick_coords("brick coords", brick_sizes[0] * brick_sizes[1] * brick_sizes[2])
      , treeIds(Kokkos::view_alloc(Kokkos::WithoutInitializing, "treeIds"),
                brick_sizes[0] * brick_sizes[1] * brick_sizes[2]){};

    brick_size_t<3>                                         brick_sizes;
    Kokkos::View<int * [dim], Kokkos::LayoutLeft, device_t> brick_coords;
    Kokkos::View<int *, device_t>                           treeIds;

    KOKKOS_INLINE_FUNCTION
    int
    get_treeId(int i, int j)
    {
      return treeIds(i + brick_sizes[0] * j);
    }

    KOKKOS_INLINE_FUNCTION
    int
    get_treeId(int i, int j, int k)
    {
      return treeIds(i + brick_sizes[0] * j + brick_sizes[0] * brick_sizes[1] * k);
    }
  };

  /// brick sizes.
  brick_size_t<dim> m_brick_sizes;

  /// dynamic array containing xyz coordinates of each tree in the brick connectivity.
  /// this is a 2 dimensional array; first dimension span treeId (0 to nbTree-1) and
  /// second dimension is the coordinate id (x,y or z).
  Kokkos::View<int * [dim], Kokkos::LayoutLeft, Kokkos::HostSpace> m_brick_coords;

  /// contains the tree ids ordered by left layout on the brick connectivity
  Kokkos::View<int *, Kokkos::HostSpace> m_treeIds;

  /// total number of trees in the brick connectivity
  int m_num_trees;

  /// return linear tree id from morton tree id
  int
  treeId(int i, int j)
  {
    return m_treeIds(i + m_brick_sizes[0] * j);
  }

  /// return linear tree id from morton tree id
  int
  treeId(int i, int j, int k)
  {
    return m_treeIds(i + m_brick_sizes[0] * j + m_brick_sizes[0] * m_brick_sizes[1] * k);
  }

  /// compute xyz coordinates for each tree.
  /// this code is adapted from p4est library, p4est_connectivity_new_brick
  /// to make sure we use the same tree numbering
  void
  init()
  {
    // tree linear morton index to xyz converter
    BrickLinearToXYZ<dim> convert(m_brick_sizes);

    // max number of trees in the smallest embedding morton space (for which sizes are powers of 2)
    auto n_iter = convert.m_n_iter;

    int treeId = 0;

    for (int i = 0; i < n_iter; ++i)
    {
      auto tree_xyz = convert.toXYZ(i);

      bool valid_tree = [&] {
        if constexpr (dim == 2)
          return (tree_xyz[0] < m_brick_sizes[0]) and (tree_xyz[1] < m_brick_sizes[1]);
        else
          return (tree_xyz[0] < m_brick_sizes[0]) and (tree_xyz[1] < m_brick_sizes[1]) and
                 (tree_xyz[2] < m_brick_sizes[2]);
      }();


      // check if current treeId value is inside our brick domain
      if (valid_tree)
      {
        // now fill m_brick_coords
        m_brick_coords(treeId, 0) = tree_xyz[0];
        m_brick_coords(treeId, 1) = tree_xyz[1];

        if constexpr (dim == 3)
        {
          if (tree_xyz[2] < m_brick_sizes[2])
            m_brick_coords(treeId, 2) = tree_xyz[2];
          else
            m_brick_coords(treeId, 2) = 0;
        }

        // now fill m_treeIds
        if constexpr (dim == 2)
        {
          auto index = tree_xyz[0] + m_brick_sizes[0] * tree_xyz[1];
          m_treeIds[index] = treeId;
        }
        if constexpr (dim == 3)
        {
          auto index = tree_xyz[0] + m_brick_sizes[0] * tree_xyz[1] +
                       m_brick_sizes[0] * m_brick_sizes[1] * tree_xyz[2];
          m_treeIds[index] = treeId;
        }

        // if constexpr (dim == 3)
        //   printf("KKK %d | %d %d %d\n",
        //          treeId,
        //          m_brick_coords(treeId, 0),
        //          m_brick_coords(treeId, 1),
        //          m_brick_coords(treeId, 2));
        // else
        //   printf("KKK %d | %d %d\n", treeId, m_brick_coords(treeId, 0), m_brick_coords(treeId,
        //   1));

        ++treeId;
      }
    }

  } // init

  /**
   * convert a valid tree number into its x,y,z coordinates in the brick connectivity.
   */
  Kokkos::Array<uint16_t, dim>
  toXYZ(int treeId) const
  {
    Kokkos::Array<uint16_t, dim> coords;

    if (treeId < m_num_trees)
    {
      coords[0] = static_cast<uint16_t>(m_brick_coords(treeId, 0));
      coords[1] = static_cast<uint16_t>(m_brick_coords(treeId, 1));

      if constexpr (dim == 3)
        coords[2] = static_cast<uint16_t>(m_brick_coords(treeId, 2));
    }
    else
    {
      coords[0] = std::numeric_limits<uint16_t>::max();
      coords[1] = std::numeric_limits<uint16_t>::max();
      if constexpr (dim == 3)
        coords[2] = std::numeric_limits<uint16_t>::max();
    }

    return coords;
  } // toXYZ

  /// export brick connectivity data to be usable on device
  template <typename device_t>
  Dev<device_t>
  to_device()
  {
    Dev<device_t> dev_data(m_brick_sizes);

    Kokkos::deep_copy(dev_data.brick_coords, m_brick_coords);
    Kokkos::deep_copy(dev_data.treeIds, m_treeIds);

    return dev_data;
  } // to_device

}; // struct BrickConnectivityData

} // namespace kalypsso

#endif // KALYPSSO_CORE_BRICK_CONNECTIVITY_UTILS_H_
