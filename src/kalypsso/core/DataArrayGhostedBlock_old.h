// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DataArrayGhostedBlock.h
 */
#ifndef KALYPSSO_CORE_DATAARRAYGHOSTEDBLOCK_OLD_H_
#define KALYPSSO_CORE_DATAARRAYGHOSTEDBLOCK_OLD_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>

#include <kalypsso/core/utils_block.h> // for definition of type block_size_t
#include <kalypsso/core/enums.h>
#include <kalypsso/core/utils_block.h> // for the definition of shift_t

#include <kalypsso/utils/log/kalypsso_log.h>

#include <kalypsso/core/DataArrayBlock.h>

namespace kalypsso
{

// ============================================================================================
// ============================================================================================
// ============================================================================================
/**
 * \class DataArrayGhostedBlock is data container similar to DataArrayBlock for storing user data
 * where each block contains ghost cells.
 *
 * In current implementation, the ghost zone is symmetric (durection by direction, it has the same
 * width on the left and on the right).
 *
 * \note that the ghost width on a given side can't be larger than block width divided by two. This
 * constraint is necessary for AMR: indeed when filling ghost cells, we need to access data from
 * neighbor block, and if ghost width where larger than block width over two, then we would need to
 * access data from neighbor block that are not direct neighbor but neighbor of neighbor which is
 * not currently possible with p4est, nor desirable (the neighbor of neighbor may has 2 AMR level
 * difference with current block level.)
 */
template <size_t dim, typename T, typename device_t>
class KALYPSSO_DEPRECATED_WITH_COMMENT(
  "This implementation is deprecated. Please refactor your code "
  "and use impl from header DataArrayGhostedBlock.h") DataArrayGhostedBlockOld
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, T, device_t>;

private:
  //! block size
  block_size_t<dim> m_bSize;

  //! ghost width
  block_size_t<dim> m_gWidth;

  //! total size (block size + 2 * ghost width)
  block_size_t<dim> m_tSize;

  //! storage array
  DataArrayBlock_t m_data;

public:
  DataArrayGhostedBlockOld() = default;

  DataArrayGhostedBlockOld(block_size_t<dim> bSize,
                           block_size_t<dim> gWidth,
                           std::string       name,
                           int               num_vars,
                           int               num_quadrants)
    : m_bSize(bSize)
    , m_gWidth(gWidth)
    , m_tSize(bSize + 2 * gWidth)
    , m_data(name, m_tSize, num_vars, num_quadrants)
  {
    KOKKOS_ASSERT((gWidth[IX] <= bSize[IX] / 2) && "Ghost width is too large");
    KOKKOS_ASSERT((gWidth[IY] <= bSize[IY] / 2) && "Ghost width is too large");
    if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((gWidth[IZ] <= bSize[IZ] / 2) && "Ghost width is too large");
    }
  }

  auto
  flat_view()
  {
    return m_data.storage();
  }

  auto &
  flat_view_ref()
  {
    return m_data.storage();
  }

  auto
  data()
  {
    return m_data;
  }

  auto &
  data_ref()
  {
    return m_data;
  }

  // ==================================================================================
  //! memory access operator() as if dealing with a regular Kokkos::View - 2d case
  template <typename I0,
            typename I1,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, IVar ivar, IOct iOct) const
  {
    return m_data(i0, i1, ivar, iOct);
  }

  // ==================================================================================
  //! memory access operator() as if dealing with a regular Kokkos::View - 3d case
  template <typename I0,
            typename I1,
            typename I2,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, I2 i2, IVar ivar, IOct iOct) const
  {
    return m_data(i0, i1, i2, ivar, iOct);
  }

  // ==================================================================================
  //! memory access operator() using a flat cell index (2d and 3d)
  template <typename ICell, typename IVar, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(ICell icell, IVar ivar, IOct iOct) const
  {
    return m_data(icell, ivar, iOct);
  }

  // ==================================================================================
  auto
  label() const
  {
    return m_data.label();
  }

  //! Reallocate by changing the number of quadrants/octants (do not initialize memory).
  void
  resize(int32_t num_quads)
  {
    m_data.resize(num_quads);
  }

  //! Reallocate by changing the number of quadrants/octants and reset memory.
  void
  resize_and_reset(int32_t num_quads)
  {
    m_data.resize_and_reset(num_quads);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_cells() const
  {
    return m_data.num_cells();
  }

  //! Return total number of cells of the original non-ghosted block.
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_cells_inner() const
  {
    return Kokkos::dim_prod(m_bSize);
  }

  //! Return the number of scalar variables.
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_vars() const
  {
    return m_data.num_vars();
  }

  //! Return the number of quadrants/octants.
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_quadrants() const
  {
    return m_data.num_quadrants();
  }

  //! Return the block size (shape) of the non-ghosted block.
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  block_size() const
  {
    return m_bSize;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  ghost_size() const
  {
    return m_gWidth;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  total_block_size() const
  {
    return m_tSize;
  }

  //! Return the total allocated memory in bytes.
  auto
  allocated_size_in_bytes() const
  {
    uint64_t size =
      static_cast<uint64_t>(m_data.num_cells() * m_data.num_vars() * m_data.num_quadrants()) *
      sizeof(T);
    return size;
  }

}; // DataArrayGhostedBlockOld

} // namespace kalypsso

#endif // KALYPSSO_CORE_DATAARRAYBLOCK_OLD_H_
