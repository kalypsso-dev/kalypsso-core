// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DataArrayGhostedBlock.h
 */
#ifndef KALYPSSO_CORE_DATAARRAYGHOSTEDBLOCK_H_
#define KALYPSSO_CORE_DATAARRAYGHOSTEDBLOCK_H_

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
 * where each block may contain ghost cells.
 *
 * For each original block of cells from a DataArrayBlock, we associate a new block of cells (that
 * may be larger, same size, smaller than the original block, and shifted in space). For example,
 * when the shift is negative along the x axis, it means the shifted block of cells will overlap
 * with a neighbor block, and we will need to either copy, prolongate or restrict data from the
 * neighbor block. This copy/interpolation operation is the job of class FillBlockGhostCells.
 *
 * Here we only design a container that we ease the job of FillBlockGhostCells. That is why we store
 * the block size (of the original block of cells, the one use in our AMR mesh), and also the block
 * size of the shifted block, as well as the vector shift. One important constraint we enforce here
 * is that the shift (space displacement) can't larger than the block size divided by two in each
 * direction. This constraint will ensure that we will be able to fill ghost cells using data from
 * the adjacent block cells, where the worst case is when the neighbor block is finer than current
 * one.
 *
 * A typical practical situation is when one want to apply a stencil operator to a DataArrayBlock,
 * e.g. compute a centered estimation of a gradient: \f$ \partial_x f \simeq
 * \frac{f(i-1,j)-f(i+1,j)}{2\delta_x} \f$. When visiting a cell at block border, one needs data
 * from a cell that belongs to a neighbor block which may live at finer, same or coarser AMR level.
 * Usually the stencil size (data access pattern) is known in advance (by design), so here we
 * provide, as a helper class, a data container where all block (associated to an octant) may be
 * equipped with ghost cells; ghost cells overlap neighbor blocks, so ghost cells data are filled
 * with either a prolongation, restriction or copy operator.
 *
 * Simple example where the ghosted block is made with one ghost cell all around.
 *
 *    ________________________
 *   |   |   |   |   |   |   |
 *   |___|___|___|___|___|___|
 *   |   | x | x | x | x |   |
 *   |___|___|___|___|___|___|
 *   |   | x | x | x | x |   |
 *   |___|___|___|___|___|___|
 *   |   | x | x | x | x |   |
 *   |___|___|___|___|___|___|
 *   |   |x0 | x | x | x |   |
 *   |___|___|___|___|___|___|
 *   | X0|   |   |   |   |   |
 *   |___|___|___|___|___|___|
 *
 * In the more general case, we may need to have only ghost cells along the x-axis, or the y-axis.
 * The ghosted block is seen as a block of cells that has a different block size compared to the
 * original non-ghosted block size, and that is shifted in space.
 *
 * So this class is designed so that the user specifies:
 * - a space shift (in number of cell units), it may be positive or negative
 * - a block size: it may be smaller of larger than the original block size
 *
 * We provide two ways to access cell data:
 * - either by using signed integers coordinates that are relative to the lower left cell of the
 * original non-ghosted block (see cell marked with label "x0" in the drawing above)
 * - either by using positive integers coordinates that relative to the lower left cell of the
 * ghosted block (see label "X0"); this is called "direct access" in the class implementation.
 *
 * \note the ghost width on a given side can't be larger than block width divided by two. This
 * constraint is necessary for AMR: indeed when filling ghost cells, we need to access data from
 * neighbor block, and if ghost width where larger than block width over two, then we would need to
 * access data from neighbor block that are not direct neighbor but neighbor of neighbor which is
 * not currently possible with p4est, nor desirable (the neighbor of neighbor may has 2 AMR level
 * difference with current block level.)
 */
template <size_t dim, typename T, typename device_t>
class DataArrayGhostedBlock
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, T, device_t>;

  //! tag dispatch class to use when using non-signed index coordinates (i.e. relative to the
  //! shifted block origin); when using direct access, valid index must be in the following ranges:
  //! i in [0, m_gSize[IX]-1]
  //! j in [0, m_gSize[IY]-1]
  //! k in [0, m_gSize[IZ]-1]
  //!
  //! \note if you reshape a DataArrayGhostedBlock, valid index are constraint by m_data.shape()
  struct DirectAccess
  {};

private:
  //! non-ghosted block size
  block_size_t<dim> m_bSize;

  //! space shift
  shift_t<dim> m_shift;

  //! storage array
  DataArrayBlock_t m_data;

  //! overlapping block size : sizes of the zones where the shifted block and the original block
  //! overlaps
  block_size_t<dim> m_bSize_overlap;

public:
  DataArrayGhostedBlock() = default;

  //! DataArrayGhostedBlock constructor.
  //!
  //! \param[in] bSize is the non-ghosted block size.
  //! \param[in] gSize is the size of the ghosted block (used for memory allocation)
  //! \param[in] shift is the vector of shift (in unit signed number of cell) to locate the origin
  //! of the shifted ghosted block
  //! \param[in] name is the ghosted block label
  //! \param[in] num_vars is the number of scalar variables
  //! \param[in] num_quadrants is the number of quadrants(2d) or octants(3d)
  DataArrayGhostedBlock(block_size_t<dim> bSize,
                        block_size_t<dim> gSize,
                        shift_t<dim>      shift,
                        std::string       name,
                        int               num_vars,
                        int               num_quadrants)
    : m_bSize(bSize)
    , m_shift(shift)
    , m_data(name, gSize, num_vars, num_quadrants)
    , m_bSize_overlap(get_block_size_overlap(gSize, shift))
  {
    KOKKOS_ASSERT((shift[IX] + bSize[IX] / 2 >= 0) && "shift[IX] is too large");
    KOKKOS_ASSERT((shift[IY] + bSize[IY] / 2 >= 0) && "shift[IY] is too large");
    if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((shift[IZ] + bSize[IZ] / 2 >= 0) && "shift[IZ] is too large");
    }

    KOKKOS_ASSERT((m_data.block_size()[IX] <= 2 * bSize[IX]) &&
                  "m_data.block_size()[IX] is too large");
    KOKKOS_ASSERT((m_data.block_size()[IY] <= 2 * bSize[IY]) &&
                  "m_data.block_size()[IY] is too large");
    if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((m_data.block_size()[IZ] <= 2 * bSize[IZ]) &&
                    "m_data.block_size()[IZ] is too large");
    }
  }

  auto
  flat_view()
  {
    return m_data.logical_view();
  }

  auto &
  flat_view_ref()
  {
    return m_data.logical_view();
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
  /**
   * Default memory access operator() - 2d case.
   *
   * \param[in] i0 cell coordinate inside block/octant along X axis
   * \param[in] i1 cell coordinate inside block/octant along Y axis
   * \param[in] ivar variable index
   * \param[in] iOct octant/block index
   *
   * \note index i0,i1,i2 must be coordinates relative to the non-ghosted block (signed values)
   */
  template <typename I0,
            typename I1,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, IVar ivar, IOct iOct) const
  {
    KOKKOS_ASSERT((i0 - m_shift[IX] >= 0) && "Invalid index value");
    KOKKOS_ASSERT((i1 - m_shift[IY] >= 0) && "Invalid index value");
    KOKKOS_ASSERT((i0 - m_shift[IX] < static_cast<int>(m_data.shape()[IX])) &&
                  "Invalid index value");
    KOKKOS_ASSERT((i1 - m_shift[IY] < static_cast<int>(m_data.shape()[IY])) &&
                  "Invalid index value");

    KOKKOS_ASSERT((ivar >= 0) && "invalid ivar index value");
    KOKKOS_ASSERT((ivar < m_data.num_vars()) && "invalid ivar index value");

    KOKKOS_ASSERT((iOct >= 0) && "invalid iOct index value");
    KOKKOS_ASSERT((iOct < m_data.num_quadrants()) && "invalid iOct index value");

    return m_data(i0 - m_shift[IX], i1 - m_shift[IY], ivar, iOct);
  }

  // ==================================================================================
  /**
   * Direct memory access operator() - 2d case.
   *
   * \param[in] i0 cell coordinate inside block/octant along X axis
   * \param[in] i1 cell coordinate inside block/octant along Y axis
   * \param[in] ivar variable index
   * \param[in] iOct octant/block index
   *
   * \note index i0,i1,i2 must be coordinates relative to the ghosted block (positive values)
   */
  template <typename I0,
            typename I1,
            typename IVar,
            typename IOct,
            int dim_ = dim,
            std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, IVar ivar, IOct iOct, DirectAccess const &) const
  {
    KOKKOS_ASSERT((i0 >= 0) && "Invalid index value");
    KOKKOS_ASSERT((i1 >= 0) && "Invalid index value");
    KOKKOS_ASSERT((i0 < m_data.shape()[IX]) && "Invalid index value");
    KOKKOS_ASSERT((i1 < m_data.shape()[IY]) && "Invalid index value");

    KOKKOS_ASSERT((ivar >= 0) && "invalid ivar index value");
    KOKKOS_ASSERT((ivar < m_data.num_vars()) && "invalid ivar index value");

    KOKKOS_ASSERT((iOct >= 0) && "invalid iOct index value");
    KOKKOS_ASSERT((iOct < m_data.num_quadrants()) && "invalid iOct index value");

    return m_data(i0, i1, ivar, iOct);
  }

  // ==================================================================================
  /**
   * Default memory access operator() - 3d case.
   *
   * \param[in] i0 cell coordinate inside block/octant along X axis
   * \param[in] i1 cell coordinate inside block/octant along Y axis
   * \param[in] i2 cell coordinate inside block/octant along Z axis
   * \param[in] ivar variable index
   * \param[in] iOct octant/block index
   *
   * \note index i0,i1,i2 must be coordinates relative to the non-ghosted block (signed values)
   */
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
    KOKKOS_ASSERT((i0 - m_shift[IX] >= 0) && "Invalid index value");
    KOKKOS_ASSERT((i1 - m_shift[IY] >= 0) && "Invalid index value");
    KOKKOS_ASSERT((i2 - m_shift[IZ] >= 0) && "Invalid index value");
    KOKKOS_ASSERT((i0 - m_shift[IX] < m_data.shape()[IX]) && "Invalid index value");
    KOKKOS_ASSERT((i1 - m_shift[IY] < m_data.shape()[IY]) && "Invalid index value");
    KOKKOS_ASSERT((i2 - m_shift[IZ] < m_data.shape()[IZ]) && "Invalid index value");

    KOKKOS_ASSERT((ivar >= 0) && "invalid ivar index value");
    KOKKOS_ASSERT((ivar < m_data.num_vars()) && "invalid ivar index value");

    KOKKOS_ASSERT((iOct >= 0) && "invalid iOct index value");
    KOKKOS_ASSERT((iOct < m_data.num_quadrants()) && "invalid iOct index value");

    return m_data(i0 - m_shift[IX], i1 - m_shift[IY], i2 - m_shift[IZ], ivar, iOct);
  }

  // ==================================================================================
  /**
   * Direct memory access operator() - 3d case.
   *
   * \param[in] i0 cell coordinate inside block/octant along X axis
   * \param[in] i1 cell coordinate inside block/octant along Y axis
   * \param[in] i2 cell coordinate inside block/octant along Z axis
   * \param[in] ivar variable index
   * \param[in] iOct octant/block index
   *
   * \note index i0,i1,i2 must be coordinates relative to the ghosted block (positive values)
   */
  template <typename I0,
            typename I1,
            typename I2,
            typename IVar,
            typename IOct,
            int dim_ = dim,
            std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, I2 i2, IVar ivar, IOct iOct, DirectAccess const &) const
  {
    KOKKOS_ASSERT((i0 >= 0) && "Invalid index value");
    KOKKOS_ASSERT((i1 >= 0) && "Invalid index value");
    KOKKOS_ASSERT((i2 >= 0) && "Invalid index value");
    KOKKOS_ASSERT((i0 < m_data.shape()[IX]) && "Invalid index value");
    KOKKOS_ASSERT((i1 < m_data.shape()[IY]) && "Invalid index value");
    KOKKOS_ASSERT((i2 < m_data.shape()[IZ]) && "Invalid index value");

    KOKKOS_ASSERT((ivar >= 0) && "invalid ivar index value");
    KOKKOS_ASSERT((ivar < m_data.num_vars()) && "invalid ivar index value");

    KOKKOS_ASSERT((iOct >= 0) && "invalid iOct index value");
    KOKKOS_ASSERT((iOct < m_data.num_quadrants()) && "invalid iOct index value");

    return m_data(i0, i1, i2, ivar, iOct);
  }

  // ==================================================================================
  /**
   * Memory access operator() using a flat cell index (2d and 3d).
   *
   * \param[in] icell cell index inside block/octant
   * \param[in] ivar variable index
   * \param[in] iOct octant/block index
   *
   * \note cell index icell must have been computed using coordinates relative to the ghosted block
   * (positive values).
   */
  template <typename ICell, typename IVar, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(ICell icell, IVar ivar, IOct iOct) const
  {
    return m_data(icell, ivar, iOct);
  }

  // ==================================================================================
  /** Memory access operator() using vector of coordinates.
   *
   * \param[in] ijk cell coordinates vector identifyin a cell inside block/octant
   * \param[in] ivar variable index
   * \param[in] iOct octant/block index
   *
   * \note indexes ijk[IX],...,ijk[dim-1] must be coordinates relative to the non-ghosted block
   * (signed values)
   */
  template <typename IVar, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(coord_t<dim> ijk, IVar ivar, IOct iOct) const
  {
    KOKKOS_ASSERT((ijk[IX] - m_shift[IX] >= 0) && "Invalid index value");
    KOKKOS_ASSERT((ijk[IY] - m_shift[IY] >= 0) && "Invalid index value");
    if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((ijk[IZ] - m_shift[IZ] >= 0) && "Invalid index value");
    }
    KOKKOS_ASSERT((ijk[IX] - m_shift[IX] < m_data.shape()[IX]) && "Invalid index value");
    KOKKOS_ASSERT((ijk[IY] - m_shift[IY] < m_data.shape()[IY]) && "Invalid index value");
    if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((ijk[IZ] - m_shift[IZ] < m_data.shape()[IZ]) && "Invalid index value");
    }

    if constexpr (dim == 2)
    {
      return this->operator()(ijk[IX], ijk[IY], ivar, iOct);
    }
    else if constexpr (dim == 3)
    {
      return this->operator()(ijk[IX], ijk[IY], ijk[IZ], ivar, iOct);
    }
  }

  // ==================================================================================
  /** Direct memory access operator() using vector of coordinates.
   *
   * \param[in] ijk cell coordinates vector identifyin a cell inside block/octant
   * \param[in] ivar variable index
   * \param[in] iOct octant/block index
   *
   * \note indexes ijk[IX],...,ijk[dim-1] must be coordinates relative to the ghosted block
   * (positive values)
   */
  template <typename IVar, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(coord_t<dim> ijk, IVar ivar, IOct iOct, DirectAccess const &) const
  {
    KOKKOS_ASSERT((ijk[IX] >= 0) && "Invalid index value");
    KOKKOS_ASSERT((ijk[IY] >= 0) && "Invalid index value");
    if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((ijk[IZ] >= 0) && "Invalid index value");
    }
    KOKKOS_ASSERT((ijk[IX] < m_data.shape()[IX]) && "Invalid index value");
    KOKKOS_ASSERT((ijk[IY] < m_data.shape()[IY]) && "Invalid index value");
    if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((ijk[IZ] < m_data.shape()[IZ]) && "Invalid index value");
    }

    if constexpr (dim == 2)
    {
      return this->operator()(ijk[IX], ijk[IY], ivar, iOct, DirectAccess{});
    }
    else if constexpr (dim == 3)
    {
      return this->operator()(ijk[IX], ijk[IY], ijk[IZ], ivar, iOct, DirectAccess{});
    }
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

  /**
   * Change ghosted block size, and shift without reallocating.
   *
   * We only require that the new shape corresponds to a smaller number of cells.
   *
   * \return true when the new shape is accepted (i.e. compatible with current memory allocation)
   */
  KOKKOS_FORCEINLINE_FUNCTION bool
  reshape(block_size_t<dim> const & new_ghosted_size, shift_t<dim> const & new_shift)
  {
    // when reshaping, as we don't want to reallocate, we need to make sure the new
    // ghosted size is not larger than current allocation

    // only reshape when possible
    if (Kokkos::dim_prod(new_ghosted_size) <= Kokkos::dim_prod(m_data.block_size()))
    {

      m_data.reshape(new_ghosted_size);

      // just make sure the new shift is valid
      KOKKOS_ASSERT((new_shift[IX] + m_bSize[IX] / 2 >= 0) && "shift[IX] is too large (negative)");
      KOKKOS_ASSERT((new_shift[IY] + m_bSize[IY] / 2 >= 0) && "shift[IY] is too large (negative)");
      if constexpr (dim == 3)
      {
        KOKKOS_ASSERT((new_shift[IZ] + m_bSize[IZ] / 2 >= 0) &&
                      "shift[IZ] is too large (negative)");
      }

      KOKKOS_ASSERT((new_shift[IX] <= m_bSize[IX] / 2) && "shift[IX] is too large");
      KOKKOS_ASSERT((new_shift[IY] <= m_bSize[IY] / 2) && "shift[IY] is too large");
      if constexpr (dim == 3)
      {
        KOKKOS_ASSERT((new_shift[IZ] <= m_bSize[IZ] / 2) && "shift[IZ] is too large");
      }

      // update shift
      m_shift = new_shift;

      // update block size of the overlapping zone
      m_bSize_overlap = get_block_size_overlap(new_ghosted_size, new_shift);

      return true;
    }

    // reshaping can't happen, don't modify neither shape, neither shift
    return false;

  } // reshape

  /**
   * Change ghosted block size, inner size and shift without reallocating.
   *
   * We only require that the new shape corresponds to a smaller number of cells.
   *
   * This variant of reshape is only useful for the corner case when
   * - the inner block is actually a flux array (i.e. slightly larger is in one direction)
   * - reshaping need the reshape both the inner shape and ghosted shape
   *
   * \return true when the new shape is accepted (i.e. compatible with current memory allocation)
   */
  KOKKOS_FORCEINLINE_FUNCTION bool
  reshape(block_size_t<dim> const & new_inner_size,
          block_size_t<dim> const & new_ghosted_size,
          shift_t<dim> const &      new_shift)
  {
    auto bSize_tmp = m_bSize;
    m_bSize = new_inner_size;

    auto valid = reshape(new_ghosted_size, new_shift);

    if (!valid)
    {
      // restore old bSize
      m_bSize = bSize_tmp;
    }
    return valid;
  } // reshape

  //! reset ghosted block shape to the original one used for memory allocation
  //! TODO: does this necessarily need to be available on device ?
  KOKKOS_FORCEINLINE_FUNCTION void
  shape_reset()
  {
    m_data.shape_reset();
  }

  //! Return total number of cells of the ghosted block.
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

  //! return the shifted block size as used in allocation by constructor; this won't are reshaping
  //! are only allowed using a smaller shape
  auto
  allocated_ghosted_block_size() const
  {
    return m_data.block_size();
  }

  //! return the block sizes of the shifted block
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  ghosted_block_size() const
  {
    return m_data.shape();
  }

  //! return the block sizes of the shifted block (same as ghosted_block_size)
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  shape() const
  {
    return m_data.shape();
  }

  //! return the shift (space displacement)
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  shift() const
  {
    return m_shift;
  }

  //! return block size of the block of cells where original block and shifted block
  //! overlap
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  get_block_size_overlap() const
  {
    return m_bSize_overlap;
  }

  //! compute and return block size of the block of cells where original block and shifted block
  //! overlap
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  get_block_size_overlap(block_size_t<dim> ghosted_size, shift_t<dim> shift) const
  {
    block_size_t<dim> bSize_overlap;

    coord_t<dim> start;
    start[IX] = shift[IX] < 0 ? 0 : shift[IX];
    start[IY] = shift[IY] < 0 ? 0 : shift[IY];
    if constexpr (dim == 3)
    {
      start[IZ] = shift[IZ] < 0 ? 0 : shift[IZ];
    }

    coord_t<dim> end;
    end[IX] =
      shift[IX] + ghosted_size[IX] >= m_bSize[IX] ? m_bSize[IX] : shift[IX] + ghosted_size[IX];
    end[IY] =
      shift[IY] + ghosted_size[IY] >= m_bSize[IY] ? m_bSize[IY] : shift[IY] + ghosted_size[IY];
    if constexpr (dim == 3)
    {
      end[IZ] =
        shift[IZ] + ghosted_size[IZ] >= m_bSize[IZ] ? m_bSize[IZ] : shift[IZ] + ghosted_size[IZ];
    }

    bSize_overlap[IX] = end[IX] - start[IX];
    bSize_overlap[IY] = end[IY] - start[IY];
    if constexpr (dim == 3)
    {
      bSize_overlap[IZ] = end[IZ] - start[IZ];
    }

    return bSize_overlap;
  }

  //! return lower left corner coordinate of the block where original block and shifted block
  //! overlap
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  get_start_overlap() const
  {

    coord_t<dim> start;
    start[IX] = m_shift[IX] < 0 ? 0 : m_shift[IX];
    start[IY] = m_shift[IY] < 0 ? 0 : m_shift[IY];
    if constexpr (dim == 3)
    {
      start[IZ] = m_shift[IZ] < 0 ? 0 : m_shift[IZ];
    }
    return start;
  }

  //! Return the total allocated memory in bytes.
  auto
  allocated_size_in_bytes() const
  {
    return m_data.allocated_size_in_bytes();
  }

}; // DataArrayGhostedBlock

} // namespace kalypsso

#endif // KALYPSSO_CORE_DATAARRAYBLOCK_H_
