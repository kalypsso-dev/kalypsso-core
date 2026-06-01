// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FaceDataArrayBlock.h
 */
#ifndef KALYPSSO_CORE_FACEDATAARRAYBLOCK_H_
#define KALYPSSO_CORE_FACEDATAARRAYBLOCK_H_

#include <kalypsso/core/FaceDataArrayBlock_utils.h>
#include <kalypsso/core/DataArrayBlock.h>

namespace kalypsso
{

// ============================================================================================
// ============================================================================================
// ============================================================================================
/**
 * A Data container helper to store d-dimensional vector field face-staggered components.
 *
 * More precisely, taking as example magnetic field used in constraint transport methods:
 * - Bx is center on x-faces
 * - By is center on y-faces
 * - Bz is center on z-faces
 *
 * \note Currently in kalypsso blocks can only be of equal size in each direction.
 * So, for example, in 3D, each component is stored using an array of N x N x (N+1) elements.
 * Later, if needed, we could add support for block of independent sizes along each direction.
 *
 * \note Bz is actually used in 2d.
 *
 */
template <size_t dim, typename T, typename device_t>
class FaceDataArrayBlock
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, T, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, T, device_t>;

  using FaceFlatArray_t = Kokkos::View<T *, device_t>;
  using FaceFlatArrayUnmanaged_t =
    Kokkos::View<T *, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using ExecutionSpace = typename device_t::execution_space;

  //! tag dispatch class to use when using non-signed index coordinates (i.e. relative to the
  //! shifted block origin); when using direct access, valid index must be in the following ranges:
  //! i in [shift[IX], shift[IX]+m_bSize_total[IX]]
  //! j in [shift[IY], shift[IY]+m_bSize_total[IY]]
  //! k in [shift[IZ], shift[IZ]+m_bSize_total[IZ]]
  //!
  struct DirectAccess
  {};

private:
  //! block size : number of cells in each direction (without counting ghost cells).
  block_size_t<dim> m_bSize;

  //! space shift in unit of number of cells
  //! in each direction, shift must be even if you plan to use prologation operator (using odd
  //! integer would be too complex to implement).
  //!
  //! \note Using a negative shift means we want to have ghost cells on the left of
  //! the block of cells.
  //!
  //! See class DataArrayGhostedBlock which uses the same semantic
  shift_t<dim> m_shift;

  //! block size : number of cells in each direction (ghost cells included).
  //! \note this parameter combined with m_shift defines a window that may completely or partially
  //! overlap with the original block of cells (from the AMR mesh).
  block_size_t<dim> m_bSize_total;

  //! block size used for memory allocation
  block_size_t<dim> m_bSize_total_allocated;

  //! number of x-faces in each direction - used for flat index computation
  block_size_t<dim> m_bSize_face_x;

  //! number of y-faces in each direction - used for flat index computation
  block_size_t<dim> m_bSize_face_y;

  //! number of z-faces in each direction - used for flat index computation
  block_size_t<dim> m_bSize_face_z;

  //! offset used as starting address of each component of the field inside a block
  //! - m_offsets[1] number of x-faces elements per octant
  //! - m_offsets[2] number of x-faces and y-faces elements per octant
  //! - m_offsets[3] total num of faces elements (x,y and z) per octant
  face_flat_index_offset_t m_offsets;

  //! total number of quadrants (owned, MPI ghost, outside, outside_ghosts)
  //! may changed when resizing
  int m_num_quadrants;

  //! storage capacity (should be larger or equal to m_offsets[3] * num_quads).
  //! actual (physical) number of elements in m_storage
  //! it is increased upon resizing only when necessary
  size_t m_storage_capacity;

  //! 1D storage array, it should contain enough space to hold m_data_x, m_data_y and m_data_z
  //! may be reallocated when the number of octants changes
  FaceFlatArray_t m_storage_data;

public:
  FaceDataArrayBlock() = default;

  /**
   * Constructor without ghost cells.
   *
   * \param[in] name is a label used to name the underlying Kokkos::View
   * \param[in] bSize is the cell block size
   * \param[in] num_quadrants number of quadrant (used for memory allocation)
   */
  FaceDataArrayBlock(std::string name, block_size_t<dim> bSize, int num_quadrants)
    : m_bSize(bSize)
    , m_shift(get_shift<dim>(0))
    , m_bSize_total(bSize)
    , m_bSize_total_allocated(bSize)
    , m_bSize_face_x(get_face_block_size<dim, IX>(bSize))
    , m_bSize_face_y(get_face_block_size<dim, IY>(bSize))
    , m_bSize_face_z(get_face_block_size<dim, IZ>(bSize))
    , m_offsets(compute_face_flat_index_offsets<dim>(bSize))
    , m_num_quadrants(num_quadrants)
    , m_storage_capacity(
        DataArrayUtils::allocated_capacity(m_offsets[3] * static_cast<size_t>(num_quadrants)))
    , m_storage_data(Kokkos::view_alloc(Kokkos::WithoutInitializing, name), m_storage_capacity)
  {} // FaceDataArrayBlock

  /**
   * Constructor with a uniform layer of ghost cells all around.
   *
   * \param[in] name is a label used to name the underlying Kokkos::View
   * \param[in] bSize is the cell block size
   * \param[in] ghostwidth is the ghost width (thickness) in unit of cell
   * \param[in] num_quadrants number of quadrant (used for memory allocation)
   */
  FaceDataArrayBlock(std::string       name,
                     block_size_t<dim> bSize,
                     int32_t           ghostwidth,
                     int               num_quadrants)
    : m_bSize(bSize)
    , m_shift(get_shift<dim>(-ghostwidth))
    , m_bSize_total(bSize + 2 * ghostwidth)
    , m_bSize_total_allocated(bSize + 2 * ghostwidth)
    , m_bSize_face_x(get_face_block_size<dim, IX>(bSize + 2 * ghostwidth))
    , m_bSize_face_y(get_face_block_size<dim, IY>(bSize + 2 * ghostwidth))
    , m_bSize_face_z(get_face_block_size<dim, IZ>(bSize + 2 * ghostwidth))
    , m_offsets(compute_face_flat_index_offsets<dim>(bSize + 2 * ghostwidth))
    , m_num_quadrants(num_quadrants)
    , m_storage_capacity(
        DataArrayUtils::allocated_capacity(m_offsets[3] * static_cast<size_t>(num_quadrants)))
    , m_storage_data(Kokkos::view_alloc(Kokkos::WithoutInitializing, name), m_storage_capacity)
  {
    KOKKOS_ASSERT(ghostwidth >= 0 and
                  "[FaceDataArrayBlock::FaceDataArrayBlock] ghostwidth has wrong value");
  } // FaceDataArrayBlock

  /**
   * Constructor with a custom layer of ghost cells all around.
   *
   * Useful e.g. when one want to have ghost faces only in one direction.
   *
   * \param[in] name is a label used to name the underlying Kokkos::View
   * \param[in] bSize is the cell block size (ghosts non-included)
   * \param[in] bSize_total is the cell block size (ghosts included)
   * \param[in] shift in unit of cell
   * \param[in] num_quadrants number of quadrant (used for memory allocation)
   */
  FaceDataArrayBlock(std::string       name,
                     block_size_t<dim> bSize,
                     block_size_t<dim> bSize_total,
                     shift_t<dim>      shift,
                     int               num_quadrants)
    : m_bSize(bSize)
    , m_shift(shift)
    , m_bSize_total(bSize_total)
    , m_bSize_total_allocated(bSize_total)
    , m_bSize_face_x(get_face_block_size<dim, IX>(bSize_total))
    , m_bSize_face_y(get_face_block_size<dim, IY>(bSize_total))
    , m_bSize_face_z(get_face_block_size<dim, IZ>(bSize_total))
    , m_offsets(compute_face_flat_index_offsets<dim>(bSize_total))
    , m_num_quadrants(num_quadrants)
    , m_storage_capacity(
        DataArrayUtils::allocated_capacity(m_offsets[3] * static_cast<size_t>(num_quadrants)))
    , m_storage_data(Kokkos::view_alloc(Kokkos::WithoutInitializing, name), m_storage_capacity)
  {} // FaceDataArrayBlock

  /**
   * accumulate over all direction the number of face elements in that direction per octant.
   */
  KOKKOS_FORCEINLINE_FUNCTION auto
  offsets() const
  {
    return m_offsets;
  } // offsets

  /**
   * convert a logical multi-index (i, j, iOct) into a flat index to storage.
   *
   * \note IMPORTANT multi-index must be non-shifted positive integer.
   */
  template <int dir, typename I0, typename I1, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION int64_t
  flat_index(I0 i0, I1 i1, IOct iOct) const
  {
    if constexpr (dir == IX)
    {
      KOKKOS_ASSERT((static_cast<int32_t>(i0) < m_bSize_face_x[IX]) and
                    (static_cast<int32_t>(i0) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT((static_cast<int32_t>(i1) < m_bSize_face_x[IY]) and
                    (static_cast<int32_t>(i1) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i1");
      KOKKOS_ASSERT((static_cast<int32_t>(iOct) < m_num_quadrants) and
                    "[FaceDataArrayBlock::flat_index] wrong index iOct");

      return m_offsets[0] + i0 + m_bSize_face_x[IX] * i1 + iOct * m_offsets[3];
    }
    else if constexpr (dir == IY)
    {
      KOKKOS_ASSERT((static_cast<int32_t>(i0) < m_bSize_face_y[IX]) and
                    (static_cast<int32_t>(i0) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT((static_cast<int32_t>(i1) < m_bSize_face_y[IY]) and
                    (static_cast<int32_t>(i1) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i1");
      KOKKOS_ASSERT(static_cast<int32_t>(iOct) < m_num_quadrants and
                    "[FaceDataArrayBlock::flat_index] wrong index iOct");

      return m_offsets[1] + i0 + m_bSize_face_y[IX] * i1 + iOct * m_offsets[3];
    }
    else if constexpr (dir == IZ)
    {
      KOKKOS_ASSERT((static_cast<int32_t>(i0) < m_bSize_face_z[IX]) and
                    (static_cast<int32_t>(i0) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT((static_cast<int32_t>(i1) < m_bSize_face_z[IY]) and
                    (static_cast<int32_t>(i1) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i1");
      KOKKOS_ASSERT(static_cast<int32_t>(iOct) < m_num_quadrants and
                    "[FaceDataArrayBlock::flat_index] wrong index iOct");

      return m_offsets[2] + i0 + m_bSize_face_z[IX] * i1 + iOct * m_offsets[3];
    }
    return 0;
  } // flat_index

  /**
   * convert a logical multi-index (i, j, k, iOct) into a flat index to storage.
   *
   * \note IMPORTANT multi-index must be non-shifted positive integer.
   */
  template <int dir, typename I0, typename I1, typename I2, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION int64_t
  flat_index(I0 i0, I1 i1, I2 i2, IOct iOct) const
  {
    if constexpr (dir == IX)
    {
      KOKKOS_ASSERT((static_cast<int32_t>(i0) < m_bSize_face_x[IX]) and
                    (static_cast<int32_t>(i0) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT((static_cast<int32_t>(i1) < m_bSize_face_x[IY]) and
                    (static_cast<int32_t>(i1) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i1");
      KOKKOS_ASSERT((static_cast<int32_t>(i2) < m_bSize_face_x[IZ]) and
                    (static_cast<int32_t>(i2) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i2");
      KOKKOS_ASSERT(static_cast<int32_t>(iOct) < m_num_quadrants and
                    "[FaceDataArrayBlock::flat_index] wrong index iOct");

      return m_offsets[0] + i0 + m_bSize_face_x[IX] * (i1 + m_bSize_face_x[IY] * i2) +
             iOct * m_offsets[3];
    }
    else if constexpr (dir == IY)
    {
      KOKKOS_ASSERT((static_cast<int32_t>(i0) < m_bSize_face_y[IX]) and
                    (static_cast<int32_t>(i0) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT((static_cast<int32_t>(i1) < m_bSize_face_y[IY]) and
                    (static_cast<int32_t>(i1) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i1");
      KOKKOS_ASSERT((static_cast<int32_t>(i2) < m_bSize_face_y[IZ]) and
                    (static_cast<int32_t>(i2) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i2");
      KOKKOS_ASSERT(static_cast<int32_t>(iOct) < m_num_quadrants and
                    "[FaceDataArrayBlock::flat_index] wrong index iOct");

      return m_offsets[1] + i0 + m_bSize_face_y[IX] * (i1 + m_bSize_face_y[IY] * i2) +
             iOct * m_offsets[3];
    }
    else if constexpr (dir == IZ)
    {
      KOKKOS_ASSERT((static_cast<int32_t>(i0) < m_bSize_face_z[IX]) and
                    (static_cast<int32_t>(i0) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT((static_cast<int32_t>(i1) < m_bSize_face_z[IY]) and
                    (static_cast<int32_t>(i1) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i1");
      KOKKOS_ASSERT((static_cast<int32_t>(i2) < m_bSize_face_z[IZ]) and
                    (static_cast<int32_t>(i2) >= 0) and
                    "[FaceDataArrayBlock::flat_index] wrong index i2");
      KOKKOS_ASSERT(static_cast<int32_t>(iOct) < m_num_quadrants and
                    "[FaceDataArrayBlock::flat_index] wrong index iOct");

      return m_offsets[2] + i0 + m_bSize_face_z[IX] * (i1 + m_bSize_face_z[IY] * i2) +
             iOct * m_offsets[3];
    }
    return 0;
  } // flat_index

  // ==================================================================================
  //!
  //! Regular memory access operator() - 2d case.
  //!
  template <typename I0, typename I1, typename IDir, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, IDir dir, IOct iOct) const
  {
    KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) and "FaceDataArrayBlock: wrong direction");
    KOKKOS_ASSERT((static_cast<int32_t>(iOct) < m_num_quadrants) and
                  "FaceDataArrayBlock: index iOct too large");

    if (dir == IX)
      return m_storage_data(flat_index<IX>(i0 - m_shift[IX], i1 - m_shift[IY], iOct));
    else if (dir == IY)
      return m_storage_data(flat_index<IY>(i0 - m_shift[IX], i1 - m_shift[IY], iOct));
    else if (dir == IZ)
      return m_storage_data(flat_index<IZ>(i0 - m_shift[IX], i1 - m_shift[IY], iOct));
    else // shouldn't be here
      return m_storage_data(flat_index<IX>(i0 - m_shift[IX], i1 - m_shift[IY], iOct));
  }

  // ==================================================================================
  //!
  //! Direct memory access operator() - 2d case.
  //!
  template <typename I0, typename I1, typename IDir, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, IDir dir, IOct iOct, DirectAccess const &) const
  {
    KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) and "FaceDataArrayBlock: wrong direction");
    KOKKOS_ASSERT((static_cast<int32_t>(iOct) < m_num_quadrants) and
                  "FaceDataArrayBlock: index iOct too large");

    if (dir == IX)
      return m_storage_data(flat_index<IX>(i0, i1, iOct));
    else if (dir == IY)
      return m_storage_data(flat_index<IY>(i0, i1, iOct));
    else if (dir == IZ)
      return m_storage_data(flat_index<IZ>(i0, i1, iOct));
    else // shouldn't be here
      return m_storage_data(flat_index<IX>(i0, i1, iOct));
  }

  // ==================================================================================
  //!
  //! Regular memory access operator() - 3d case.
  //!
  template <typename I0, typename I1, typename I2, typename IDir, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, I2 i2, IDir dir, IOct iOct) const
  {
    KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) and "FaceDataArrayBlock: wrong direction");
    KOKKOS_ASSERT((static_cast<int32_t>(iOct) < m_num_quadrants) and
                  "FaceDataArrayBlock: index iOct too large");

    if (dir == IX)
      return m_storage_data(
        flat_index<IX>(i0 - m_shift[IX], i1 - m_shift[IY], i2 - m_shift[IZ], iOct));
    else if (dir == IY)
      return m_storage_data(
        flat_index<IY>(i0 - m_shift[IX], i1 - m_shift[IY], i2 - m_shift[IZ], iOct));
    else if (dir == IZ)
      return m_storage_data(
        flat_index<IZ>(i0 - m_shift[IX], i1 - m_shift[IY], i2 - m_shift[IZ], iOct));
    else // shouldn't be here
      return m_storage_data(
        flat_index<IX>(i0 - m_shift[IX], i1 - m_shift[IY], i2 - m_shift[IZ], iOct));
  }

  // ==================================================================================
  //!
  //! Direct memory access operator() - 3d case.
  //!
  template <typename I0, typename I1, typename I2, typename IDir, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, I2 i2, IDir dir, IOct iOct, DirectAccess const &) const
  {
    KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) and "FaceDataArrayBlock: wrong direction");
    KOKKOS_ASSERT((static_cast<int32_t>(iOct) < m_num_quadrants) and
                  "FaceDataArrayBlock: index iOct too large");

    if (dir == IX)
      return m_storage_data(flat_index<IX>(i0, i1, i2, iOct));
    else if (dir == IY)
      return m_storage_data(flat_index<IY>(i0, i1, i2, iOct));
    else if (dir == IZ)
      return m_storage_data(flat_index<IZ>(i0, i1, i2, iOct));
    else // shouldn't be here
      return m_storage_data(flat_index<IX>(i0, i1, i2, iOct));
  }

  // ==================================================================================
  //!
  //! Regular memory access operator() - 2d and 3d case.
  //!
  template <typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(face_multiindex_t<dim> const & face_index, IOct iOct) const
  {

    if constexpr (dim == 2)
    {
      return this->operator()(face_index[IX], face_index[IY], face_index[dim], iOct);
    }
    else if constexpr (dim == 3)
    {
      return this->operator()(
        face_index[IX], face_index[IY], face_index[IZ], face_index[dim], iOct);
    }
  }

  // ==================================================================================
  //! Access to raw pointer
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  data() const
  {
    return m_storage_data.data();
  }

  // ==================================================================================
  //! return internal Kokkos::View used for raw storage.
  //! A regular user should probably never have to use the physical, but most surely the
  //! logical_view (with logical sizes).
  auto
  physical_view() -> decltype(m_storage_data)
  {
    return m_storage_data;
  }

  auto
  physical_view_ref() -> decltype(m_storage_data) &
  {
    return m_storage_data;
  }

public:
  // ==================================================================================
  //! return logical size (total number of elements). Not to be confused with capacity (physical
  //! size)
  auto
  logical_size_in_elements() const
  {
    return m_offsets[3] * static_cast<size_t>(m_num_quadrants);
  }

  // ==================================================================================
  auto
  logical_view() const
  {
    const auto logical_range =
      std::pair<std::size_t, std::size_t>(0, this->logical_size_in_elements());
    return Kokkos::subview(m_storage_data, logical_range);
  }

  //! Resize m_storage data using a new number of octants.
  //!
  //! This function resizes data without copying old data, without even initializing the new array.
  //! This is enough especially when doing load balancing; data will be initialized by MPI
  //! communications.
  void
  resize(int32_t num_quads)
  {
    // update number of quadrants
    m_num_quadrants = num_quads;

    const size_t new_size = m_offsets[3] * static_cast<size_t>(m_num_quadrants);

    // only resize when the requested new size is larger than capacity
    if (new_size > m_storage_capacity)
    {
      // new storage capacity
      size_t new_storage_capacity =
        DataArrayUtils::allocated_capacity(m_offsets[3] * static_cast<size_t>(num_quads));

      Kokkos::resize(
        Kokkos::view_alloc(Kokkos::WithoutInitializing), m_storage_data, new_storage_capacity);

      m_storage_capacity = new_storage_capacity;
    }

  } // resize

  //! Resize m_storage data using a new number of octant.
  //! This function resizes data; it doesn't copy old data, but default initialize to zero.
  void
  resize_and_reset(int32_t num_quads)
  {
    resize(num_quads);
    Kokkos::deep_copy(m_storage_data, 0);

  } // resize_and_reset

  /**
   * Change ghosted block size, and shift without reallocating.
   *
   * We only require that the new shape corresponds to a smaller number of cells.
   *
   * \return true when the new shape is accepted (i.e. compatible with current memory allocation)
   */
  KOKKOS_FORCEINLINE_FUNCTION bool
  reshape(block_size_t<dim> const & new_bSize_total, shift_t<dim> const & new_shift)
  {
    // when reshaping, as we don't want to reallocate, we need to make sure the new
    // ghosted size is not larger than current allocation
    // KOKKOS_ASSERT(Kokkos::dim_prod(new_ghosted_size) <= Kokkos::dim_prod(m_data.block_size()) &&
    //               "Can't reshape DataArrayGhostedBlock, new ghosted size is too large.");

    // only reshape when possible
    if (compute_total_number_of_faces(new_bSize_total) <=
        compute_total_number_of_faces(m_bSize_total_allocated))
    {

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

      // recompute face block sizes and offsets
      m_bSize_total = new_bSize_total;
      m_bSize_face_x = get_face_block_size<dim, IX>(m_bSize_total);
      m_bSize_face_y = get_face_block_size<dim, IY>(m_bSize_total);
      m_bSize_face_z = get_face_block_size<dim, IZ>(m_bSize_total);
      m_offsets = compute_face_flat_index_offsets<dim>(m_bSize_total);

      // update shift
      m_shift = new_shift;

      return true;
    }

    // reshaping can't happen, don't modify neither shape, neither shift
    return false;

  } // reshape

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_quadrants() const
  {
    return m_num_quadrants;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int32_t
  num_elements_per_octant() const
  {
    return m_offsets[3];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  cell_block_size() const
  {
    return m_bSize_total;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  cell_block_size_inner() const
  {
    return m_bSize;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  shift() const
  {
    return m_shift;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  face_block_size(int dir) const
  {
    if (dir == IX)
      return m_bSize_face_x;
    else if (dir == IY)
      return m_bSize_face_y;
    else if (dir == IZ)
      return m_bSize_face_z;

    return m_bSize_face_x;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  is_non_ghosted() const
  {
    if constexpr (dim == 2)
    {
      return m_shift[IX] == 0 and m_shift[IY] == 0 and m_bSize_total == m_bSize;
    }
    else if constexpr (dim == 3)
    {
      return m_shift[IX] == 0 and m_shift[IY] == 0 and m_shift[IZ] == 0 and
             m_bSize_total == m_bSize;
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  is_ghosted() const
  {
    return !is_non_ghosted();
  }

  auto
  label() const
  {
    return m_storage_data.label();
  }

  auto
  allocated_size_in_bytes() const
  {
    uint64_t size = m_storage_data.extent(0) * sizeof(T);
    return size;
  }

  /**
   * \return maximum number of faces inside a block (ghost cells included).
   */
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  max_num_faces_per_leaf() const
  {
    return maximum_number_faces_per_leaf<dim>(m_bSize_total);
  }

  /**
   * For a given component (e.g. Bx, By or Bz), get the last valid index in given direction.
   *
   * \param[in] component (IX, IY, or IZ for Bx, By or Bz)
   * \param[in] direction : IX, IY or IZ
   *
   */
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  last_valid_index_in_direction(int component, int dir) const
  {
    // in 2D, dir can only by IX or IY
    if constexpr (dim == 2)
    {
      KOKKOS_ASSERT((dir == IX or dir == IY) and "FaceDataArrayBlock: wrong direction in 2D");
    }
    else if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) and
                    "FaceDataArrayBlock: wrong direction in 3D");
    }

    return (dir == component) ? m_bSize_total[dir] : m_bSize_total[dir] - 1;

  } // last_valid_index

  /**
   * Just for symmetry with last_valid_index_in_direction
   */
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  first_valid_index_in_direction(int component, int dir) const
  {
    return 0;
  }

public:
  /**
   * A static utility to convert a FaceDataArrayBlock_t into a DataArrayBlock_t separating left from
   * right faces.
   *
   * The resulting DataArrayBlock has 6 components: 3 for left faces and 3 for right faces.
   */
  static DataArrayBlock_t
  to_DataArrayBlock(FaceDataArrayBlock_t fdata)
  {
    const auto label = fdata.label();

    const auto    bSize_total = fdata.cell_block_size();
    const auto    nbCellsPerLeaf = Kokkos::dim_prod(bSize_total);
    const int64_t total_num_cells = nbCellsPerLeaf * fdata.num_quadrants();
    // - 2d: 2 left + 2 right + 1 for IZ direction
    // - 3d: 3 left + 3 right
    const auto num_vars = dim == 3 ? 6 : 5;

    auto res = DataArrayBlock_t(label, bSize_total, num_vars, fdata.num_quadrants());

    Kokkos::parallel_for(
      "to_DataArrayBlock",
      Kokkos::RangePolicy<ExecutionSpace>(0, total_num_cells),
      KOKKOS_LAMBDA(const int64_t & global_index) {
        // convert global index into
        // - octant id
        // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
        const auto iOct = global_index / nbCellsPerLeaf;
        const auto cell_index = static_cast<int32_t>(global_index - iOct * nbCellsPerLeaf);

        const auto iCoord = cellindex_to_coord<dim>(cell_index, bSize_total) + fdata.shift();

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support capturing variables inside the inner lambda inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if ((fdata.num_quadrants() == 0) or (res.num_cells() == 0))
          dummy++;
#endif

        if constexpr (dim == 2)
        {
          auto const & i = iCoord[IX];
          auto const & j = iCoord[IY];

          // clang-format off
          res(cell_index, 0, iOct) = fdata(i    , j    , IX, iOct);
          res(cell_index, 1, iOct) = fdata(i + 1, j    , IX, iOct);
          res(cell_index, 2, iOct) = fdata(i    , j    , IY, iOct);
          res(cell_index, 3, iOct) = fdata(i    , j + 1, IY, iOct);
          res(cell_index, 4, iOct) = fdata(i    , j    , IZ, iOct);
          // clang-format on
        }
        else
        {
          auto const & i = iCoord[IX];
          auto const & j = iCoord[IY];
          auto const & k = iCoord[IZ];

          // clang-format off
          res(cell_index, 0, iOct) = fdata(i    , j    , k    , IX, iOct);
          res(cell_index, 1, iOct) = fdata(i + 1, j    , k    , IX, iOct);
          res(cell_index, 2, iOct) = fdata(i    , j    , k    , IY, iOct);
          res(cell_index, 3, iOct) = fdata(i    , j + 1, k    , IY, iOct);
          res(cell_index, 4, iOct) = fdata(i    , j    , k    , IZ, iOct);
          res(cell_index, 5, iOct) = fdata(i    , j    , k + 1, IZ, iOct);
          // clang-format on
        }
      });

    return res;
  } // FaceDataArrayBlock::to_DataArrayBlock

  /**
   * A static utility to convert one component to cell centered data.
   *
   * \note This should probably only be used when input FacaDataArrayBlock_t is non-ghosted.
   */
  static DataArrayBlock_t
  to_DataArrayBlockCentered(FaceDataArrayBlock_t fdata, int dir)
  {
    const std::string type_str("_cell_centered_");
    const auto        label = fdata.label() + type_str + std::to_string(dir);

    const auto    bSize_total = fdata.cell_block_size();
    const auto    nbCellsPerLeaf = Kokkos::dim_prod(bSize_total);
    const int64_t total_num_cells = nbCellsPerLeaf * fdata.num_quadrants();
    const auto    num_vars = 1; // only one value per cell

    auto res = DataArrayBlock_t(label, bSize_total, num_vars, fdata.num_quadrants());

    // compute unit vector associated to direction
    const auto unit_vector = [](int direction) {
      Kokkos::Array<int, dim> v;
      for (int i = 0; i < static_cast<int>(dim); ++i)
      {
        v[i] = i == direction ? 1 : 0;
      }
      return v;
    }(dir);

    Kokkos::parallel_for(
      "to_DataArrayBlock",
      Kokkos::RangePolicy<ExecutionSpace>(0, total_num_cells),
      KOKKOS_LAMBDA(const int64_t & global_index) {
        // convert global index into
        // - octant id
        // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
        const auto iOct = global_index / nbCellsPerLeaf;
        const auto cell_index = global_index - iOct * nbCellsPerLeaf;

        const auto iCoord =
          cellindex_to_coord<dim>(static_cast<int32_t>(cell_index), bSize_total) + fdata.shift();

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support capturing variables inside the inner lambda inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if ((fdata.num_quadrants() == 0) or (res.num_cells() == 0) or dir != dir or
            unit_vector[0] != unit_vector[0])
          dummy++;
#endif

        if constexpr (dim == 2)
        {
          auto const & i = iCoord[IX];
          auto const & j = iCoord[IY];

          res(cell_index, 0, iOct) =
            HALF_F *
            (fdata(i, j, dir, iOct) + fdata(i + unit_vector[IX], j + unit_vector[IY], dir, iOct));
        }
        else
        {
          auto const & i = iCoord[IX];
          auto const & j = iCoord[IY];
          auto const & k = iCoord[IZ];

          res(cell_index, 0, iOct) =
            HALF_F *
            (fdata(i, j, k, dir, iOct) +
             fdata(i + unit_vector[IX], j + unit_vector[IY], k + unit_vector[IZ], dir, iOct));
        }
      });

    return res;
  } // FaceDataArrayBlock::to_DataArrayBlockCentered

  /**
   * Compute divergence of a FaceDataArrayBlock.
   *
   * \param[in] fdata is the face data array
   * \param[in] orchard_keys
   *
   * \note This should probably only be used when input FacaDataArrayBlock_t is non-ghosted.
   *
   */
  static DataArrayBlock_t
  compute_divergence(FaceDataArrayBlock_t                          fdata,
                     typename orchard_key_base_t<device_t>::view_t orchard_keys)
  {
    const std::string div("divergence_of_");
    const auto        label = div + fdata.label();

    const auto    bSize_total = fdata.cell_block_size();
    const auto    nbCellsPerLeaf = Kokkos::dim_prod(bSize_total);
    const int64_t total_num_cells = nbCellsPerLeaf * fdata.num_quadrants();
    const auto    num_vars = 1; // only one value per cell ( divergence is a scalar )

    auto res = DataArrayBlock_t(label, bSize_total, num_vars, fdata.num_quadrants());

    Kokkos::parallel_for(
      "compute divergence of FaceDataArrayBlock",
      Kokkos::RangePolicy<ExecutionSpace>(0, total_num_cells),
      KOKKOS_LAMBDA(const int64_t & global_index) {
        // convert global index into
        // - octant id
        // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
        const auto iOct = global_index / nbCellsPerLeaf;
        const auto cell_index = global_index - iOct * nbCellsPerLeaf;

        // get block level
        const auto level = orchard_key_t<dim>::level(orchard_keys(iOct));

        const auto iCoord =
          cellindex_to_coord<dim>(static_cast<int32_t>(cell_index), bSize_total) + fdata.shift();

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support capturing variables inside the inner lambda inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (fdata.num_quadrants() == 0 or res.num_cells() == 0)
          dummy++;
#endif

        if constexpr (dim == 2)
        {
          auto const & i = iCoord[IX];
          auto const & j = iCoord[IY];
          const auto   dx = compute_cell_length<dim>(level, bSize_total[IX]);
          const auto & dy = dx; // assuming square cells

          res(cell_index, 0, iOct) = (fdata(i + 1, j, IX, iOct) - fdata(i, j, IX, iOct)) / dx +
                                     (fdata(i, j + 1, IY, iOct) - fdata(i, j, IY, iOct)) / dy;
        }
        else if constexpr (dim == 3)
        {
          auto const & i = iCoord[IX];
          auto const & j = iCoord[IY];
          auto const & k = iCoord[IZ];

          const auto   dx = compute_cell_length<dim>(level, bSize_total[IX]);
          const auto & dy = dx; // assuming square cells
          const auto & dz = dx; // assuming square cells

          res(cell_index, 0, iOct) =
            (fdata(i + 1, j, k, IX, iOct) - fdata(i, j, k, IX, iOct)) / dx +
            (fdata(i, j + 1, k, IY, iOct) - fdata(i, j, k, IY, iOct)) / dy +
            (fdata(i, j, k + 1, IZ, iOct) - fdata(i, j, k, IZ, iOct)) / dz;
        }
      });

    return res;

  } // FaceDataArrayBlock::compute_divergence

}; // FaceDataArrayBlock

} // namespace kalypsso

#endif // KALYPSSO_CORE_FACEDATAARRAYBLOCK_H_
