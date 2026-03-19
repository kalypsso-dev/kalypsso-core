// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DataArrayBlockMultiVar.h
 */

#ifndef KALYPSSO_CORE_DATA_ARRAY_BLOCK_MULTI_VAR_H_
#define KALYPSSO_CORE_DATA_ARRAY_BLOCK_MULTI_VAR_H_

#include <kalypsso/core/DataArray.h>
#include <kalypsso/core/DataArrayBlock.h>
#include <kalypsso/core/DataArrayGhostedBlock.h>
#include <kalypsso/core/MaterialPresence.h>

namespace kalypsso
{

// ================================================================================================
// ================================================================================================

//! Wide index { i_var, i_oct }
using MVBlockIndex_t = Kokkos::Array<uint32_t, 2>;

/**
 * \brief Gets the block's index of an octant's variable
 */
template <typename IVar, typename IOct, typename device_t>
KOKKOS_FORCEINLINE_FUNCTION int64_t
flat_mv_block_index(const IVar                            i_var,
                    const IOct                            i_oct,
                    const DataArray<uint32_t, device_t> & offsets)
{
  return static_cast<int64_t>(offsets(i_oct)) + i_var;
}

/**
 * \brief Gets the octant's variable from a block's index (reverse of flat_mv_block_index)
 */
template <typename device_t>
KOKKOS_FORCEINLINE_FUNCTION auto
flat_mv_block_index_unravel(const int64_t index, const DataArray<uint32_t, device_t> & offsets)
{
  uint32_t i_oct = 0;
  while (i_oct + 1 < offsets.size() and offsets(i_oct + 1) <= index)
    i_oct++;

  return MVBlockIndex_t{ static_cast<uint32_t>(index) - offsets(i_oct), i_oct };
}

// ================================================================================================
// ================================================================================================

/**
 * \class DataArrayBlockMultiVar
 *
 * \brief Wrapper around a DataArrayBlock that can handle multiple variables per octant
 *
 * It is essentially a wrapper around a regular DataArrayBlock. It works along an "offset" view that
 * indicates where does the octant starts in the main array.
 *
 * For example, if we have the following setup:
 *
 *           |
 *  Oct 2    | Oct 3
 *  2 Vars   | 1 Var
 *           |
 * ----------+----------
 *           |
 *  Oct 0    | Oct 1
 *  2 Vars   | 1 Var
 *           |
 *
 * Then the offset view will be: [0, 2, 3, 5, 6] (running sum of the number of variables)
 * and the corresponding view will be:
 *
 *      Oct 0     || Oct 1 ||     Oct 2     || Oct 3
 * -------+-------++-------++-------+-------++-------
 *  Var 0 | Var 1 || Var 0 || Var 0 | Var 1 || Var 0
 *
 * In the end, this is a wrapper around a 1-variable DataArrayBlock with the following
 * transformation where the variable and octant indices are fused to have full control on the blocks
 * arrangement:
 *
 * DataArrayBlockMultiVar(ijk, i_var, i_oct) <=> DataArrayBlock(ijk, 0, offset[i_oct] + i_var)
 *
 * This lets us keep the same interface as the usual DataArrayBlock.
 *
 * \tparam dim Dimension of the block
 * \tparam T Value of inner type
 * \tparam device_t Location of the data
 */
template <size_t dim, typename T, typename device_t>
class DataArrayBlockMultiVar
{
public:
  // using Offsets_t = Kokkos::View<uint32_t *, device_t>;
  using Offsets_t = DataArray<uint32_t, device_t>;
  using NumVars_t = Kokkos::View<uint32_t *, device_t>;
  using DataArrayBlock_t = DataArrayBlock<dim, T, device_t>;
  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<dim, T, device_t>;
  using MatPresView_t = MaterialPresenceView<device_t>;

  // ==============================================================================================
  // ==============================================================================================

  //! Default constructor. Empty offset and empty storage
  DataArrayBlockMultiVar() = default;

  /**
   * \brief Constructs an empty DataArrayBlockMultiVar
   *
   * \param name Views' labels
   * \param block_size Block size
   */
  DataArrayBlockMultiVar(const std::string & name, const block_size_t<dim> block_size)
    : m_storage(name, block_size, 1, 0)
    , m_offsets(name + " [offsets]", 0)
  {}

  /**
   * \brief Constructs a new DataArrayBlockMultiVar
   *
   * \param name Views' labels
   * \param block_size Block size
   * \param num_vars Array indicating how many variables an octant has
   */
  DataArrayBlockMultiVar(const std::string &     name,
                         const block_size_t<dim> block_size,
                         const NumVars_t         num_vars)
    : DataArrayBlockMultiVar(name, block_size)
  {
    reorganize(num_vars);
  }

  //! Move constructor
  DataArrayBlockMultiVar(Offsets_t && offsets, DataArrayBlock_t && storage)
    : m_storage(storage)
    , m_offsets(offsets)
  {}

  // ==============================================================================================
  // ==============================================================================================

  /**
   * \brief Recomputes the offsets and resizes the internal storage. Replacement for `resize`.
   */
  void
  reorganize(const NumVars_t num_vars)
  {
    const auto num_blocks = compute_offsets(num_vars);
    m_storage.resize(num_blocks);
  }

  /**
   * \brief Recomputes the offsets and resizes the internal storage from a 'material presence view'.
   */
  void
  reorganize(const MatPresView_t mat_pres, const uint32_t num_vars_per_mat)
  {
    const auto num_blocks = compute_offsets(mat_pres, num_vars_per_mat);
    m_storage.resize(num_blocks);
  }

  /**
   * \brief Recomputes the offsets, resizes the internal storage and resets the data. Replacement
   * for `resize_and_reset`.
   */
  void
  reorganize_and_reset(const NumVars_t num_vars)
  {
    const auto num_blocks = compute_offsets(num_vars);
    m_storage.resize_and_reset(num_blocks);
  }

  /**
   * \brief Updates the object with another's offsets
   */
  template <size_t dim2, typename T2>
  void
  align_with(const DataArrayBlockMultiVar<dim2, T2, device_t> & other)
  {
    m_offsets = other.offsets();
    m_storage.resize(other.storage().num_quadrants());
  }

  // ==============================================================================================
  // ==============================================================================================

  /**
   * \brief Gets the internal DataArrayBlock
   */
  KOKKOS_INLINE_FUNCTION auto const
  storage() const
  {
    return m_storage;
  }

  /**
   * \brief Gets the internal DataArrayBlock
   */
  KOKKOS_INLINE_FUNCTION auto
  storage()
  {
    return m_storage;
  }

  /**
   * \brief Gets the internal offset array
   */
  KOKKOS_INLINE_FUNCTION auto const
  offsets() const
  {
    return m_offsets;
  }

  /**
   * \brief Gets the internal offset array
   */
  KOKKOS_INLINE_FUNCTION auto
  offsets()
  {
    return m_offsets;
  }

  // ==============================================================================================
  // ==============================================================================================
  // DataArrayBlock proxy functions
  // ==============================================================================================
  // ==============================================================================================

  /**
   * \brief Creates host mirror
   */
  static auto
  create_host_mirror_view(DataArrayBlockMultiVar_t src)
  {
    return DataArrayBlockMultiVar<dim, T, HostDevice>(
      Offsets_t::create_host_mirror_view(src.m_offsets),
      DataArrayBlock_t::create_host_mirror_view(src.m_storage));
  }

  /**
   * \brief Creates host mirror and copy offsets
   */
  static auto
  create_host_mirror_view_and_copy_offsets(DataArrayBlockMultiVar_t src)
  {
    return DataArrayBlockMultiVar<dim, T, HostDevice>(
      Offsets_t::create_host_mirror_view_and_copy(src.m_offsets),
      DataArrayBlock_t::create_host_mirror_view(src.m_storage));
  }

  /**
   * \brief Creates host mirror and copy
   */
  static auto
  create_host_mirror_view_and_copy(DataArrayBlockMultiVar_t src)
  {
    return DataArrayBlockMultiVar<dim, T, HostDevice>(
      Offsets_t::create_host_mirror_view_and_copy(src.m_offsets),
      DataArrayBlock_t::create_host_mirror_view_and_copy(src.m_storage));
  }

  // ==============================================================================================
  // ==============================================================================================

  KOKKOS_FORCEINLINE_FUNCTION bool
  are_shape_and_size_equal() const
  {
    return m_storage.are_shape_and_size_equal();
  }

  /**
   * \brief Reshapes the inner block
   */
  KOKKOS_FORCEINLINE_FUNCTION void
  reshape(block_size_t<dim> const & new_shape)
  {
    m_storage.reshape(new_shape);
  }

  /**
   * \brief Resets the inner block back to its original shape
   */
  KOKKOS_FORCEINLINE_FUNCTION void
  shape_reset()
  {
    m_storage.shape_reset();
  }

  // ==============================================================================================
  // ==============================================================================================

  /**
   * \brief Converts a logical multi-index (cell, var, oct) to a flat index storage
   */
  template <typename ICell, typename IVar, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION auto
  flat_index(ICell i_cell, IVar i_var, IOct i_oct) const
  {
    return m_storage.flat_index(i_cell, 0, flat_mv_block_index(i_var, i_oct, m_offsets));
  }

  /**
   * \brief Converts a logical multi-index (i, j, var, oct) to a flat index storage
   */
  template <typename I0,
            typename I1,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION auto
  flat_index(I0 i_0, I1 i_1, IVar i_var, IOct i_oct) const
  {
    return m_storage.flat_index(i_0, i_1, 0, flat_mv_block_index(i_var, i_oct, m_offsets));
  }

  /**
   * \brief Converts a logical multi-index (i, j, k, var, oct) to a flat index storage
   */
  template <typename I0,
            typename I1,
            typename I2,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION auto
  flat_index(I0 i_0, I1 i_1, I2 i_2, IVar i_var, IOct i_oct) const
  {
    return m_storage.flat_index(i_0, i_1, i_2, 0, flat_mv_block_index(i_var, i_oct, m_offsets));
  }

  /**
   * \brief Converts a flat index storage to a logical multi-index (i, j, [k,] var, oct)
   */
  KOKKOS_FORCEINLINE_FUNCTION auto
  flat_index_unravel(int64_t flat_index) const
  {
    auto indices =
      ::kalypsso::flat_index_unravel<dim>(flat_index, m_storage.shape(), m_storage.num_vars());
    const auto octant_unraveled = flat_mv_block_index_unravel(indices[dim + 1], m_offsets);
    indices[dim] = octant_unraveled[0];
    indices[dim + 1] = octant_unraveled[1];
    return indices;
  }

  // ==============================================================================================
  // ==============================================================================================

  /**
   * \brief Memory access operator() with indices
   */
  template <typename I0,
            typename I1,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i_0, I1 i_1, IVar i_var, IOct i_oct) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_offsets.size() - 1) && "Wrong value for i_oct");
    KOKKOS_ASSERT((static_cast<int32_t>(i_var) < num_vars(i_oct)) && "Wrong value for i_var");
#endif

    return m_storage(i_0, i_1, 0, flat_mv_block_index(i_var, i_oct, m_offsets));
  }

  /**
   * \brief Memory access operator() with indices
   */
  template <typename I0,
            typename I1,
            typename I2,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i_0, I1 i_1, I2 i_2, IVar i_var, IOct i_oct) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_offsets.size() - 1) && "Wrong value for i_oct");
    KOKKOS_ASSERT((static_cast<int32_t>(i_var) < num_vars(i_oct)) && "Wrong value for i_var");
#endif

    return m_storage(i_0, i_1, i_2, 0, flat_mv_block_index(i_var, i_oct, m_offsets));
  }

  /**
   * \brief Memory access operator() with cell index
   */
  template <typename ICell, typename IVar, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(ICell i_cell, IVar i_var, IOct i_oct) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_offsets.size() - 1) && "Wrong value for i_oct");
    KOKKOS_ASSERT((static_cast<int32_t>(i_var) < num_vars(i_oct)) && "Wrong value for i_var");
#endif

    return m_storage(i_cell, 0, flat_mv_block_index(i_var, i_oct, m_offsets));
  }

  /**
   * \brief Memory access operator() with cell index and block index
   */
  template <typename ICell, typename IBlk>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(ICell i_cell, IBlk i_blk) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_blk) < m_storage.num_quadrants()) &&
                  "Wrong value for i_blk");
#endif

    return m_storage(i_cell, 0, i_blk);
  }

  // ==============================================================================================
  // ==============================================================================================

  KOKKOS_FORCEINLINE_FUNCTION bool
  is_flux_array(block_size_t<dim> const & b_sizes, int dir) const
  {
    return m_storage.is_flux_array(b_sizes, dir);
  }

  auto
  label() const
  {
    return m_storage.label();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto const &
  block_size() const
  {
    return m_storage.block_size();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto const &
  shape() const
  {
    return m_storage.shape();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_cells() const
  {
    return m_storage.num_cells();
  }

  template <typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION auto
  num_vars(IOct i_oct) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_offsets.size() - 1) && "Wrong value for i_oct");
#endif

    return static_cast<int32_t>(m_offsets(i_oct + 1) - m_offsets(i_oct));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_quadrants() const
  {
    return static_cast<int32_t>(m_offsets.size()) - 1;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_blocks() const
  {
    return m_storage.num_quadrants();
  }

  auto
  allocated_size_in_bytes() const
  {
    return m_offsets.size() * sizeof(typename Offsets_t::value_type) +
           m_storage.allocated_size_in_bytes();
  }

  /**
   * \brief Computes the offset view from an array of number of variables. It simply computes a
   * postfix sum with the total sum at the end.
   */
  int32_t
  compute_offsets(const NumVars_t num_vars)
  {
    m_offsets.resize(num_vars.size() + 1);

    // avoid capturing class member in lambda
    auto offsets = m_offsets;

    Kokkos::RangePolicy<typename device_t::execution_space> policy(0, m_offsets.size());
    uint32_t                                                num_blocks = 0;
    Kokkos::parallel_scan(
      "kalypsso::DataArrayBlockMultiVar::offsets",
      policy,
      KOKKOS_LAMBDA(const uint32_t i_var, uint32_t & offset, const bool is_final) {
        if (is_final)
          offsets(i_var) = offset;
        if (i_var != num_vars.size())
          offset += num_vars(i_var);
      },
      num_blocks);

    return static_cast<int32_t>(num_blocks);
  }

  /**
   * \brief Computes the offset view from a MaterialPresenceView. A "number of variables per
   * material" is still needed but it simply computes a postfix sum with the total sum at the end.
   */
  int32_t
  compute_offsets(const MatPresView_t mat_pres, const uint32_t num_vars_per_mat)
  {
    m_offsets.resize(mat_pres.size() + 1);

    // avoid capturing class member in lambda
    auto offsets = m_offsets;

    Kokkos::RangePolicy<typename device_t::execution_space> policy(0, m_offsets.size());
    uint32_t                                                num_blocks = 0;
    Kokkos::parallel_scan(
      "kalypsso::DataArrayBlockMultiVar::offsets",
      policy,
      KOKKOS_LAMBDA(const uint32_t i_var, uint32_t & offset, const bool is_final) {
        if (is_final)
          offsets(i_var) = offset;
        if (i_var != mat_pres.size())
          offset += static_cast<uint32_t>(mat_pres.num_materials(static_cast<int32_t>(i_var))) *
                    num_vars_per_mat;
      },
      num_blocks);

    return static_cast<int32_t>(num_blocks);
  }

private:
  //! Contains the actual data. Essentially, the handling of an octant's variables is done via the
  //! m_offsets member. So this structure has a single variable per block and an index flattener
  //! picks the correct block from the variable and octant indices.
  DataArrayBlock_t m_storage;

  //! Octant's variables are located at block index [ m_offsets(i) : m_offsets(i+1) ).
  Offsets_t m_offsets;

}; // class DataArrayBlockMultiVar

// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
// ==============================================================================================

// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
// ==============================================================================================

/**
 * \class DataArrayGhostedBlockMultiVar
 *
 * \brief Wrapper around a DataArrayGhostedBlock that can handle multiple variables per octant
 *
 * \tparam dim Dimension of the block
 * \tparam T Value of inner type
 * \tparam device_t Location of the data
 */
template <size_t dim, typename T, typename device_t>
class DataArrayGhostedBlockMultiVar
{
public:
  using Offsets_t = DataArray<uint32_t, device_t>;
  using NumVars_t = Kokkos::View<uint32_t *, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, T, device_t>;
  using DataArrayGhostedBlockMultiVar_t = DataArrayGhostedBlockMultiVar<dim, T, device_t>;
  using MatPresView_t = MaterialPresenceView<device_t>;

  // ==============================================================================================
  // ==============================================================================================

  //! Default constructor. Empty offset and empty storage
  DataArrayGhostedBlockMultiVar() = default;

  /**
   * \brief Constructs an empty DataArrayGhostedBlockMultiVar
   *
   * \param name Views' labels
   * \param block_size Block size
   * \param ghost_size Ghosted size
   * \param shift Shift of original block
   */
  DataArrayGhostedBlockMultiVar(const std::string &     name,
                                const block_size_t<dim> block_size,
                                const block_size_t<dim> ghost_size,
                                const shift_t<dim>      shift)
    : m_storage(block_size, ghost_size, shift, name, 1, 0)
    , m_offsets(name + " [offsets]", 0)
  {}

  /**
   * \brief Constructs a new DataArrayGhostedBlockMultiVar
   *
   * \param name Views' labels
   * \param block_size Block size
   * \param ghost_size Ghosted size
   * \param shift Shift of original block
   * \param num_vars Array indicating how many variables an octant has
   */
  DataArrayGhostedBlockMultiVar(const std::string &     name,
                                const block_size_t<dim> block_size,
                                const block_size_t<dim> ghost_size,
                                const shift_t<dim>      shift,
                                const NumVars_t         num_vars)
    : DataArrayGhostedBlockMultiVar(name, block_size, ghost_size, shift)
  {
    reorganize(num_vars);
  }

  //! Move constructor
  DataArrayGhostedBlockMultiVar(Offsets_t && offsets, DataArrayGhostedBlock_t && storage)
    : m_storage(storage)
    , m_offsets(offsets)
  {}

  // ==============================================================================================
  // ==============================================================================================

  /**
   * \brief Recomputes the offsets and resizes the internal storage. Replacement for `resize`.
   */
  void
  reorganize(const NumVars_t num_vars)
  {
    const auto num_blocks = compute_offsets(num_vars);
    m_storage.resize(num_blocks);
  }

  /**
   * \brief Recomputes the offsets and resizes the internal storage from a 'material presence view'.
   */
  void
  reorganize(const MatPresView_t mat_pres, const uint32_t num_vars_per_mat)
  {
    const auto num_blocks = compute_offsets(mat_pres, num_vars_per_mat);
    m_storage.resize(num_blocks);
  }

  /**
   * \brief Recomputes the offsets, resizes the internal storage and resets the data. Replacement
   * for `resize_and_reset`.
   */
  void
  reorganize_and_reset(const NumVars_t num_vars)
  {
    const auto num_blocks = compute_offsets(num_vars);
    m_storage.resize_and_reset(num_blocks);
  }

  /**
   * \brief Updates the object with another's offsets
   */
  template <size_t dim2, typename T2>
  void
  align_with(const DataArrayGhostedBlockMultiVar<dim2, T2, device_t> & other)
  {
    m_offsets = other.offsets();
    m_storage.resize(other.storage().num_quadrants());
  }

  // ==============================================================================================
  // ==============================================================================================

  /**
   * \brief Gets the internal DataArrayGhostedBlock
   */
  KOKKOS_INLINE_FUNCTION auto const
  storage() const
  {
    return m_storage;
  }

  /**
   * \brief Gets the internal DataArrayGhostedBlock
   */
  KOKKOS_INLINE_FUNCTION auto
  storage()
  {
    return m_storage;
  }

  /**
   * \brief Gets the internal DataArrayGhostedBlock
   */
  KOKKOS_INLINE_FUNCTION auto &
  storage_ref()
  {
    return m_storage;
  }

  /**
   * \brief Gets the internal offset array
   */
  KOKKOS_INLINE_FUNCTION auto const
  offsets() const
  {
    return m_offsets;
  }

  /**
   * \brief Gets the internal offset array
   */
  KOKKOS_INLINE_FUNCTION auto
  offsets()
  {
    return m_offsets;
  }

  /**
   * \brief Gets the internal offset array
   */
  KOKKOS_INLINE_FUNCTION auto &
  offsets_ref()
  {
    return m_offsets;
  }

  // ==============================================================================================
  // ==============================================================================================
  // DataArrayGhostedBlock proxy functions
  // ==============================================================================================
  // ==============================================================================================

  /**
   * \brief Reshapes the inner block
   */
  KOKKOS_FORCEINLINE_FUNCTION void
  reshape(block_size_t<dim> const & new_ghosted_size, shift_t<dim> const & new_shift)
  {
    m_storage.reshape(new_ghosted_size, new_shift);
  }

  /**
   * \brief Resets the inner block back to its original shape
   */
  KOKKOS_FORCEINLINE_FUNCTION void
  shape_reset()
  {
    m_storage.shape_reset();
  }

  // ==============================================================================================
  // ==============================================================================================

  /**
   * \brief Memory access operator() with indices
   */
  template <typename I0,
            typename I1,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i_0, I1 i_1, IVar i_var, IOct i_oct) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_offsets.size() - 1) && "Wrong value for i_oct");
    KOKKOS_ASSERT((static_cast<int32_t>(i_var) < num_vars(i_oct)) && "Wrong value for i_var");
#endif

    return m_storage(i_0, i_1, 0, flat_mv_block_index(i_var, i_oct, m_offsets));
  }

  /**
   * \brief Memory access operator() with indices
   */
  template <typename I0,
            typename I1,
            typename I2,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i_0, I1 i_1, I2 i_2, IVar i_var, IOct i_oct) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_offsets.size() - 1) && "Wrong value for i_oct");
    KOKKOS_ASSERT((static_cast<int32_t>(i_var) < num_vars(i_oct)) && "Wrong value for i_var");
#endif

    return m_storage(i_0, i_1, i_2, 0, flat_mv_block_index(i_var, i_oct, m_offsets));
  }

  /**
   * \brief Memory access operator() with cell index
   */
  template <typename ICell, typename IVar, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(ICell i_cell, IVar i_var, IOct i_oct) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_offsets.size() - 1) && "Wrong value for i_oct");
    KOKKOS_ASSERT((static_cast<int32_t>(i_var) < num_vars(i_oct)) && "Wrong value for i_var");
#endif

    return m_storage(i_cell, 0, flat_mv_block_index(i_var, i_oct, m_offsets));
  }

  /**
   * \brief Memory access operator() with cell index and block index
   */
  template <typename ICell, typename IBlk>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(ICell i_cell, IBlk i_blk) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_blk) < m_storage.num_quadrants()) &&
                  "Wrong value for i_blk");
#endif

    return m_storage(i_cell, 0, i_blk);
  }

  // ==============================================================================================
  // ==============================================================================================

  auto
  label() const
  {
    return m_storage.label();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto const
  block_size() const
  {
    return m_storage.block_size();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto const
  ghost_block_size() const
  {
    return m_storage.allocated_ghosted_block_size();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto const
  shape() const
  {
    return m_storage.shape();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto const
  shift() const
  {
    return m_storage.shift();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_cells() const
  {
    return m_storage.num_cells();
  }

  template <typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION auto
  num_vars(IOct i_oct) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_offsets.size() - 1) && "Wrong value for i_oct");
#endif

    return static_cast<int32_t>(m_offsets(i_oct + 1) - m_offsets(i_oct));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_quadrants() const
  {
    return static_cast<int32_t>(m_offsets.size()) - 1;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_blocks() const
  {
    return m_storage.num_quadrants();
  }

  auto
  allocated_size_in_bytes() const
  {
    return m_offsets.size() * sizeof(typename Offsets_t::value_type) +
           m_storage.allocated_size_in_bytes();
  }

  /**
   * \brief Computes the offset view from an array of number of variables. It simply computes a
   * postfix sum with the total sum at the end.
   */
  int32_t
  compute_offsets(const NumVars_t num_vars)
  {
    m_offsets.resize(num_vars.size() + 1);

    // avoid capturing class member in lambda
    auto offsets = m_offsets;

    Kokkos::RangePolicy<typename device_t::execution_space> policy(0, m_offsets.size());
    uint32_t                                                num_blocks = 0;
    Kokkos::parallel_scan(
      "kalypsso::DataArrayBlockMultiVar::offsets",
      policy,
      KOKKOS_LAMBDA(const uint32_t i_var, uint32_t & offset, const bool is_final) {
        if (is_final)
          offsets(i_var) = offset;
        if (i_var != num_vars.size())
          offset += num_vars(i_var);
      },
      num_blocks);

    return static_cast<int32_t>(num_blocks);
  }

  /**
   * \brief Computes the offset view from a MaterialPresenceView. A "number of variables per
   * material" is still needed but it simply computes a postfix sum with the total sum at the end.
   */
  int32_t
  compute_offsets(const MatPresView_t mat_pres, const uint32_t num_vars_per_mat)
  {
    m_offsets.resize(mat_pres.size() + 1);

    // avoid capturing class member in lambda
    auto offsets = m_offsets;

    Kokkos::RangePolicy<typename device_t::execution_space> policy(0, m_offsets.size());
    uint32_t                                                num_blocks = 0;
    Kokkos::parallel_scan(
      "kalypsso::DataArrayBlockMultiVar::offsets",
      policy,
      KOKKOS_LAMBDA(const uint32_t i_var, uint32_t & offset, const bool is_final) {
        if (is_final)
          offsets(i_var) = offset;
        if (i_var != mat_pres.size())
          offset += static_cast<uint32_t>(mat_pres.num_materials(static_cast<int32_t>(i_var))) *
                    num_vars_per_mat;
      },
      num_blocks);

    return static_cast<int32_t>(num_blocks);
  }

private:
  //! Contains the actual data. Essentially, the handling of an octant's variables is done via the
  //! m_offsets member. So this structure has a single variable per block and an index flattener
  //! picks the correct block from the variable and octant indices.
  DataArrayGhostedBlock_t m_storage;

  //! Octant's variables are located at block index [ m_offsets(i) : m_offsets(i+1) ).
  Offsets_t m_offsets;

}; // class DataArrayGhostedBlockMultiVar

} // namespace kalypsso

#endif // KALYPSSO_CORE_DATA_ARRAY_BLOCK_MULTI_VAR_H_
