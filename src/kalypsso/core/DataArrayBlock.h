// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DataArrayBlock.h
 */
#ifndef KALYPSSO_CORE_DATAARRAYBLOCK_H_
#define KALYPSSO_CORE_DATAARRAYBLOCK_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>

#include <kalypsso/core/utils_block.h> // for definition of type block_size_t and shift_t
#include <kalypsso/core/enums.h>
#include <kalypsso/core/DataArrayUtils.h>

#include <kalypsso/utils/log/kalypsso_log.h>

#include <type_traits> // for std::conditional


namespace kalypsso
{

enum KokkosLayout
{
  KOKKOS_LAYOUT_LEFT,
  KOKKOS_LAYOUT_RIGHT
};

/**
 * DataArrayLeafSoA is used to store data attached to the leaves of the octrees.
 *
 * first index is leaf id (curvilinear index along the Morton curve)
 * last index is some scalar index
 *
 * The class name is suffixed by SoA (Structure of Array) because the "fast" index is the first
 * index, that is, the leaf id (structure of array of leaf index)
 */
template <typename T, typename device_t>
using DataArrayLeafSoA = Kokkos::View<T **, Kokkos::LayoutLeft, device_t>;

template <typename T, typename device_t>
using DataArrayLeafSoAUnmanaged =
  Kokkos::View<T **, Kokkos::LayoutLeft, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T, typename device_t>
using DataArrayHostLeafSoA = typename DataArrayLeafSoA<T, device_t>::host_mirror_type;

/**
 * DataArrayLeafAoS is used to store data attached to the leaves of the octrees.
 *
 * first index is leaf id (curvilinear index along the Morton curve)
 * last index is some scalar index
 *
 * The class name is suffixed by AoS (Array of Structure) because the "fast" index is the last
 * index, that is, is variable index (array indexed by leaf index of a structure containing nvar
 * scalar values).
 */
template <typename T, typename device_t>
using DataArrayLeafAoS = Kokkos::View<T **, Kokkos::LayoutRight, device_t>;

template <typename T, typename device_t>
using DataArrayLeafAoSUnmanaged = Kokkos::View<T **,
                                               Kokkos::LayoutRight,
                                               typename device_t::memory_space,
                                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T, typename device_t>
using DataArrayHostLeafAoS = typename DataArrayLeafAoS<T, device_t>::host_mirror_type;

/**
 * DataArrayLeaf mostly used when using one cell per octree leaf.
 *
 * first index is leaf id (curvilinear index along the Morton curve)
 * last index is hydro variable
 */
template <typename T, typename device_t>
using DataArrayLeaf = DataArrayLeafAoS<T, device_t>;

template <typename T, typename device_t>
using DataArrayLeafUnmanaged = DataArrayLeafAoSUnmanaged<T, device_t>;

template <typename T, typename device_t>
using DataArrayLeafHost = typename DataArrayLeaf<T, device_t>::host_mirror_type;

/**
 * DataArrayBlockLegacy used when designing a solver with a block of
 * data per leaf of the octree.
 *
 * \note this class is deprecated, please use DataArrayBlock instead.
 *
 * first index identifies a cell inside a block (left layout, from 0 to bx by -1)
 * second index identifies the variable (rho, momentum, energy, ...)
 * third index is the leaf id (curvilinear index along the Morton curve)
 *
 * Note that we enforce Left layout here, since we plan to the Kokkos TeamPolicy with one team per
 * leaf, so we favor memory locality inside a block.
 */
template <typename T, typename device_t>
using DataArrayBlockLegacy = Kokkos::View<T ***, Kokkos::LayoutLeft, device_t>;

template <typename T, typename device_t>
using DataArrayBlockLegacyUnmanaged =
  Kokkos::View<T ***, Kokkos::LayoutLeft, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T, typename device_t>
using DataArrayBlockLegacyHost = typename DataArrayBlockLegacy<T, device_t>::host_mirror_type;

// ============================================================================================
// ============================================================================================
/**
 * Decode i,j,k from a flat cell index.
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
cell_index_unravel(int32_t cell_index, block_size_t<dim> const & shape)
{
  coord_t<dim> ijk;

  if constexpr (dim == 2)
  {
    // cell_index = i + shape[IX] * j
    ijk[IY] = cell_index / shape[IX];
    ijk[IX] = cell_index - shape[IX] * ijk[IY];
  }
  else if (dim == 3)
  {
    // cell_index = i + shape[IX] * j + shape[IX] * shape[IY] * k
    ijk[IZ] = cell_index / (shape[IX] * shape[IY]);
    cell_index -= ijk[IZ] * shape[IX] * shape[IY];

    ijk[IY] = cell_index / shape[IX];
    ijk[IX] = cell_index - shape[IX] * ijk[IY];
  }

  return ijk;

} // cell_index_unravel

// ============================================================================================
// ============================================================================================
/**
 * Decode i,j,k,ivar,iOct from a flat index (into a DataBlockArray)
 */
template <size_t dim>
KOKKOS_FORCEINLINE_FUNCTION auto
flat_index_unravel(int64_t flat_index, block_size_t<dim> const & shape, int32_t const & num_vars)
{

  block_multiindex_t<dim> index;

  const auto num_cells = Kokkos::dim_prod(shape);

  // iOct
  index[dim + 1] = flat_index / (num_cells * num_vars);
  flat_index -= index[dim + 1] * (num_cells * num_vars);

  // ivar
  index[dim] = flat_index / num_cells;

  // cell index
  flat_index -= index[dim] * num_cells;

  if constexpr (dim == 2)
  {
    index[IY] = flat_index / shape[IX];
    index[IX] = flat_index - shape[IX] * index[IY];
  }
  else if constexpr (dim == 3)
  {
    index[IZ] = flat_index / (shape[IY] * shape[IX]);
    flat_index -= index[IZ] * shape[IY] * shape[IX];

    index[IY] = flat_index / shape[IX];
    index[IX] = flat_index - shape[IX] * index[IY];
  }

  return index;

} // flat_index_unravel

// ============================================================================================
// ============================================================================================
// ============================================================================================
/**
 * \class DataArrayBlock
 *
 * A helper class around a 1d Kokkos::View providing different left-layout multi-dimensional data
 * access.
 */
template <size_t dim, typename T, typename device_t, StorageType st = StorageType::OWNED>
class DataArrayBlock
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, T, device_t>;

  using FlatArrayOwned_t = Kokkos::View<T *, device_t>;
  using FlatArrayUnmanaged_t = Kokkos::View<T *, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using FlatArray_t = typename std::
    conditional<st == StorageType::OWNED, FlatArrayOwned_t, FlatArrayUnmanaged_t>::type;

  using host_mirror_t = typename FlatArray_t::host_mirror_type;

private:
  //! block size (used for memory allocation). Can only be changed by resizing (re-allocating
  //! without copy)
  block_size_t<dim> m_bSize;

  //! array shape (used for memory addressing). Initially and after resizing, m_shape and m_bSize
  //! are equal, but shape can be change by reshaping (no memory reallocation, no data modification,
  //! just changing index linearization to access memory differently).
  //! Example application: a 2d array of sizes N x (N+1) can be reshaped into (N+1) x N
  //! current limitation: we only support reshaping the block size (not the number of variables, or
  //! the number of quadrants. => TODO if it can be useful to some application).
  block_size_t<dim> m_shape;

  //! num cells
  int32_t m_num_cells;

  //! num of physical values
  int32_t m_num_vars;

  //! number of quadrants
  int32_t m_num_quadrants;

  //! storage capacity (should be larger or equal to num_cells * num_vars * num_quadrants).
  //! actual (physical) number of elements in m_storage
  //! it is increased upon resizing only when necessary
  size_t m_storage_capacity;

  //! storage array
  FlatArray_t m_storage;

public:
  DataArrayBlock() = default;

  //! this constructor is only allowed when we want a class that does own data
  template <StorageType st_ = st, std::enable_if_t<(st_ == StorageType::OWNED), bool> = true>
  DataArrayBlock(std::string name, block_size_t<dim> bSize, int32_t num_vars, int32_t num_quadrants)
    : m_bSize(bSize)
    , m_shape(bSize)
    , m_num_cells(Kokkos::dim_prod(bSize))
    , m_num_vars(num_vars)
    , m_num_quadrants(num_quadrants)
    , m_storage_capacity(DataArrayUtils::allocated_capacity(Kokkos::dim_prod(bSize) * num_vars *
                                                            static_cast<size_t>(num_quadrants)))
    , m_storage(Kokkos::view_alloc(Kokkos::WithoutInitializing, name), m_storage_capacity)
  {}

  //! this constructor is only allowed when we want a class that doesn't own data
  template <StorageType st_ = st, std::enable_if_t<(st_ == StorageType::UNMANAGED), bool> = true>
  DataArrayBlock(T * ptr, block_size_t<dim> bSize, int32_t num_vars, int32_t num_quadrants)
    : m_bSize(bSize)
    , m_shape(bSize)
    , m_num_cells(Kokkos::dim_prod(bSize))
    , m_num_vars(num_vars)
    , m_num_quadrants(num_quadrants)
    , m_storage_capacity(DataArrayUtils::allocated_capacity(Kokkos::dim_prod(bSize) * num_vars *
                                                            static_cast<size_t>(num_quadrants)))
    , m_storage(ptr, m_storage_capacity)
  {}

  //! move constructor
  template <StorageType st_ = st, std::enable_if_t<(st_ == StorageType::OWNED), bool> = true>
  DataArrayBlock(block_size_t<dim> bSize,
                 int32_t           num_vars,
                 int32_t           num_quadrants,
                 FlatArray_t &&    some_data)
    : m_bSize(bSize)
    , m_shape(bSize)
    , m_num_cells(Kokkos::dim_prod(bSize))
    , m_num_vars(num_vars)
    , m_num_quadrants(num_quadrants)
    , m_storage_capacity(some_data.size())
    , m_storage(some_data)
  {}

  // ==================================================================================
  /**
   * create host mirror.
   */
  static auto
  create_host_mirror_view(DataArrayBlock_t src)
  {

    return DataArrayBlock<dim, T, HostDevice>(src.block_size(),
                                              src.num_vars(),
                                              src.num_quadrants(),
                                              Kokkos::create_mirror_view(src.physical_view()));
  }

  // ==================================================================================
  /**
   * create host mirror and copy.
   */
  static auto
  create_host_mirror_view_and_copy(DataArrayBlock_t src)
  {

    // auto storage_res = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
    // src.storage_ref());

    auto res = DataArrayBlock<dim, T, HostDevice>(
      src.block_size(),
      src.num_vars(),
      src.num_quadrants(),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, src.physical_view_ref()));
    return res;
  }

  // ==================================================================================
  /**
   * Return true when current shape is equal to size set by constructor.
   */
  KOKKOS_FORCEINLINE_FUNCTION bool
  are_shape_and_size_equal() const
  {
    return m_bSize == m_shape;
  }

  /**
   * Reshape DataArrayBlock without reallocating.
   *
   * We only accept reshaping when the new shape (block sizes) is compatible current memory
   * allocation.
   *
   * \return true when DataArrayBlock is actually reshaped.
   */
  KOKKOS_FORCEINLINE_FUNCTION bool
  reshape(block_size_t<dim> const & new_shape)
  {
    // when reshaping, as we don't want to reallocate, we need to make sure the new
    // shape is not larger than current allocation
    // KOKKOS_ASSERT(Kokkos::dim_prod(new_shape) <= Kokkos::dim_prod(m_bSize) &&
    //               "Can't reshape DataArrayBlock, new shape is too large.");

    // only reshape when possible
    if (Kokkos::dim_prod(new_shape) <= Kokkos::dim_prod(m_bSize))
    {
      m_shape = new_shape;
      m_num_cells = Kokkos::dim_prod(new_shape);
      return true;
    }
    return false;
  }

  /**
   * Restore shape to be equal to the original shape set by constructor.
   */
  KOKKOS_FORCEINLINE_FUNCTION void
  shape_reset()
  {
    m_shape = m_bSize;
    m_num_cells = Kokkos::dim_prod(m_shape);
  }

  // ==================================================================================
  /**
   * convert a logical multi-index (icell, ivar, iOct) into a flat index to storage (2D).
   */
  template <typename ICell, typename IVar, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION int64_t
  flat_index(ICell icell, IVar ivar, IOct iOct) const
  {
    return icell + m_num_cells * (ivar + m_num_vars * iOct);
  }

  // ==================================================================================
  /**
   * convert a logical multi-index (i, j, ivar, iOct) into a flat index to storage (2D).
   */
  template <typename I0,
            typename I1,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION int64_t
  flat_index(I0 i0, I1 i1, IVar ivar, IOct iOct) const
  {
    return i0 + m_shape[IX] * (i1 + m_shape[IY] * (ivar + m_num_vars * iOct));
  }

  // ==================================================================================
  /**
   * convert a logical multi-index (i, j, k, ivar, iOct) into a flat index to storage (3D).
   */
  template <typename I0,
            typename I1,
            typename I2,
            typename IVar,
            typename IOct,
            size_t dim_ = dim,
            std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_FORCEINLINE_FUNCTION int64_t
  flat_index(I0 i0, I1 i1, I2 i2, IVar ivar, IOct iOct) const
  {
    return i0 + m_shape[IX] * (i1 + m_shape[IY] * (i2 + m_shape[IZ] * (ivar + m_num_vars * iOct)));
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
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<int32_t>(i0) < m_shape[IX]) && "Wrong value for i0");
    KOKKOS_ASSERT((static_cast<int32_t>(i1) < m_shape[IY]) && "Wrong value for i1");
    KOKKOS_ASSERT((static_cast<int32_t>(ivar) < m_num_vars) && "Wrong value for ivar");
    KOKKOS_ASSERT((static_cast<int32_t>(iOct) < m_num_quadrants) && "Wrong value for iOct");
#endif

    return m_storage(flat_index(i0, i1, ivar, iOct));
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
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<int32_t>(i0) < m_shape[IX]) && "Wrong value for i0");
    KOKKOS_ASSERT((static_cast<int32_t>(i1) < m_shape[IY]) && "Wrong value for i1");
    KOKKOS_ASSERT((static_cast<int32_t>(i2) < m_shape[IZ]) && "Wrong value for i2");
    KOKKOS_ASSERT((static_cast<int32_t>(ivar) < m_num_vars) && "Wrong value for ivar");
    KOKKOS_ASSERT((static_cast<int32_t>(iOct) < m_num_quadrants) && "Wrong value for iOct");
#endif

    return m_storage(flat_index(i0, i1, i2, ivar, iOct));
  }

  // ==================================================================================
  //! memory access operator() using a flat cell index (2d and 3d)
  template <typename ICell, typename IVar, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(ICell icell, IVar ivar, IOct iOct) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<int32_t>(icell) < m_num_cells) && "Wrong value for icell");
    KOKKOS_ASSERT((static_cast<int32_t>(ivar) < m_num_vars) && "Wrong value for ivar");
    KOKKOS_ASSERT((static_cast<int32_t>(iOct) < m_num_quadrants) && "Wrong value for iOct");
#endif

    return m_storage(flat_index(icell, ivar, iOct));
  }

  // ==================================================================================
  //! memory access operator() using a multi-index (2d and 3d)
  template <typename IVar, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(coord_t<dim> ijk, IVar ivar, IOct iOct) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<int32_t>(ijk[IX]) < m_shape[IX]) && "Wrong value for ijk[IX]");
    KOKKOS_ASSERT((static_cast<int32_t>(ijk[IY]) < m_shape[IY]) && "Wrong value for ijk[IY]");
    if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((static_cast<int32_t>(ijk[IZ]) < m_shape[IZ]) && "Wrong value for ijk[IZ]");
    }
    KOKKOS_ASSERT((static_cast<int32_t>(ivar) < m_num_vars) && "Wrong value for ivar");
    KOKKOS_ASSERT((static_cast<int32_t>(iOct) < m_num_quadrants) && "Wrong value for iOct");
#endif

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
  //! return true this DataArrayBlock can be considered a flux array.
  //!
  //! being a flux array means current shape has one extra sizes in one dimension when compared to
  //! input bSizes.
  KOKKOS_FORCEINLINE_FUNCTION bool
  is_flux_array(block_size_t<dim> const & bSizes, int dir) const
  {

    if constexpr (dim == 2)
    {
      KOKKOS_ASSERT((dir == IX or dir == IY) && "Wrong direction");

      return ((dir == IX) and (m_shape[IX] == bSizes[IX] + 1) and (m_shape[IY] == bSizes[IY])) or
             ((dir == IY) and (m_shape[IY] == bSizes[IY] + 1) and (m_shape[IX] == bSizes[IX]));
    }
    else if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) && "Wrong direction");
      // clang-format off
      return ((dir == IX) and (m_shape[IX] == bSizes[IX] + 1) and
                              (m_shape[IY] == bSizes[IY]    ) and
                              (m_shape[IZ] == bSizes[IZ]    )     ) or
             ((dir == IY) and (m_shape[IX] == bSizes[IX]    ) and
                              (m_shape[IY] == bSizes[IY] + 1) and
                              (m_shape[IZ] == bSizes[IZ]    )     ) or
             ((dir == IZ) and (m_shape[IX] == bSizes[IX]    ) and
                              (m_shape[IY] == bSizes[IY]    ) and
                              (m_shape[IZ] == bSizes[IZ] + 1)     );
      // clang-format on
    }
    return false;

  } // is_flux_array

  // ==================================================================================
  auto
  label() const
  {
    return m_storage.label();
  }

  // ==================================================================================
  KOKKOS_FORCEINLINE_FUNCTION
  auto const &
  block_size() const
  {
    return m_bSize;
  }

  // ==================================================================================
  KOKKOS_FORCEINLINE_FUNCTION
  auto const &
  shape() const
  {
    return m_shape;
  }

  // ==================================================================================
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_cells() const
  {
    return m_num_cells;
  }

  // ==================================================================================
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_vars() const
  {
    return m_num_vars;
  }

  // ==================================================================================
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_quadrants() const
  {
    return m_num_quadrants;
  }

  // ==================================================================================
  //! Access to raw pointer
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  data() const
  {
    return m_storage.data();
  }

  // ==================================================================================
  //! return internal Kokkos::View used for raw storage.
  //! A regular user should probably never have to use the physical, but most surely the
  //! logical_view (with logical sizes).
  auto
  physical_view() -> decltype(m_storage)
  {
    return m_storage;
  }

  // ==================================================================================
  auto
  physical_view_ref() -> decltype(m_storage) &
  {
    return m_storage;
  }

  // ==================================================================================
  //! return logical size (total number of elements). Not to be confused with capacity (physical
  //! size)
  auto
  logical_size_in_elements() const
  {
    return m_num_cells * m_num_vars * static_cast<size_t>(m_num_quadrants);
  }

  // ==================================================================================
  auto
  logical_view() const
  {
    const auto logical_range =
      std::pair<std::size_t, std::size_t>(0, this->logical_size_in_elements());
    return Kokkos::subview(m_storage, logical_range);
  }

  // ==================================================================================
  void
  resize(block_size_t<dim> bSize, int32_t num_vars, int32_t num_quadrants)
  {
    if constexpr (st == StorageType::OWNED)
    {
      m_bSize = bSize;
      m_shape = bSize;
      m_num_cells = Kokkos::dim_prod(bSize);
      m_num_vars = num_vars;
      m_num_quadrants = num_quadrants;

      // only resize when the requested new size is larger than capacity
      if (logical_size_in_elements() > m_storage_capacity)
      {
        size_t new_storage_capacity = DataArrayUtils::allocated_capacity(
          Kokkos::dim_prod(bSize) * num_vars * static_cast<size_t>(num_quadrants));

        Kokkos::resize(
          Kokkos::view_alloc(Kokkos::WithoutInitializing), m_storage, new_storage_capacity);

        m_storage_capacity = new_storage_capacity;
      }
    }
    else
    {
      // resizing is not allowed when the class doesn't own its data
      KALYPSSO_WARN("Attempting to resize an unmanaged DataArrayBlock. This is not allowed.");
    }
  }

  // ==================================================================================
  //! resize (by re-allocating) but do not initialize
  void
  resize(int32_t num_quadrants)
  {
    m_num_quadrants = num_quadrants;

    // only resize when the requested new size is larger than capacity
    if (logical_size_in_elements() > m_storage_capacity)
    {
      size_t new_storage_capacity = DataArrayUtils::allocated_capacity(
        m_num_cells * m_num_vars * static_cast<size_t>(num_quadrants));

      Kokkos::resize(
        Kokkos::view_alloc(Kokkos::WithoutInitializing), m_storage, new_storage_capacity);
      m_storage_capacity = new_storage_capacity;
    }
  } // resize

  // ==================================================================================
  //! resize (by re-allocating) and initialize to zero
  void
  resize_and_reset(int32_t num_quadrants)
  {
    resize(num_quadrants);
    Kokkos::deep_copy(m_storage, 0);
  } // resize_and_reset

  // ==================================================================================
  //! Return the total allocated memory in bytes.
  auto
  allocated_size_in_bytes() const
  {
    if constexpr (st == StorageType::OWNED)
    {
      uint64_t size = m_storage.extent(0) * sizeof(T);
      return size;
    }
    return static_cast<uint64_t>(0);
  }

}; // class DataArrayBlock

} // namespace kalypsso

#endif // KALYPSSO_CORE_DATAARRAYBLOCK_H_
