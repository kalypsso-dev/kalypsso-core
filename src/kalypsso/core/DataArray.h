// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DataArray.h
 */
#ifndef KALYPSSO_CORE_DATAARRAY_H_
#define KALYPSSO_CORE_DATAARRAY_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>

#include <kalypsso/core/enums.h>
#include <kalypsso/core/DataArrayUtils.h>

#include <kalypsso/utils/log/kalypsso_log.h>

#include <type_traits> // for std::conditional


namespace kalypsso
{

// ============================================================================================
// ============================================================================================
// ============================================================================================
/**
 * \class DataArray
 *
 * A helper class wrapping a 1d Kokkos::View with specific memory allocation strategy (capacity).
 */
template <typename T, typename device_t, StorageType st = StorageType::OWNED>
class DataArray
{
public:
  using DataArray_t = DataArray<T, device_t>;

  using FlatArrayOwned_t = Kokkos::View<T *, device_t>;
  using FlatArrayUnmanaged_t = Kokkos::View<T *, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using FlatArray_t = typename std::
    conditional<st == StorageType::OWNED, FlatArrayOwned_t, FlatArrayUnmanaged_t>::type;

  using view_t = FlatArray_t;
  using value_type = typename FlatArray_t::value_type;

  using host_mirror_t = typename FlatArray_t::host_mirror_type;

private:
  //! number of elements
  size_t m_num_elements;

  //! storage capacity.
  //! actual (physical) number of elements in m_storage
  //! it is increased upon resizing only when necessary
  size_t m_storage_capacity;

  //! storage array
  FlatArray_t m_storage;

public:
  DataArray() = default;

  ~DataArray() = default;

  //! this constructor is only allowed when we want a class that does own data
  template <StorageType st_ = st, std::enable_if_t<(st_ == StorageType::OWNED), bool> = true>
  DataArray(std::string name, size_t num_elements)
    : m_num_elements(num_elements)
    , m_storage_capacity(DataArrayUtils::allocated_capacity(num_elements))
    , m_storage(Kokkos::view_alloc(Kokkos::WithoutInitializing, name), m_storage_capacity)
  {}

  //! this constructor is only allowed when we want a class that doesn't own data
  template <StorageType st_ = st, std::enable_if_t<(st_ == StorageType::UNMANAGED), bool> = true>
  DataArray(T * ptr, size_t num_elements)
    : m_num_elements(num_elements)
    , m_storage_capacity(DataArrayUtils::allocated_capacity(static_cast<size_t>(num_elements)))
    , m_storage(ptr, m_storage_capacity)
  {}

  //! move constructor
  template <StorageType st_ = st, std::enable_if_t<(st_ == StorageType::OWNED), bool> = true>
  DataArray(size_t num_elements, FlatArray_t && some_data)
    : m_num_elements(num_elements)
    , m_storage_capacity(some_data.size())
    , m_storage(some_data)
  {}

  DataArray(const DataArray & other) = default;

  DataArray(DataArray && other) = default;

  DataArray &
  operator=(const DataArray & other) = default;

  DataArray &
  operator=(DataArray && other) = default;


  // ==================================================================================
  /**
   * create host mirror.
   */
  static auto
  create_host_mirror_view(DataArray_t src)
  {

    return DataArray<T, HostDevice>(src.num_elements(),
                                    Kokkos::create_mirror_view(src.physical_view()));
  }

  // ==================================================================================
  /**
   * create host mirror and copy.
   */
  static auto
  create_host_mirror_view_and_copy(DataArray_t src)
  {

    // auto storage_res = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
    // src.storage_ref());

    auto res = DataArray<T, HostDevice>(
      src.num_elements(),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, src.physical_view_ref()));
    return res;
  }

  // ==================================================================================
  //! memory access operator()
  template <typename I0>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<size_t>(i0) < m_num_elements) && "Wrong value for i0");
#endif

    return m_storage(i0);
  }

  // ==================================================================================
  auto
  label() const
  {
    return m_storage.label();
  }

  // ==================================================================================
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_elements() const
  {
    return m_num_elements;
  }

  // ==================================================================================
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  size() const
  {
    return static_cast<size_t>(this->num_elements());
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
    return m_num_elements;
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
  resize(size_t num_elements)
  {

    if constexpr (st == StorageType::OWNED)
    {
      m_num_elements = num_elements;

      // only resize when the requested new size is larger than capacity
      if (logical_size_in_elements() > m_storage_capacity)
      {
        size_t new_storage_capacity =
          DataArrayUtils::allocated_capacity(static_cast<size_t>(num_elements));

        Kokkos::resize(
          Kokkos::view_alloc(Kokkos::WithoutInitializing), m_storage, new_storage_capacity);

        m_storage_capacity = new_storage_capacity;
      }
    }
    else
    {
      // resizing is not allowed when the class doesn't own its data
      KALYPSSO_WARN("Attempting to resize an unmanaged DataArray. This is not allowed.");
    }

  } // DataArray::resize

  // ==================================================================================
  //! resize (by re-allocating) and initialize to zero
  void
  resize_and_reset(size_t num_elements)
  {

    resize(num_elements);
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

}; // class DataArray

} // namespace kalypsso

#endif // KALYPSSO_CORE_DATAARRAY_H_
