// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MaterialPresence.h
 */

#ifndef KALYPSSO_CORE_MATERIAL_PRESENCE_H_
#define KALYPSSO_CORE_MATERIAL_PRESENCE_H_

#include <Kokkos_Core.hpp>
#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/DataArrayUtils.h>

namespace kalypsso
{

/**
 * \class MaterialPresenceView
 *
 * \brief Indicates the presence of a material inside an octant
 *
 * A MaterialPresenceView is essentially a wrapper around a view of bitsets with extra metadata.
 * Each bit in the bitsets correspond to a material and is set to 0 if it is not present and 1 if it
 * is.
 *
 * For example, if we have the following setup:
 *
 *           |
 *  Oct 2    | Oct 3
 *  Mat 1    | Mat 0
 *           |
 * ----------+----------
 *           |
 *  Oct 0    | Oct 1
 *  Mat 1, 2 | Mat 0, 2
 *           |
 *
 * Then the object will look like this (if the problem contains 3 materials total)
 *
 *  Oct 0 | Oct 1 | Oct 2 | Oct 3
 * -------+-------+-------+-------
 *  1 1 0 | 1 0 1 | 0 1 0 | 0 0 1
 *
 * Names:
 * - Material Number (or `mat_num`) is the absolute index of the material in the system. It's the
 *   numbers used just above.
 * - Material Index (or `mat_index`) is the local index of the material inside an octant. For
 *   example, the Material Index of material 2 in octant 0 or 1 is 2 and is undefined in octant 2
 *   or 3.
 *
 * \tparam device_t Where the inner array is located
 */
template <typename device_t>
class MaterialPresenceView
{
public:
  using BitBlock_t = uint64_t;
  using View_t = Kokkos::View<BitBlock_t *, device_t>;
  using MaterialPresenceView_t = MaterialPresenceView<device_t>;

  /**
   * \brief Creates a new MaterialPresenceView
   */
  MaterialPresenceView(const std::string & name, const uint32_t max_mat, const uint32_t num_octs)
    : m_max_mat(max_mat)
    , m_len_per_oct((max_mat / NUM_BITS_PER_BLOCK) + (max_mat % NUM_BITS_PER_BLOCK != 0))
    , m_num_octants(num_octs)
    , m_mat_pres(name,
                 DataArrayUtils::allocated_capacity(m_len_per_oct * static_cast<size_t>(num_octs)))
  {}

  /**
   * \brief Default constructor
   */
  MaterialPresenceView() = default;

  /**
   * \brief Move constructor
   */
  MaterialPresenceView(const uint32_t max_mat,
                       const uint32_t len_per_oct,
                       const uint32_t num_octs,
                       View_t &&      mat_pres)
    : m_max_mat(max_mat)
    , m_len_per_oct(len_per_oct)
    , m_num_octants(num_octs)
    , m_mat_pres(mat_pres)
  {}

  /**
   * \brief Creates host mirror
   */
  static auto
  create_host_mirror_view(MaterialPresenceView_t src)
  {
    return MaterialPresenceView<HostDevice>(src.m_max_mat,
                                            src.m_len_per_oct,
                                            src.m_num_octants,
                                            Kokkos::create_mirror_view(src.m_mat_pres));
  }

  /**
   * \brief Creates host mirror and copy
   */
  static auto
  create_host_mirror_view_and_copy(MaterialPresenceView_t src)
  {
    return MaterialPresenceView<HostDevice>(
      src.m_max_mat,
      src.m_len_per_oct,
      src.m_num_octants,
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, src.m_mat_pres));
  }

  // ==================================================================================
  /**
   * \brief Access to raw pointer of internal data
   */
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  data() const
  {
    return m_mat_pres.data();
  }

  /**
   * \brief Gets the internal view (raw)
   */
  auto
  physical_view()
  {
    return m_mat_pres;
  }

  /**
   * \brief Gets the internal view (raw)
   */
  KOKKOS_INLINE_FUNCTION auto const
  physical_view() const
  {
    return m_mat_pres;
  }

  // ==================================================================================
  //! return logical size (total number of elements). Not to be confused with capacity (physical
  //! size)
  auto
  logical_size_in_elements() const
  {
    return m_len_per_oct * static_cast<size_t>(m_num_octants);
  }

  // ==================================================================================
  //! return internal storage as a Kokkos::view sized upon the logical number of octant
  auto
  logical_view() const
  {
    const auto logical_range =
      std::pair<std::size_t, std::size_t>(0, this->logical_size_in_elements());
    return Kokkos::subview(m_mat_pres, logical_range);
  }

  /**
   * \brief Zeroes out the material presence
   */
  void
  resets() const
  {
    Kokkos::deep_copy(m_mat_pres, 0);
  }

  /**
   * \brief Return inner array capacity.
   */
  KOKKOS_INLINE_FUNCTION auto
  capacity() const
  {
    return m_mat_pres.size();
  }

  /**
   * \brief Resizes the inner array.
   *
   * \note Memory will be re-allocated only the new requested size is larger than capacity,
   * otherwise we just update the (logical) number of octants.
   */
  void
  resize(const int32_t num_octs)
  {
    m_num_octants = static_cast<uint32_t>(num_octs);

    if (logical_size_in_elements() > capacity())
    {
      size_t new_storage_capacity =
        DataArrayUtils::allocated_capacity(m_len_per_oct * static_cast<size_t>(m_num_octants));

      Kokkos::resize(m_mat_pres, new_storage_capacity);
    }
  }

  /**
   * \brief Gets the inner size in bytes
   */
  auto
  allocated_size_in_bytes() const
  {
    return capacity() * sizeof(typename View_t::value_type);
  }

  /**
   * \brief Gets the number of octants
   */
  KOKKOS_INLINE_FUNCTION
  auto
  size() const
  {
    return m_num_octants;
  }

  /**
   * \brief Gets the maximum number of materials
   */
  KOKKOS_INLINE_FUNCTION
  auto
  max_mat() const
  {
    return m_max_mat;
  }

  /**
   * \brief Gets the label
   */
  auto
  label() const
  {
    return m_mat_pres.label();
  }

  /**
   * \brief Gets the number of bit blocks per octant
   */
  auto
  block_length_per_octant() const
  {
    return static_cast<int32_t>(m_len_per_oct);
  }

  /**
   * \brief Returns true if the material number is considered present
   */
  KOKKOS_INLINE_FUNCTION bool
  get(int32_t i_oct, int32_t mat_num) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(mat_num) < m_max_mat) && "Wrong value for mat_num");
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_num_octants) && "Wrong value for i_oct");
#endif

    return (m_mat_pres(var(i_oct, mat_num)) & bit(mat_num)) != 0;
  }

  /**
   * \brief Sets the material presence
   */
  KOKKOS_INLINE_FUNCTION void
  set(int32_t i_oct, int32_t mat_num) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(mat_num) < m_max_mat) && "Wrong value for mat_num");
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_num_octants) && "Wrong value for i_oct");
#endif

    m_mat_pres(var(i_oct, mat_num)) |= bit(mat_num);
  }

  /**
   * \brief Unsets the material presence
   */
  KOKKOS_INLINE_FUNCTION void
  unset(int32_t i_oct, int32_t mat_num) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(mat_num) < m_max_mat) && "Wrong value for mat_num");
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_num_octants) && "Wrong value for i_oct");
#endif

    m_mat_pres(var(i_oct, mat_num)) &= ~bit(mat_num);
  }

  /**
   * \brief Returns the index of the material number. It is supposed that get(i_oct, mat_num) ==
   * true.
   *
   * It essentially counts the number of bits set before (and excluding) the material number's bit.
   */
  KOKKOS_INLINE_FUNCTION int32_t
  material_index(int32_t i_oct, int32_t mat_num) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(mat_num) < m_max_mat) && "Wrong value for mat_num");
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_num_octants) && "Wrong value for i_oct");
    KOKKOS_ASSERT((get(i_oct, mat_num)) && "Wrong value for mat_num");
#endif

    return material_index_unchecked(i_oct, mat_num);
  }

  /**
   * \brief Returns the material number at the material index for said octant. It is guaranteed that
   * get(i_oct, material_num(n)) == true. Is the inverse of 'material_index'
   *
   * It essentially counts the number of bits set and the index until we reach 'mat_index'
   */
  KOKKOS_INLINE_FUNCTION int32_t
  material_num(int32_t i_oct, int32_t mat_index) const
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((static_cast<uint32_t>(i_oct) < m_num_octants) && "Wrong value for i_oct");
    KOKKOS_ASSERT((mat_index < num_materials(i_oct)) && "Wrong value for mat_index");
#endif

    int32_t        total = 0;
    const uint32_t start_var = var(i_oct, 0);
    uint32_t       nvar = start_var;

    // We count the number of '1' until we get over the material index
    for (; total <= mat_index && nvar < start_var + m_len_per_oct; nvar++)
      total += bit_block_popcount(m_mat_pres(nvar));

    auto bit_block = m_mat_pres(nvar - 1);
    total -= bit_block_popcount(bit_block);

    // Then we count until we have our number
    for (uint8_t i = 0; i < NUM_BITS_PER_BLOCK; i++)
      if (total + bit_block_popcount(bit_block, i + 1) > mat_index)
        return static_cast<int32_t>(((nvar - 1 - start_var) * NUM_BITS_PER_BLOCK) + i);

    // Failsafe
    return static_cast<int32_t>(m_max_mat);
  }

  /**
   * \brief Returns the total number of materials
   *
   * Simply counts the total number of set bits
   */
  KOKKOS_INLINE_FUNCTION auto
  num_materials(const int32_t i_oct) const
  {
    // Index of last material + 1
    return material_index_unchecked(i_oct, static_cast<int32_t>(m_max_mat));
  }

  /**
   * \brief Updates the left set at octant with the other one
   */
  KOKKOS_INLINE_FUNCTION static void
  update(const MaterialPresenceView_t & lhs,
         const int32_t                  l_oct,
         const MaterialPresenceView_t & rhs,
         const int32_t                  r_oct)
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((lhs.m_len_per_oct && rhs.m_len_per_oct) && "Wrong value for lhs and rhs");
    KOKKOS_ASSERT((static_cast<uint32_t>(l_oct) * lhs.m_len_per_oct < lhs.m_mat_pres.size()) &&
                  "Wrong value for l_oct");
    KOKKOS_ASSERT((static_cast<uint32_t>(r_oct) * rhs.m_len_per_oct < rhs.m_mat_pres.size()) &&
                  "Wrong value for r_oct");
#endif

    for (size_t i = 0; i < lhs.m_len_per_oct; i++)
      lhs.m_mat_pres(lhs.var(l_oct, 0) + i) |= rhs.m_mat_pres(rhs.var(r_oct, 0) + i);
  }

  /**
   * \brief Restricts the left set at octant with the other one
   */
  KOKKOS_INLINE_FUNCTION static void
  do_restriction(const MaterialPresenceView_t & lhs,
                 const int32_t                  l_oct,
                 const MaterialPresenceView_t & rhs,
                 const int32_t                  r_oct)
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((lhs.m_len_per_oct && rhs.m_len_per_oct) && "Wrong value for lhs and rhs");
    KOKKOS_ASSERT((static_cast<uint32_t>(l_oct) * lhs.m_len_per_oct < lhs.m_mat_pres.size()) &&
                  "Wrong value for l_oct");
    KOKKOS_ASSERT((static_cast<uint32_t>(r_oct) * rhs.m_len_per_oct < rhs.m_mat_pres.size()) &&
                  "Wrong value for r_oct");
#endif

    for (size_t i = 0; i < lhs.m_len_per_oct; i++)
      lhs.m_mat_pres(lhs.var(l_oct, 0) + i) &= rhs.m_mat_pres(rhs.var(r_oct, 0) + i);
  }

  /**
   * \brief Copies the right set at octant on the other one
   */
  KOKKOS_INLINE_FUNCTION static void
  copy(const MaterialPresenceView_t & lhs,
       const int32_t                  l_oct,
       const MaterialPresenceView_t & rhs,
       const int32_t                  r_oct)
  {
#ifdef KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
    KOKKOS_ASSERT((lhs.m_len_per_oct && rhs.m_len_per_oct) && "Wrong value for lhs and rhs");
    KOKKOS_ASSERT((static_cast<uint32_t>(l_oct) * lhs.m_len_per_oct < lhs.m_mat_pres.size()) &&
                  "Wrong value for l_oct");
    KOKKOS_ASSERT((static_cast<uint32_t>(r_oct) * rhs.m_len_per_oct < rhs.m_mat_pres.size()) &&
                  "Wrong value for r_oct");
#endif

    for (size_t i = 0; i < lhs.m_len_per_oct; i++)
      lhs.m_mat_pres(lhs.var(l_oct, 0) + i) = rhs.m_mat_pres(rhs.var(r_oct, 0) + i);
  }


  /**
   * \brief Helper that computes the num_var array (then given to DataArrayBlockMultiVar)
   */
  void
  compute_num_vars(const uint32_t                     num_vars_per_mat,
                   Kokkos::View<uint32_t *, device_t> num_vars) const
  {
    Kokkos::RangePolicy<typename device_t::execution_space> policy(0, num_vars.size());
    Kokkos::parallel_for(
      "kalypsso::MaterialPresence::compute_num_vars",
      policy,
      KOKKOS_CLASS_LAMBDA(const uint32_t i_oct) {
        num_vars(i_oct) = num_materials(i_oct) * num_vars_per_mat;
      });
  }


private:
  static constexpr auto NUM_BITS_PER_BLOCK = static_cast<uint32_t>(sizeof(BitBlock_t) * 8);

  /**
   * \brief Returns the index of the material number.
   *
   * It essentially counts the number of bits set before (and excluding) the material number's bit.
   */
  KOKKOS_INLINE_FUNCTION int32_t
  material_index_unchecked(int32_t i_oct, int32_t mat_num) const
  {
    int32_t    total = 0;
    const auto start_var = var(i_oct, 0);
    const auto end_var = var(i_oct, mat_num);

    // Counts the '1' in bit blocks before the nth bit
    for (uint32_t i = start_var; i < end_var; i++)
      total += bit_block_popcount(m_mat_pres(i));

    // Counts the remaining '1'
    total +=
      bit_block_popcount(m_mat_pres(end_var), static_cast<uint32_t>(mat_num) % NUM_BITS_PER_BLOCK);
    return total;
  }

  /**
   * \brief Given a material number, returns a mask of said bit in a bit block
   */
  KOKKOS_INLINE_FUNCTION BitBlock_t
  bit(int32_t mat_num) const
  {
    return BitBlock_t(1) << (static_cast<uint32_t>(mat_num) % NUM_BITS_PER_BLOCK);
  }

  /**
   * \brief Given a material number and an octant, returns the block index of where it is located
   */
  KOKKOS_INLINE_FUNCTION uint32_t
  var(int32_t i_oct, int32_t mat_num) const
  {
    return static_cast<uint32_t>(i_oct) * m_len_per_oct +
           static_cast<uint32_t>(mat_num) / NUM_BITS_PER_BLOCK;
  }

  /**
   * \brief Helper for counting the number of bits inside a block or a block's prefix.
   *
   * Is essentially a wrapper around a popcount operation. It uses Kokkos' experimental popcount
   * because it calls compiler's specific code to execute popc. The ternary operator is used
   * because if offset == NUM_BITS_PER_BLOCK (eg the bit width of MAX_VAR), then the shift is
   * ignored.
   */
  static KOKKOS_INLINE_FUNCTION uint8_t
  bit_block_popcount(BitBlock_t v, uint8_t max_bit_mask = NUM_BITS_PER_BLOCK)
  {
    static constexpr BitBlock_t MAX_VAR = Kokkos::Experimental::finite_max_v<BitBlock_t>;
    const uint8_t               offset = NUM_BITS_PER_BLOCK - max_bit_mask;
    return (max_bit_mask == 0) ? 0
                               : static_cast<uint8_t>(
                                   Kokkos::Experimental::popcount_builtin(v & (MAX_VAR >> offset)));
  }

  //! Indicates the maximum number of materials
  uint32_t m_max_mat;

  //! Indicates the length of the bitset
  uint32_t m_len_per_oct;

  //! number of octants (must be smaller of equal to m_mat_pres actual size)
  uint32_t m_num_octants;

  //! Actual inner view
  View_t m_mat_pres;

}; // class MaterialPresenceView

} // namespace kalypsso

#endif // KALYPSSO_CORE_MATERIAL_PRESENCE_H_
