// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file BitFieldHelper.h
 *
 * Adapted from svom eclairs software (original author F. Chateau)
 */
#ifndef KALYPSSO_CORE_BITFIELDHELPER_H
#define KALYPSSO_CORE_BITFIELDHELPER_H

#include <Kokkos_Macros.hpp>

#include <cstdint>
#include <type_traits>
#include <climits> // for CHAR_BIT
#include <cassert>

#include <kalypsso/core/kalypsso_macros.h>

KALYPSSO_DISABLE_CONVERSION_WARNINGS_PUSH()
KALYPSSO_DISABLE_NVCC_WARNINGS_PUSH()

namespace kalypsso
{

using BitCount = uint8_t;
using BitIdx = uint8_t;

/**
 * This class provides template methods to manipulate bit-fields easily and efficiently.
 *
 * @note setField, setBit and clearBit do not modify their argument.
 */
template <typename WordType>
class BitFieldHelper
{
public:
  using WordBasicType = std::remove_cv_t<WordType>;
  using WordSignedType = std::make_signed_t<WordBasicType>;
  using WordUnsignedType = std::make_unsigned_t<WordBasicType>;

  /**
   * Compile-time mapping that returns the signed or unsigned version of WordType
   * depending on the sign of the specified template parameter.
   */
  template <typename ValueType>
  using WordTypeMatchingSignOf =
    std::conditional_t<std::is_signed<ValueType>::value, WordSignedType, WordUnsignedType>;

  static constexpr WordBasicType ZERO = 0;
  static constexpr WordBasicType ONE = 1;
  static constexpr WordBasicType ALL = ~0;

  static constexpr BitCount WORD_BYTECOUNT = static_cast<BitCount>(sizeof(WordBasicType));
  static constexpr BitCount WORD_BITCOUNT = static_cast<BitCount>(WORD_BYTECOUNT * CHAR_BIT);

  ///@name Compile-time bit-field manipulation.
  ///@{
  template <BitIdx POS>
  KOKKOS_FUNCTION static constexpr bool
  getBit(WordBasicType var);

  template <BitIdx POS>
  KOKKOS_FUNCTION static constexpr WordType
  clearBit(WordType var);

  template <BitIdx POS>
  KOKKOS_FUNCTION static constexpr WordType
  setBit(WordType var);

  template <BitIdx POS>
  KOKKOS_FUNCTION static constexpr WordType
  setBit(WordType var, bool askSet);

  template <BitIdx POS>
  KOKKOS_FUNCTION static constexpr WordType
  toggleBit(WordType var);

  template <unsigned POS, unsigned WIDTH>
  KOKKOS_FUNCTION static auto
  foobar(WordType var) -> WordType;

  template <BitIdx POS, BitCount WIDTH>
  KOKKOS_FUNCTION static constexpr auto
  getField(WordBasicType var) -> WordBasicType;

  template <BitIdx POS, BitCount WIDTH, typename FieldType>
  KOKKOS_FUNCTION static constexpr auto
  getField(WordBasicType var) -> FieldType;

  template <BitIdx POS, BitCount WIDTH, typename FieldType>
  KOKKOS_FUNCTION static constexpr void
  getField(WordBasicType var, FieldType & result);

  template <BitIdx POS, BitCount WIDTH>
  KOKKOS_FUNCTION static constexpr WordType
  setField(WordType var, WordBasicType value);

  template <BitIdx POS, BitCount WIDTH>
  KOKKOS_FUNCTION static constexpr WordType
  toggleField(WordType var);

  template <BitIdx POS, BitCount WIDTH>
  KOKKOS_FUNCTION static constexpr auto
  getReadMask() -> WordUnsignedType;

  template <BitIdx POS, BitCount WIDTH>
  KOKKOS_FUNCTION static constexpr auto
  getClearMask() -> WordUnsignedType;
  ///@}

  ///@name Run-time bit-field manipulation.
  ///@{
  KOKKOS_FUNCTION static bool
  getBit(WordBasicType var, BitIdx pos);

  KOKKOS_FUNCTION static WordType
  clearBit(WordType var, BitIdx pos);

  KOKKOS_FUNCTION static WordType
  setBit(WordType var, BitIdx pos);

  KOKKOS_FUNCTION static WordType
  setBit(WordType var, BitIdx pos, bool askSet);

  KOKKOS_FUNCTION static WordType
  toggleBit(WordType var, BitIdx pos);

  KOKKOS_FUNCTION static auto
  getField(WordBasicType var, BitIdx pos, BitCount width) -> WordBasicType;

  template <typename FieldType>
  KOKKOS_FUNCTION static FieldType
  getField(WordBasicType var, BitIdx pos, BitCount width);

  template <typename FieldType>
  KOKKOS_FUNCTION static void
  getField(WordBasicType var, BitIdx pos, BitCount width, FieldType & result);

  KOKKOS_FUNCTION static WordType
  setField(WordType var, BitIdx pos, BitCount width, WordBasicType value);

  KOKKOS_FUNCTION static WordType
  toggleField(WordType var, BitIdx pos, BitCount width);

  KOKKOS_FUNCTION static auto
  getReadMask(BitIdx pos, BitCount width) -> WordUnsignedType;

  KOKKOS_FUNCTION static auto
  getClearMask(BitIdx pos, BitCount width) -> WordUnsignedType;
  ///@}

  BitFieldHelper() = delete;
  ~BitFieldHelper() = delete;
};

template <typename WordType>
using NativeBitField = BitFieldHelper<WordType>;

/**
 * Performs a bitwise NOT, or bit-toggling.
 * @param value
 * @return
 */
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T
toggle(T value)
{
  return static_cast<T>(~static_cast<std::make_unsigned_t<T>>(value));
}

/**
 * Computes an integer exponential of base 2.
 * The result essentially have the nth bit at 1 and the rest at 0.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION T
exponent2(BitIdx arg)
{
  assert(arg < sizeof(T) * CHAR_BIT);
  return static_cast<T>(static_cast<T>(1) << arg);
}

/**
 * Computes an integer exponential of base 2 with compile-time argument.
 * The result essentially have the nth bit at 1 and the rest at 0.
 */
template <typename T, BitIdx ARG>
KOKKOS_INLINE_FUNCTION constexpr T
exponent2()
{
  static_assert(ARG < sizeof(T) * CHAR_BIT, "exponent2 template argument too big for type");
  return static_cast<T>(static_cast<T>(1) << ARG);
}

/**
 * Binary left shift for unsigned values.
 * @param value the value to shift
 * @param nbits the number of bits to shift
 * @return Returns value left-shifted by nbits.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION T
lhs(T value, std::enable_if_t<std::is_unsigned<T>::value, BitCount> nbits)
{
  assert(nbits <= sizeof(T) * CHAR_BIT);
  return static_cast<T>(value << nbits);
}

/**
 * Binary left shift for unsigned values, with compile-time parameter.
 * @param value the value to shift
 * @tparam NBITS the number of bits to shift
 * @return Returns value left-shifted by NBITS.
 */
template <BitIdx NBITS, typename T, std::enable_if_t<std::is_unsigned<T>::value, T> = 0>
KOKKOS_INLINE_FUNCTION constexpr T
lhs(T value)
{
  static_assert(NBITS <= sizeof(T) * CHAR_BIT, "lhs template argument too big for type");
  return static_cast<T>(value << NBITS);
}

/**
 * Binary left shift for signed values.
 * @param value the value to shift
 * @param nbits the number of bits to shift
 * @return Returns value left-shifted by nbits.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION T
lhs(T value, std::enable_if_t<std::is_signed<T>::value, BitIdx> nbits)
{
  assert(nbits < sizeof(T) * CHAR_BIT);
  return static_cast<T>(static_cast<std::make_unsigned_t<T>>(value) << nbits);
}

/**
 * Binary left shift for signed values, with compile-time parameter.
 * @param value the value to shift
 * @tparam NBITS the number of bits to shift
 * @return Returns value left-shifted by NBITS.
 */
template <uint8_t NBITS, typename T, std::enable_if_t<std::is_signed<T>::value, T> = 0>
KOKKOS_INLINE_FUNCTION constexpr T
lhs(T value)
{
  static_assert(NBITS < sizeof(T) * CHAR_BIT, "lhs template argument too big for type");
  return static_cast<T>(static_cast<std::make_unsigned_t<T>>(value) << NBITS);
}

/**
 * Binary right shift.
 * @param value the value to shift
 * @param nbits the number of bits to shift
 * @return Returns value right-shifted by nbits.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION T
rhs(T value, BitCount nbits)
{
  assert(nbits < sizeof(T) * CHAR_BIT);
  // Here we suppose right shifting signed values will extend the sign.
  return value >> nbits;
}

/**
 * Binary right shift, with compile-time parameter.
 * @param value the value to shift
 * @tparam NBITS the number of bits to shift
 * @return Returns value right-shifted by NBITS.
 */
template <uint8_t NBITS, typename T>
KOKKOS_INLINE_FUNCTION constexpr T
rhs(T value)
{
  static_assert(NBITS < sizeof(T) * CHAR_BIT, "rhs template argument too big for type");
  // Here we suppose right shifting signed values will extend the sign.
  return value >> NBITS;
}

/**
 * Swaps the bytes of a 8-bits unsigned integer.
 * Basically does nothing, only here for covering the whole set of integer types.
 * @param value the value to swap.
 * @return Returns the swapped integer.
 */
KOKKOS_INLINE_FUNCTION constexpr uint8_t
byteSwap(const uint8_t value)
{
  return value;
}

/**
 * Swaps the bytes of a 16-bits unsigned integer.
 * @param value the value to swap.
 * @return Returns the swapped integer.
 */
KOKKOS_INLINE_FUNCTION constexpr uint16_t
byteSwap(const uint16_t value)
{
  return NativeBitField<uint16_t>::getField<8, 8>(value) |
         lhs<8>(NativeBitField<uint16_t>::getField<0, 8>(value));
}

/**
 * Swaps the bytes of a 32-bits unsigned integer.
 * @param value the value to swap.
 * @return Returns the swapped integer.
 */
KOKKOS_INLINE_FUNCTION constexpr uint32_t
byteSwap(const uint32_t value)
{
  return NativeBitField<uint32_t>::getField<24, 8>(value) |
         lhs<8>(NativeBitField<uint32_t>::getField<16, 8>(value)) |
         lhs<16>(NativeBitField<uint32_t>::getField<8, 8>(value)) |
         lhs<24>(NativeBitField<uint32_t>::getField<0, 8>(value));
}

/**
 * Swaps the bytes of a 64-bits unsigned integer.
 * @param value the value to swap.
 * @return Returns the swapped integer.
 */
KOKKOS_INLINE_FUNCTION constexpr uint64_t
byteSwap(const uint64_t value)
{
  return NativeBitField<uint64_t>::getField<56, 8>(value) |
         lhs<8>(NativeBitField<uint64_t>::getField<48, 8>(value)) |
         lhs<16>(NativeBitField<uint64_t>::getField<40, 8>(value)) |
         lhs<24>(NativeBitField<uint64_t>::getField<32, 8>(value)) |
         lhs<32>(NativeBitField<uint64_t>::getField<24, 8>(value)) |
         lhs<40>(NativeBitField<uint64_t>::getField<16, 8>(value)) |
         lhs<48>(NativeBitField<uint64_t>::getField<8, 8>(value)) |
         lhs<56>(NativeBitField<uint64_t>::getField<0, 8>(value));
}

/**
 * Swaps the bytes of any signed integer.
 * As the standard states that converting a negative number to unsigned uses the next congruent
 * positive value, which changes nothing at the underlying twos-complement representation, we can
 * just use the unsigned version of the byteSwap and performs casts to make the compiler happy.
 * @param value
 * @return
 */
template <typename T, std::enable_if_t<std::is_signed<T>::value, T> = 0>
KOKKOS_INLINE_FUNCTION constexpr T
byteSwap(T value)
{
  return static_cast<T>(byteSwap(static_cast<std::make_unsigned_t<T>>(value)));
}


// =============================================================================
// BitFieldHelper inline methods body
// -----------------------------------------------------------------------------

/**
 * Returns the value of a bit of an integer variable (compile-time version).
 */
template <typename WordType>
template <BitIdx POS>
KOKKOS_FUNCTION constexpr bool
BitFieldHelper<WordType>::getBit(WordBasicType var)
{
  return (var & exponent2<WordUnsignedType, POS>()) != 0;
}

/**
 * Returns the value of a bit of an integer variable.
 */
template <typename WordType>
KOKKOS_FUNCTION bool
BitFieldHelper<WordType>::getBit(WordBasicType var, BitIdx pos)
{
  return (var & exponent2<WordUnsignedType>(pos)) != 0;
}

/**
 * Clears the bit at the specified compile-time position.
 */
template <typename WordType>
template <BitIdx POS>
KOKKOS_FUNCTION constexpr WordType
BitFieldHelper<WordType>::clearBit(WordType var)
{
  return var & toggle(exponent2<WordUnsignedType, POS>());
}

/**
 * Clears the bit at the specified parameter position.
 */
template <typename WordType>
KOKKOS_FUNCTION WordType
BitFieldHelper<WordType>::clearBit(WordType var, BitIdx pos)
{
  return var & toggle(exponent2<WordUnsignedType>(pos));
}

/**
 * Sets the bit at the specified compile-time position.
 */
template <typename WordType>
template <BitIdx POS>
KOKKOS_FUNCTION constexpr WordType
BitFieldHelper<WordType>::setBit(WordType var)
{
  return var | exponent2<WordUnsignedType, POS>();
}

/**
 * Sets the bit at the specified position in parameter.
 */
template <typename WordType>
KOKKOS_FUNCTION WordType
BitFieldHelper<WordType>::setBit(WordType var, BitIdx pos)
{
  return var | exponent2<WordUnsignedType>(pos);
}

/**
 * Sets or clears a bit of a variable (compile-time version).
 */
template <typename WordType>
template <BitIdx POS>
KOKKOS_FUNCTION constexpr WordType
BitFieldHelper<WordType>::setBit(WordType var, bool askSet)
{
  if (askSet)
  {
    return setBit<POS>(var);
  }
  else
  {
    return clearBit<POS>(var);
  }
}

template <typename WordType>
template <BitIdx POS>
KOKKOS_FUNCTION constexpr WordType
BitFieldHelper<WordType>::toggleBit(WordType var)
{
  return var ^ exponent2<WordUnsignedType, POS>();
}

/**
 * Sets or clears a bit of a variable.
 */
template <typename WordType>
KOKKOS_FUNCTION WordType
BitFieldHelper<WordType>::setBit(WordType var, BitIdx pos, bool askSet)
{
  if (askSet)
  {
    return setBit(var, pos);
  }
  else
  {
    return clearBit(var, pos);
  }
}

template <typename WordType>
KOKKOS_FUNCTION WordType
BitFieldHelper<WordType>::toggleBit(WordType var, BitIdx pos)
{
  return var ^ exponent2<WordUnsignedType>(pos);
}

/**
 * Returns the value of a bit-field, with the same type as the input (compile-time version).
 */
template <typename WordType>
template <unsigned POS, unsigned WIDTH>
KOKKOS_FUNCTION WordType
BitFieldHelper<WordType>::foobar(WordType var)
{
  WordBasicType result = 0; // init should not be necessary but BCC2 requires it.
  getField<POS, WIDTH>(var, result);
  return result;
}

/**
 * Returns the value of a bit-field, with the same type as the input (compile-time version).
 */
template <typename WordType>
template <BitIdx POS, BitCount WIDTH>
KOKKOS_FUNCTION constexpr auto
BitFieldHelper<WordType>::getField(WordBasicType var) -> WordBasicType
{
  WordBasicType result = 0; // init should not be necessary but BCC2 requires it.
  getField<POS, WIDTH>(var, result);
  return result;
}
/**
 * Returns the value of a bit-field, with the same type as the input.
 */
template <typename WordType>
KOKKOS_FUNCTION auto
BitFieldHelper<WordType>::getField(WordBasicType var, BitIdx pos, BitCount width) -> WordBasicType
{
  WordBasicType result;
  getField(var, pos, width, result);
  return result;
}

/**
 * Returns the value of a bit-field, allowing the field to have a different type (compile-time
 * version).
 */
template <typename WordType>
template <BitIdx POS, BitCount WIDTH, typename FieldType>
KOKKOS_FUNCTION constexpr FieldType
BitFieldHelper<WordType>::getField(WordBasicType var)
{
  FieldType result{}; // init should not be necessary but BCC2 requires it.
  getField<POS, WIDTH>(var, result);
  return result;
}

/**
 * Returns the value of a bit-field, allowing the field to have a different type.
 */
template <typename WordType>
template <typename FieldType>
KOKKOS_FUNCTION FieldType
BitFieldHelper<WordType>::getField(WordBasicType var, BitIdx pos, BitCount width)
{
  FieldType result;
  getField(var, pos, width, result);
  return result;
}

/**
 * Retrieves the value of a bit-field (compile-time version).
 * The field does not need to have the same type (or same signed-ness) as the input.
 */
template <typename WordType>
template <BitIdx POS, BitCount WIDTH, typename FieldType>
KOKKOS_FUNCTION constexpr void
BitFieldHelper<WordType>::getField(WordBasicType var, FieldType & result)
{
  using SignCorrectWordType = WordTypeMatchingSignOf<FieldType>;

  const WordBasicType       leftMost = lhs<WORD_BITCOUNT - WIDTH - POS>(var);
  const SignCorrectWordType leftMostSignCorrect = static_cast<SignCorrectWordType>(leftMost);
  result = static_cast<FieldType>(rhs<WORD_BITCOUNT - WIDTH>(leftMostSignCorrect));
}

/**
 * Retrieves the value of a bit-field.
 * The field does not need to have the same type (or same signed-ness) as the input.
 */
template <typename WordType>
template <typename FieldType>
KOKKOS_FUNCTION void
BitFieldHelper<WordType>::getField(WordBasicType var,
                                   BitIdx        pos,
                                   BitCount      width,
                                   FieldType &   result)
{
  using SignCorrectWordType = WordTypeMatchingSignOf<FieldType>;

  const WordBasicType       leftMost = lhs(var, WORD_BITCOUNT - width - pos);
  const SignCorrectWordType leftMostSignCorrect = static_cast<SignCorrectWordType>(leftMost);
  result = static_cast<FieldType>(rhs(leftMostSignCorrect, WORD_BITCOUNT - width));
}

/**
 * Sets the value of a bit-field of a variable (compile-time version).
 */
template <typename WordType>
template <BitIdx POS, BitCount WIDTH>
KOKKOS_FUNCTION constexpr WordType
BitFieldHelper<WordType>::setField(WordType var, WordBasicType value)
{
  const WordUnsignedType clearedOld =
    static_cast<WordUnsignedType>(var) & getClearMask<POS, WIDTH>();
  const WordUnsignedType shiftedNew =
    static_cast<WordUnsignedType>(lhs<POS>(value)) & getReadMask<POS, WIDTH>();
  return static_cast<WordBasicType>(clearedOld | shiftedNew);
}

/**
 * Sets the value of a bit-field of a variable.
 */
template <typename WordType>
KOKKOS_FUNCTION WordType
BitFieldHelper<WordType>::setField(WordType var, BitIdx pos, BitCount width, WordBasicType value)
{
  const WordUnsignedType clearedOld = static_cast<WordUnsignedType>(var) & getClearMask(pos, width);
  const WordUnsignedType shiftedNew =
    static_cast<WordUnsignedType>(lhs(value, pos)) & getReadMask(pos, width);
  return static_cast<WordBasicType>(clearedOld | shiftedNew);
}

template <typename WordType>
template <BitIdx POS, BitCount WIDTH>
KOKKOS_FUNCTION constexpr WordType
BitFieldHelper<WordType>::toggleField(WordType var)
{
  const WordUnsignedType value = static_cast<WordUnsignedType>(var);
  const WordUnsignedType toggled = value ^ getReadMask<POS, WIDTH>();
  return static_cast<WordBasicType>(toggled);
}

template <typename WordType>
KOKKOS_FUNCTION WordType
BitFieldHelper<WordType>::toggleField(WordType var, BitIdx pos, BitCount width)
{
  const WordUnsignedType value = static_cast<WordUnsignedType>(var);
  const WordUnsignedType toggled = value ^ getReadMask(pos, width);
  return static_cast<WordBasicType>(toggled);
}

/**
 * Returns the mask used to isolate the bit-field range, in order to read it (compile-time
 * version). Protection against the machine-dependent case (and seen at run-time) for which width
 * == size of the word type
 */
template <typename WordType>
template <BitIdx POS, BitCount WIDTH>
KOKKOS_FUNCTION constexpr typename BitFieldHelper<WordType>::WordUnsignedType
BitFieldHelper<WordType>::getReadMask()
{
  return lhs<POS>(toggle(lhs<WIDTH>(ALL)));
}

/**
 * Returns the mask used to isolate the bit-field range, in order to read it.
 * Protection against the machine-dependent case (and seen at run-time)
 * for which width == size of the word type
 */
template <typename WordType>
KOKKOS_FUNCTION auto
BitFieldHelper<WordType>::getReadMask(BitIdx pos, BitCount width) -> WordUnsignedType
{
  return lhs(toggle(lhs(ALL, width)), pos);
}

/**
 * Returns the mask used to exclude the bit-field range, in order to clear it (compile-time
 * version).
 */
template <typename WordType>
template <BitIdx POS, BitCount WIDTH>
KOKKOS_FUNCTION constexpr auto
BitFieldHelper<WordType>::getClearMask() -> WordUnsignedType
{
  return toggle(getReadMask<POS, WIDTH>());
}

/**
 * Returns the mask used to exclude the bit-field range, in order to clear it.
 */
template <typename WordType>
KOKKOS_FUNCTION auto
BitFieldHelper<WordType>::getClearMask(BitIdx pos, BitCount width) -> WordUnsignedType
{
  return toggle(getReadMask(pos, width));
}

} // namespace kalypsso

KALYPSSO_DISABLE_CONVERSION_WARNINGS_POP()
KALYPSSO_DISABLE_NVCC_WARNINGS_POP()

#endif // KALYPSSO_CORE_BITFIELDHELPER_H
