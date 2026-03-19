// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file BitFieldInteger.h
 *
 * Adapted from svom eclairs software (original author F. Chateau)
 */

#ifndef KALYPSSO_CORE_BITFIELDINTEGER_H_
#define KALYPSSO_CORE_BITFIELDINTEGER_H_

#include "BitFieldHelper.h"

#include <type_traits>

namespace kalypsso
{

/**
 * This should be used as the base class of datatypes that implements a bit field semantics.
 *
 * It is designed to be used in conjunction with DECLARE_BIT and DECLARE_FIELD
 * macros. These macro must be called in the subclass body. They generate
 * accessors for each bit and fields.
 */
template <typename T>
class BitFieldInteger
{
public:
  using WordType = T;
  using WordBasicType = std::remove_cv_t<WordType>;

  KOKKOS_DEFAULTED_FUNCTION constexpr BitFieldInteger() = default;
};

} // namespace kalypsso

#define DECLARE_RO_BIT(name, pos)                                            \
  static KOKKOS_INLINE_FUNCTION constexpr bool name(WordBasicType readValue) \
  {                                                                          \
    return BitFieldHelper<WordType>::getBit<(pos)>(readValue);               \
  }

#define DECLARE_WO_BIT(name, pos)                                                            \
  static KOKKOS_INLINE_FUNCTION constexpr WordBasicType cset_##name(WordBasicType readValue, \
                                                                    bool          val)       \
  {                                                                                          \
    return BitFieldHelper<WordType>::setBit<(pos)>(readValue, val);                          \
  }                                                                                          \
  static KOKKOS_INLINE_FUNCTION void set_##name(WordBasicType & readValue, bool val)         \
  {                                                                                          \
    readValue = BitFieldHelper<WordType>::setBit<(pos)>(readValue, val);                     \
  }

#define DECLARE_BIT(name, pos) \
  DECLARE_RO_BIT(name, (pos))  \
  DECLARE_WO_BIT(name, (pos))

#define DECLARE_CASTED_RO_FIELD(name, pos, width, type)                         \
  static KOKKOS_INLINE_FUNCTION constexpr type name(WordBasicType readValue)    \
  {                                                                             \
    return BitFieldHelper<WordType>::getField<(pos), (width), type>(readValue); \
  }

#define DECLARE_CASTED_WO_FIELD(name, pos, width, type)                                         \
  static KOKKOS_INLINE_FUNCTION constexpr WordBasicType cset_##name(WordBasicType readValue,    \
                                                                    type          val)          \
  {                                                                                             \
    return BitFieldHelper<WordType>::setField<(pos), (width)>(readValue,                        \
                                                              static_cast<WordBasicType>(val)); \
  }                                                                                             \
  static KOKKOS_INLINE_FUNCTION void set_##name(WordBasicType & readValue, type val)            \
  {                                                                                             \
    readValue = BitFieldHelper<WordType>::setField<(pos), (width)>(                             \
      readValue, static_cast<WordBasicType>(val));                                              \
  }

#define DECLARE_CASTED_FIELD(name, pos, width, type)  \
  DECLARE_CASTED_RO_FIELD(name, (pos), (width), type) \
  DECLARE_CASTED_WO_FIELD(name, (pos), (width), type)

#define DECLARE_RO_FIELD(name, pos, width) \
  DECLARE_CASTED_RO_FIELD(name, (pos), (width), WordBasicType)

#define DECLARE_WO_FIELD(name, pos, width) \
  DECLARE_CASTED_WO_FIELD(name, (pos), (width), WordBasicType)

#define DECLARE_FIELD(name, pos, width)  \
  DECLARE_RO_FIELD(name, (pos), (width)) \
  DECLARE_WO_FIELD(name, (pos), (width))

#endif /* KALYPSSO_CORE_BITFIELDINTEGER_H_ */
