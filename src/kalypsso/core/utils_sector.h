// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file utils_sector.h
 *
 * Define utilities to convert a sector into a set of face ids.
 *
 * Here are the sector enumeration id 2d
 *
 *               |                     |
 *  sector 6     |     sector 7        |  sector 8
 *          _____|_____________________|_______
 *               |                     |
 *               |                     |
 *               |                     |
 *               |                     |
 *  sector 3     |     sector 4        |  sector 5
 *               |                     |
 *               |                     |
 *               |                     |
 *               |                     |
 *          _____|_____________________|_______
 *               |                     |
 *  sector 0     |     sector 1        | sector 2
 *               |                     |
 *
 *
 * face sectors are 1, 3, 5 and 7. They are identified by a single face id
 *
 * sector 1 : YMIN
 * sector 3 : XMIN
 * sector 5 : XMAX
 * sector 7 : YMAX
 *
 * corner sector are 0, 2, 6 and 8. They are identified by a pair of face ids
 *
 * sector 0 : XMIN, YMIN
 * sector 2 : XMAX, YMIN
 * sector 6 : XMIN, YMAX
 * sector 8 : XMAX, YMAX
 *
 * All this can be extended in 3d, see below.
 */
#ifndef KALYPSSO_CORE_UTILSSECTOR_H_
#define KALYPSSO_CORE_UTILSSECTOR_H_
#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/mesh_utils.h>

#include <string>

namespace kalypsso
{

namespace core
{

// ====================================================================================
// 2D
// ====================================================================================

//! return true if sector s corresponds to a face neighbor.
//!
//! face sectors in 2d are : 1, 3, 5, 7
//! \note
//! uint32_t mask used to filter face sectors:
//! (1<<1) | (1<<3) | (1<<5) | (1<<7)
//!
//! \param[in] s sector
KOKKOS_INLINE_FUNCTION bool
is_face_sector_2d(uint32_t s)
{
  return (s < 9u) && ((0x00AAu >> s) & 1u);
}

//! return true if sector s corresponds to a corner neighbor.
//!
//! corner sectors in 2d are : 0, 2, 6, 8
//! \note
//! uint32_t mask used to filter face sectors:
//! (1<<0) | (1<<2) | (1<<6) | (1<<8)
//!
//! \param[in] s sector
KOKKOS_INLINE_FUNCTION bool
is_corner_sector_2d(uint32_t s)
{
  return (s < 9u) && ((0x0145u >> s) & 1u);
}

// ====================================================================================
// ====================================================================================
/**
 * Utils for converting a sector id into a face id in 2d.
 *
 * \param[in] s sector (must be integer between 0 and 8)
 *
 * Implementation note: face label is computed using a packed table (to avoid a switch case)
 *
 * Face sectors are : 1, 3, 5 and 7
 *
 * | sector (Decimal) | sector (binary) | face label | face id |
 * |                  |   s2 s1 s0      |            | f1 f0   |
 * |                  |                 |            |         |
 * |      1           |    0  0  1      |  YMIN      |  1  0   |
 * |      3           |    0  1  1      |  XMIN      |  0  0   |
 * |      5           |    1  0  1      |  XMAX      |  0  1   |
 * |      7           |    1  1  1      |  YMAX      |  1  1   |
 */
KOKKOS_INLINE_FUNCTION uint8_t
get_face_id_2d(uint32_t s)
{
  // 2 bits per odd sector (s >> 1 used as index) :
  // s=1 (idx 0) -> YMIN = 0b10 (2)
  // s=3 (idx 1) -> XMIN = 0b00 (0)
  // s=5 (idx 2) -> XMAX = 0b01 (1)
  // s=7 (idx 3) -> YMAX = 0b11 (3)
  const uint32_t FACE_LUT_2D =
    (2u << (0 * 2)) | (0u << (1 * 2)) | (1u << (2 * 2)) | (3u << (3 * 2)); // = 0xD2

  return (FACE_LUT_2D >> ((s >> 1) * 2u)) & 0x3u;
}

/**
 * Utils for converting a sector id into a pair of face ids (face0, face1) in 2d.
 *
 * \param[in] s sector (must be integer between 0 and 8)
 *
 * Implementation note: face label is computed using a packed table (to avoid a switch case)
 *
 * Corner sectors (even numbers):
 *
 * | sector (Decimal) | sector (binary) | corner labels |    corner ids  |
 * |                  |  s3 s2 s1 s0    |               |  c3 c2   c1 c0 |
 * |                  |                 |               |                |
 * |      O           |   0  0  0  0    |  XMIN, YMIN   |   0  0,   1  0 |
 * |      2           |   0  0  1  0    |  XMAX, YMIN   |   0  1,   1  0 |
 * |      6           |   0  1  1  0    |  XMIN, YMAX   |   0  0,   1  1 |
 * |      8           |   1  0  0  0    |  XMAX, YMAX   |   0  1,   1  1 |
 *
 */
KOKKOS_INLINE_FUNCTION void
get_corner_ids_2d(uint32_t s, uint8_t & f0, uint8_t & f1)
{
  // f0 low bit is 1 for XMAX (s = 2, 8)
  const uint32_t MASK_X = (1u << 2) | (1u << 8);

  // f1 low bit is 1 for YMAX (s = 6, 8)
  const uint32_t MASK_Y = (1u << 6) | (1u << 8);

  f0 = 0x0u | ((MASK_X >> s) & 1u); // 00 or 01
  f1 = 0x2u | ((MASK_Y >> s) & 1u); // 10 or 11
}

// ====================================================================================
// 3D
// ====================================================================================

//! return true if sector s corresponds to a face neighbor.
//!
//! face sectors in 3d are : 4, 10, 12, 14, 16, 22
//! \note
//! uint32_t mask used to filter face sectors:
//! (1<<4) | (1<<10) | (1<<12) | (1<<14) | (1<<16) | (1<<22)
//!
//! \param[in] s sector
KOKKOS_INLINE_FUNCTION bool
is_face_sector_3d(uint32_t s)
{
  return (s < 32u) && ((0x00415410u >> s) & 1u);
}

//! return true if sector s corresponds to an edge neighbor.
//!
//! edge sectors in 3d are : 1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25
//! \note
//! uint32_t mask used to filter edge sectors:
//! (1<<1) | (1<<3) | (1<<5) | (1<<7) | (1<<9) | (1<<11) | (1<<15) | (1<<17) | (1<<19) | (1<<21) |
//! (1<<23) | (1<<25)
//!
//! \param[in] s sector
KOKKOS_INLINE_FUNCTION bool
is_edge_sector_3d(uint32_t s)
{
  return (s < 32u) && ((0x02AA8AAAu >> s) & 1u);
}

//! return true if sector s corresponds to a corner sector.
//!
//! corner sectors in 3d are : 0, 2, 6, 8, 18, 20, 24, 26
//! \note
//! uint32_t mask used to filter corner sectors:
//! (1<<0) | (1<<2) | (1<<6) | (1<<8) | (1<<18) | (1<<20) | (1<<24) | (1<<26)
//!
//! \param[in] s sector
KOKKOS_INLINE_FUNCTION bool
is_corner_sector_3d(uint32_t s)
{
  return (s < 32u) && ((0x05140145u >> s) & 1u);
}

// ====================================================================================
// ====================================================================================
/**
 * Utils for converting a sector id into a face id in 3d.
 *
 * \param[in] s sector (must be integer between 0 and 26)
 *
 * Implementation note: face label is computed using a packed table (to avoid a switch case)
 *
 * Face sectors are : 4, 10, 12, 14, 16 and 22
 *
 * | sector (decimal) | sector (binary) | face label  |  face ids   |
 * |                  |  s4 s3 s2 s1 s0 |             |  f2  f1  f0 |
 * |                  |                 |             |             |
 * |            4     |   0  0  1  0  0 |    ZMIN     |   1   0   0 |
 * |           10     |   0  1  0  1  0 |    YMIN     |   0   1   0 |
 * |           12     |   0  1  1  0  0 |    XMIN     |   0   0   0 |
 * |           14     |   0  1  1  1  0 |    XMAX     |   0   0   1 |
 * |           16     |   1  0  0  0  0 |    YMAX     |   0   1   1 |
 * |           22     |   1  0  1  1  0 |    ZMAX     |   1   0   1 |
 *
 * +----+----+----+   +----+----+----+   +----+----+----+
 *
 * |  . |  . |  . |   |  . | 16 |  . |   |  . |  . |  . |
 * +----+----+----+   +----+----+----+   +----+----+----+
 *
 * |  . |  4 |  . |   | 12 |  . | 14 |   |  . | 22 |  . |
 * +----+----+----+   +----+----+----+   +----+----+----+
 *
 * |  . |  . |  . |   |  . | 10 |  . |   |  . |  . |  . |
 * +----+----+----+   +----+----+----+   +----+----+----+
 *
 * Face sectors are all even, so used s divided by 2 as input to the face lookup table
 *
 */
KOKKOS_INLINE_FUNCTION
uint8_t
get_face_id_3d(uint32_t s)
{
  KOKKOS_ASSERT(s < 27 && "Wrong value: sector must be strictly smaller than 27");

  // used (s >> 1) as input index, 3 bits per input :
  // s=4  (idx 2)  -> 0b100 (4) ZMIN
  // s=10 (idx 5)  -> 0b010 (2) YMIN
  // s=12 (idx 6)  -> 0b000 (0) XMIN
  // s=14 (idx 7)  -> 0b001 (1) XMAX
  // s=16 (idx 8)  -> 0b011 (3) YMAX
  // s=22 (idx 11) -> 0b111 (5) ZMAX
  // clang-format off
  constexpr uint64_t FACE_LUT =
    (4ULL << (2 * 3)) |
    (2ULL << (5 * 3)) |
    (0ULL << (6 * 3)) |
    (1ULL << (7 * 3)) |
    (3ULL << (8 * 3)) |
    (5ULL << (11 * 3));
  // clang-format on

  // (s >> 1) * 3
  uint32_t shift = (s >> 1) * 3u;
  return (FACE_LUT >> shift) & 0x7u;
}

// ====================================================================================
// ====================================================================================
/**
 * Utils for converting a sector id into a pair of face ids (face0,face1) in 3d.
 *
 * \param[in] s sector (must be integer between 0 and 26)
 */
KOKKOS_INLINE_FUNCTION void
get_edge_ids_3d(uint32_t s, uint8_t & f0, uint8_t & f1)
{
  // id1 (0..3) using 2 bits :
  // 0 -> XMIN
  // 1 -> XMAX
  // 2 -> YMIN
  // 3 -> YMAX
  const uint64_t LUT_ID1 = (2ULL << (1 * 2)) | (0ULL << (3 * 2)) | (1ULL << (5 * 2)) |
                           (3ULL << (7 * 2)) | (0ULL << (9 * 2)) | (1ULL << (11 * 2)) |
                           (0ULL << (15 * 2)) | (1ULL << (17 * 2)) | (2ULL << (19 * 2)) |
                           (0ULL << (21 * 2)) | (1ULL << (23 * 2)) | (3ULL << (25 * 2));

  // id2 - 2 (0..3 -> 0b010..0b101) using 2 bits :
  // 0 -> YMIN
  // 1 -> YMAX
  // 2 -> ZMIN
  // 3 -> ZMAX
  const uint64_t LUT_ID2 = (2ULL << (1 * 2)) | (2ULL << (3 * 2)) | (2ULL << (5 * 2)) |
                           (2ULL << (7 * 2)) | (0ULL << (9 * 2)) | (0ULL << (11 * 2)) |
                           (1ULL << (15 * 2)) | (1ULL << (17 * 2)) | (3ULL << (19 * 2)) |
                           (3ULL << (21 * 2)) | (3ULL << (23 * 2)) | (3ULL << (25 * 2));

  uint32_t shift = s << 1;
  f0 = (LUT_ID1 >> shift) & 0x3u;
  f1 = ((LUT_ID2 >> shift) & 0x3u) + 2u;
}

// ====================================================================================
// ====================================================================================
/**
 * Utils for converting a sector id into a tuple of face ids (face0, face1, face2) in 3d.
 *
 * \param[in] s sector (must be integer between 0 and 26)
 *
 * In 3d, corner sectors are :
 *
 * | sector (decimal) | sector (binary) | corner labels     |      corner ids    |
 * |                  |  s4 s3 s2 s1 s0 |                   |  c8-c6 c5-c3 c2-c0 |
 * |                  |                 |                   |                    |
 * |            0     |   0  0  0  0  0 | XMIN, YMIN, ZMIN  |  000   010   100   |
 * |            2     |   0  0  0  1  0 | XMAX, YMIN, ZMIN  |  001   010   100   |
 * |            6     |   0  0  1  1  0 | XMIN, YMAX, ZMIN  |  000   011   100   |
 * |            8     |   0  1  0  0  0 | XMAX, YMAX, ZMIN  |  001   011   100   |
 * |           18     |   1  0  0  1  0 | XMIN, YMIN, ZMAX  |  000   010   101   |
 * |           20     |   1  0  1  0  0 | XMAX, YMIN, ZMAX  |  001   010   101   |
 * |           24     |   1  1  0  0  0 | XMIN, YMAX, ZMAX  |  000   011   101   |
 * |           26     |   1  1  0  1  0 | XMAX, YMAX, ZMAX  |  001   011   101   |
 *
 * +----+----+----+   +----+----+----+   +----+----+----+
 *
 * | *6*|  7 | *8*|   | 15 | 16 | 17 |   |*24*| 25 |*26*|
 * +----+----+----+   +----+----+----+   +----+----+----+
 *
 * |  3 |  4 |  5 |   | 12 | 13 | 14 |   | 21 | 22 | 23 |
 * +----+----+----+   +----+----+----+   +----+----+----+
 *
 * | *0*|  1 | *2*|   |  9 | 10 | 11 |   |*18*| 19 |*20*|
 * +----+----+----+   +----+----+----+   +----+----+----+
 */
KOKKOS_INLINE_FUNCTION void
get_corner_ids_3d(uint32_t s, uint8_t & f0, uint8_t & f1, uint8_t & f2)
{
  // Bit c6 (XMAX vs XMIN) : is equal to 1 for s = 2, 8, 20, 26
  const uint32_t MASK_C6 = (1u << 2) | (1u << 8) | (1u << 20) | (1u << 26);

  // Bit c3 (YMAX vs YMIN) : is equal to 1 for s = 6, 8, 24, 26
  const uint32_t MASK_C3 = (1u << 6) | (1u << 8) | (1u << 24) | (1u << 26);

  // Bit c0 (ZMAX vs ZMIN) : is equal to 1 for s = 18, 20, 24, 26
  const uint32_t MASK_C0 = (1u << 18) | (1u << 20) | (1u << 24) | (1u << 26);

  f0 = 0x0u | ((MASK_C6 >> s) & 1u); // 000 ou 001
  f1 = 0x2u | ((MASK_C3 >> s) & 1u); // 010 ou 011
  f2 = 0x4u | ((MASK_C0 >> s) & 1u); // 100 ou 101
}

// ====================================================================================
// ====================================================================================
inline std::string
face_to_string(uint8_t face)
{
  if (face == 0)
    return std::string("XMIN");
  else if (face == 1)
    return std::string("XMAX");
  else if (face == 2)
    return std::string("YMIN");
  else if (face == 3)
    return std::string("YMAX");
  else if (face == 4)
    return std::string("ZMIN");
  else if (face == 5)
    return std::string("ZMAX");

  return std::string("unknown");
}

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_UTILSSECTOR_H_
