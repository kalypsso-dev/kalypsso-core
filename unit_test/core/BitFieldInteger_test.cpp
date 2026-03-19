// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kalypsso/core/BitFieldInteger.h"
#include "kalypsso/core/orchard_key.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace kalypsso
{

TEST(kalypsso_shared_test, BitFieldInteger)
{

  class SomeData : public BitFieldInteger<uint32_t>
  {
  public:
    using BitFieldInteger::BitFieldInteger;

    DECLARE_CASTED_FIELD(f1, 0, 8, uint8_t)
    DECLARE_CASTED_FIELD(f2, 8, 4, uint8_t)
    DECLARE_CASTED_FIELD(f3, 12, 8, uint8_t)
    DECLARE_CASTED_FIELD(f4, 20, 12, uint16_t)
  };

  uint32_t value = (1 << 8) + (1 << 12) + (1 << 13);

  auto v1 = SomeData::f1(value);
  EXPECT_EQ(v1, 0) << "SomeData v1 failed";

  auto v2 = SomeData::f2(value);
  EXPECT_EQ(v2, 1) << "SomeData v2 failed";

  auto v3 = SomeData::f3(value);
  EXPECT_EQ(v3, 3) << "SomeData v3 failed";

  // value = SomeData::cset_f1(value, 15);
  SomeData::set_f1(value, 15);
  auto v4 = SomeData::f1(value);
  // fprintf(stderr, "[          ] v4 = %d | value = %ld | 0x%x\n", v4, value, value);
  EXPECT_EQ(v4, 15) << "SomeData v4 failed";

  SomeData::set_f2(value, 12);
  auto v5 = SomeData::f2(value);
  // fprintf(stderr, "[          ] v5 = %d, value = %ld | 0x%x\n", v5, value, value);
  EXPECT_EQ(v5, 12) << "SomeData v5 failed";

  auto v6 = SomeData::f1(value);
  EXPECT_EQ(v6, 15) << "SomeData v6 failed";

  SomeData::set_f3(value, 36);
  auto v7 = SomeData::f3(value);
  // fprintf(stderr, "[          ] v7 = %d, value = %ld | 0x%x\n", v7, value, value);
  EXPECT_EQ(v7, 36) << "SomeData v7 failed";

  value = 0x85301830;

  auto v8 = SomeData::f4(value);
  EXPECT_EQ(v8, 2131) << "SomeData v8 failed";

  v8++;
  SomeData::set_f4(value, v8);
  auto v9 = SomeData::f4(value);
  EXPECT_EQ(v9, 2132) << "SomeData v9 failed";
  EXPECT_EQ(value, 0x85401830) << "SomeData v9 failed";

} // BitFieldInteger

TEST(kalypsso_shared_test, BitFieldInteger2)
{

  class SomeData2 : public BitFieldInteger<uint64_t>
  {
  public:
    using BitFieldInteger::BitFieldInteger;

    DECLARE_CASTED_FIELD(f1, 0, 12, uint16_t)
    DECLARE_CASTED_FIELD(f2, 12, 8, uint8_t)
    DECLARE_CASTED_FIELD(f3, 20, 20, uint32_t)
    DECLARE_CASTED_FIELD(f4, 40, 24, uint32_t)
  };

  uint64_t value = (1LL << 8) + (1LL << 12) + (1LL << 13) + (1LL << 40);

  // fprintf(stderr, "value=%lu 0x%llx\n", value, value);

  auto v1 = SomeData2::f1(value);
  EXPECT_EQ(v1, 256) << "SomeData2 v1 failed";

  auto v2 = SomeData2::f2(value);
  EXPECT_EQ(v2, 3) << "SomeData2 v2 failed";

  auto v3 = SomeData2::f3(value);
  EXPECT_EQ(v3, 0) << "SomeData2 v3 failed";

  auto v31 = SomeData2::f4(value);
  EXPECT_EQ(v31, 1) << "SomeData2 v31 failed";

  // value = SomeData2::cset_f1(value, 15);
  SomeData2::set_f1(value, 15);
  auto v4 = SomeData2::f1(value);
  // fprintf(stderr, "[          ] v4 = %d | value = %ld | 0x%llx\n", v4, value, value);
  EXPECT_EQ(v4, 15) << "SomeData2 v4 failed";

  SomeData2::set_f2(value, 12);
  auto v5 = SomeData2::f2(value);
  // fprintf(stderr, "[          ] v5 = %d, value = %ld | 0x%llx\n", v5, value, value);
  EXPECT_EQ(v5, 12) << "SomeData2 v5 failed";

  auto v6 = SomeData2::f1(value);
  EXPECT_EQ(v6, 15) << "SomeData2 v6 failed";

  SomeData2::set_f3(value, 36);
  auto v7 = SomeData2::f3(value);
  // fprintf(stderr, "[          ] v7 = %d, value = %ld | 0x%x\n", v7, value, value);
  EXPECT_EQ(v7, 36) << "SomeData2 v7 failed";

  value = 0x90000000000;

  auto v8 = SomeData2::f4(value);
  EXPECT_EQ(v8, 9) << "SomeData2 v8 failed";

  v8++;
  SomeData2::set_f4(value, v8);
  auto v9 = SomeData2::f4(value);
  EXPECT_EQ(v9, 10) << "SomeData2 v9 failed";
  EXPECT_EQ(value, 0xA0000000000) << "SomeData2 v9 failed";

} // BitFieldInteger2

TEST(kalypsso_shared_test, Orchard)
{
  // class Orchard : public BitFieldInteger<uint64_t>
  // {
  // public:
  //   using BitFieldInteger::BitFieldInteger;
  //   using key_t = uint64_t;

  //   DECLARE_CASTED_FIELD(level, 0, 6, uint8_t)
  //   DECLARE_CASTED_FIELD(octant, 6, 44, uint64_t)
  //   DECLARE_CASTED_FIELD(tree, 50, 64, uint16_t)
  // };

  uint64_t value = (1LL << 11) + (1LL << 4);

  auto v1 = orchard_key_t<2>::level(value);
  EXPECT_EQ(v1, 4) << "orchard_key_t<2> v1 failed";

  auto v2 = orchard_key_t<2>::morton_octant(value);
  EXPECT_EQ(v2, 8) << "orchard_key_t<2> v2 failed";

  auto v3 = orchard_key_t<2>::morton_tree(value);
  EXPECT_EQ(v3, 0) << "orchard_key_t<2> v3 failed";

  Kokkos::Array<uint16_t, 2> tree_coord{ 0, 1 };
  Kokkos::Array<uint32_t, 2> octant_coord{ 2097152, 0 };
  // Kokkos::Array<uint32_t, 2> octant_coord{ 0, 0 };

  uint16_t oct_lev = 4;

  uint64_t value2 = orchard_key_t<2>::encode_orchard(tree_coord, octant_coord, oct_lev);

  auto oct_lev2 = orchard_key_t<2>::level(value2);
  // printf("value2=%llu  value2=0x%llx | level = %llu\n", value2, value2, oct_lev2);
  printf("value2=%lu  value2=0x%lx | level = %d\n", value2, value2, oct_lev2);

  // uint64_t some_number = exponent2<uint64_t>(40);
  // printf("exponent2 : %llx\n", some_number);

} // Orchard

} // namespace kalypsso
