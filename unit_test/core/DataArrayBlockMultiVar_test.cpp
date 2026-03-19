// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/DataArrayBlockMultiVar.h>

#include "gtest/gtest.h"

namespace kalypsso
{

// ================================================================================================
// ================================================================================================
void
run_data_array_block_multi_var_2d()
{
  constexpr int DIM = 2;
  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<DIM, uint32_t, DefaultDevice>;

  const block_size_t<DIM>                 block_size{ 3, 5 };
  const uint32_t                          num_octs_1 = 5;
  Kokkos::View<uint32_t *, DefaultDevice> num_vars_1("Num vars per octant 1", num_octs_1);

  {
    auto host_num_vars_1 = Kokkos::create_mirror_view(num_vars_1);
    host_num_vars_1(0) = 2; //
    host_num_vars_1(1) = 1; // +---+---+---+---+---+---+---+
    host_num_vars_1(2) = 0; // | 0 | 0 | 1 | 3 | 3 | 3 | 4 |
    host_num_vars_1(3) = 3; // +---+---+---+---+---+---+---+
    host_num_vars_1(4) = 1; //
    Kokkos::deep_copy(num_vars_1, host_num_vars_1);
  }

  DataArrayBlockMultiVar_t data("data", block_size, num_vars_1);

  { // Metadata checks
    auto storage = data.storage();
    auto offsets = data.offsets();

    auto offsets_host =
      DataArrayBlockMultiVar_t::Offsets_t::create_host_mirror_view_and_copy(offsets);
    EXPECT_EQ(offsets.size(), num_octs_1 + 1);
    EXPECT_EQ(offsets_host(0), 0); // <- always 0
    EXPECT_EQ(offsets_host(1), 2);
    EXPECT_EQ(offsets_host(2), 3);
    EXPECT_EQ(offsets_host(3), 3);
    EXPECT_EQ(offsets_host(4), 6);
    EXPECT_EQ(offsets_host(5), 7); // <- always the number of var blocks
    EXPECT_EQ(storage.num_quadrants(), 7);
    EXPECT_EQ(storage.num_vars(), 1); // <- always 1
    EXPECT_EQ(storage.num_cells(), Kokkos::dim_prod(block_size));
    EXPECT_EQ(storage.block_size(), block_size);
  }

  { // Num vars
    auto data_host = DataArrayBlockMultiVar_t::create_host_mirror_view_and_copy_offsets(data);
    auto host_num_vars_1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, num_vars_1);

    for (uint32_t i_oct = 0; i_oct < host_num_vars_1.size(); i_oct++)
      EXPECT_EQ(data_host.num_vars(i_oct), host_num_vars_1(i_oct));
  }

  { // Flat indices
    auto data_host = DataArrayBlockMultiVar_t::create_host_mirror_view_and_copy(data);

    EXPECT_EQ(flat_mv_block_index(1, 0, data_host.offsets()), 1);
    EXPECT_EQ(flat_mv_block_index(0, 1, data_host.offsets()), 2);
    EXPECT_EQ(flat_mv_block_index(1, 3, data_host.offsets()), 4);
    EXPECT_EQ(flat_mv_block_index(0, 4, data_host.offsets()), 6);

    MVBlockIndex_t ret;

    ret = flat_mv_block_index_unravel(1, data_host.offsets());
    EXPECT_EQ(ret[0], 1);
    EXPECT_EQ(ret[1], 0);

    ret = flat_mv_block_index_unravel(2, data_host.offsets());
    EXPECT_EQ(ret[0], 0);
    EXPECT_EQ(ret[1], 1);

    ret = flat_mv_block_index_unravel(3, data_host.offsets());
    EXPECT_EQ(ret[0], 0);
    EXPECT_EQ(ret[1], 3);

    ret = flat_mv_block_index_unravel(5, data_host.offsets());
    EXPECT_EQ(ret[0], 2);
    EXPECT_EQ(ret[1], 3);

    ret = flat_mv_block_index_unravel(6, data_host.offsets());
    EXPECT_EQ(ret[0], 0);
    EXPECT_EQ(ret[1], 4);

    ret = flat_mv_block_index_unravel(1'000'000, data_host.offsets());
    EXPECT_EQ(ret[1], num_octs_1);
  }

  const uint32_t                          num_octs_2 = num_octs_1 + 2;
  Kokkos::View<uint32_t *, DefaultDevice> num_vars_2("Num vars per octant 2", num_octs_2);

  {
    auto host_num_vars_2 = Kokkos::create_mirror_view(num_vars_2);
    host_num_vars_2(0) = 0; //
    host_num_vars_2(1) = 0; // +---+---+
    host_num_vars_2(2) = 2; // | 2 | 2 |
    host_num_vars_2(3) = 0; // +---+---+
    host_num_vars_2(4) = 0; //
    host_num_vars_2(5) = 0; //
    host_num_vars_2(6) = 0; //
    Kokkos::deep_copy(num_vars_2, host_num_vars_2);
  }

  data.reorganize(num_vars_2);

  { // Metadata checks
    auto storage = data.storage();
    auto offsets = data.offsets();

    auto offsets_host =
      DataArrayBlockMultiVar_t::Offsets_t::create_host_mirror_view_and_copy(offsets);
    EXPECT_EQ(offsets.size(), num_octs_2 + 1);
    EXPECT_EQ(offsets_host(0), 0); // <- always 0
    EXPECT_EQ(offsets_host(1), 0);
    EXPECT_EQ(offsets_host(2), 0);
    EXPECT_EQ(offsets_host(3), 2);
    EXPECT_EQ(offsets_host(4), 2);
    EXPECT_EQ(offsets_host(5), 2);
    EXPECT_EQ(offsets_host(6), 2);
    EXPECT_EQ(offsets_host(7), 2); // <- always the number of var blocks
    EXPECT_EQ(storage.num_quadrants(), 2);
    EXPECT_EQ(storage.num_vars(), 1); // <- always 1
    EXPECT_EQ(storage.num_cells(), Kokkos::dim_prod(block_size));
    EXPECT_EQ(storage.block_size(), block_size);
  }

  { // Flat indices
    auto data_host = DataArrayBlockMultiVar_t::create_host_mirror_view_and_copy(data);

    EXPECT_EQ(flat_mv_block_index(0, 2, data_host.offsets()), 0);
    EXPECT_EQ(flat_mv_block_index(1, 2, data_host.offsets()), 1);

    MVBlockIndex_t ret;

    ret = flat_mv_block_index_unravel(0, data_host.offsets());
    EXPECT_EQ(ret[0], 0);
    EXPECT_EQ(ret[1], 2);

    ret = flat_mv_block_index_unravel(1, data_host.offsets());
    EXPECT_EQ(ret[0], 1);
    EXPECT_EQ(ret[1], 2);

    ret = flat_mv_block_index_unravel(1'000'000, data_host.offsets());
    EXPECT_EQ(ret[1], num_octs_2);
  }

  data.reorganize(num_vars_1);

  { // Setup data
    const auto nb_cells = data.num_cells() * data.storage().num_quadrants();
    Kokkos::RangePolicy<typename DefaultDevice::execution_space> policy(0, nb_cells);

    Kokkos::parallel_for(
      "fill", policy, KOKKOS_LAMBDA(const size_t flat_index) {
        const auto index = data.flat_index_unravel(flat_index);
        data(index[IX], index[IY], index[DIM], index[DIM + 1]) = flat_index;
      });
  }

  { // Data checks
    auto data_host = DataArrayBlockMultiVar_t::create_host_mirror_view_and_copy(data);

    EXPECT_EQ(data_host(2, 2, 1, 0), 23);
    EXPECT_EQ(data_host(2, 4, 0, 1), 44);
    EXPECT_EQ(data_host(1, 0, 1, 3), 61);
    EXPECT_EQ(data_host(1, 3, 0, 4), 100);
  }

  data.reorganize_and_reset(num_vars_1);

  { // Data checks
    auto data_host = DataArrayBlockMultiVar_t::create_host_mirror_view_and_copy(data);

    EXPECT_EQ(data_host(2, 2, 1, 0), 0);
    EXPECT_EQ(data_host(2, 4, 0, 1), 0);
    EXPECT_EQ(data_host(1, 0, 1, 3), 0);
    EXPECT_EQ(data_host(1, 3, 0, 4), 0);
  }

  block_size_t<5 - DIM>                               other_block_size{ 2, 2, 3 };
  DataArrayBlockMultiVar<5 - DIM, int, DefaultDevice> other_data("other data", other_block_size);
  other_data.align_with(data);

  { // Metadata checks
    auto storage = other_data.storage();
    auto offsets = other_data.offsets();

    auto offsets_host =
      DataArrayBlockMultiVar_t::Offsets_t::create_host_mirror_view_and_copy(offsets);
    EXPECT_EQ(offsets.size(), num_octs_1 + 1);
    EXPECT_EQ(offsets_host(0), 0); // <- always 0
    EXPECT_EQ(offsets_host(1), 2);
    EXPECT_EQ(offsets_host(2), 3);
    EXPECT_EQ(offsets_host(3), 3);
    EXPECT_EQ(offsets_host(4), 6);
    EXPECT_EQ(offsets_host(5), 7); // <- always the number of var blocks
    EXPECT_EQ(storage.num_quadrants(), 7);
    EXPECT_EQ(storage.num_vars(), 1); // <- always 1
    EXPECT_EQ(storage.num_cells(), Kokkos::dim_prod(other_block_size));
    EXPECT_EQ(storage.block_size(), other_block_size);
  }
}

// ================================================================================================
// ================================================================================================
void
run_data_array_block_multi_var_3d()
{
  constexpr int DIM = 3;
  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<DIM, uint32_t, DefaultDevice>;

  const block_size_t<DIM>                 block_size{ 4, 3, 2 };
  const uint32_t                          num_octs_1 = 5;
  Kokkos::View<uint32_t *, DefaultDevice> num_vars_1("Num vars per octant", num_octs_1);

  {
    auto host_num_vars_1 = Kokkos::create_mirror_view(num_vars_1);
    host_num_vars_1(0) = 0; //
    host_num_vars_1(1) = 3; // +---+---+---+---+---+---+---+---+
    host_num_vars_1(2) = 1; // | 1 | 1 | 1 | 2 | 3 | 3 | 3 | 3 |
    host_num_vars_1(3) = 4; // +---+---+---+---+---+---+---+---+
    host_num_vars_1(4) = 0; //
    Kokkos::deep_copy(num_vars_1, host_num_vars_1);
  }

  DataArrayBlockMultiVar_t data("data", block_size, num_vars_1);

  { // Metadata checks
    auto storage = data.storage();
    auto offsets = data.offsets();

    auto offsets_host =
      DataArrayBlockMultiVar_t::Offsets_t::create_host_mirror_view_and_copy(offsets);
    EXPECT_EQ(offsets.size(), num_octs_1 + 1);
    EXPECT_EQ(offsets_host(0), 0); // <- always 0
    EXPECT_EQ(offsets_host(1), 0);
    EXPECT_EQ(offsets_host(2), 3);
    EXPECT_EQ(offsets_host(3), 4);
    EXPECT_EQ(offsets_host(4), 8);
    EXPECT_EQ(offsets_host(5), 8); // <- always the number of var blocks
    EXPECT_EQ(storage.num_quadrants(), 8);
    EXPECT_EQ(storage.num_vars(), 1); // <- always 1
    EXPECT_EQ(storage.num_cells(), Kokkos::dim_prod(block_size));
    EXPECT_EQ(storage.block_size(), block_size);
  }

  { // Num vars
    auto data_host = DataArrayBlockMultiVar_t::create_host_mirror_view_and_copy_offsets(data);
    auto host_num_vars_1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, num_vars_1);

    for (uint32_t i_oct = 0; i_oct < host_num_vars_1.size(); i_oct++)
      EXPECT_EQ(data_host.num_vars(i_oct), host_num_vars_1(i_oct));
  }

  { // Flat indices
    auto data_host = DataArrayBlockMultiVar_t::create_host_mirror_view_and_copy(data);

    EXPECT_EQ(flat_mv_block_index(2, 1, data_host.offsets()), 2);
    EXPECT_EQ(flat_mv_block_index(0, 2, data_host.offsets()), 3);
    EXPECT_EQ(flat_mv_block_index(3, 3, data_host.offsets()), 7);

    MVBlockIndex_t ret;

    ret = flat_mv_block_index_unravel(1, data_host.offsets());
    EXPECT_EQ(ret[0], 1);
    EXPECT_EQ(ret[1], 1);

    ret = flat_mv_block_index_unravel(2, data_host.offsets());
    EXPECT_EQ(ret[0], 2);
    EXPECT_EQ(ret[1], 1);

    ret = flat_mv_block_index_unravel(3, data_host.offsets());
    EXPECT_EQ(ret[0], 0);
    EXPECT_EQ(ret[1], 2);

    ret = flat_mv_block_index_unravel(5, data_host.offsets());
    EXPECT_EQ(ret[0], 1);
    EXPECT_EQ(ret[1], 3);

    ret = flat_mv_block_index_unravel(7, data_host.offsets());
    EXPECT_EQ(ret[0], 3);
    EXPECT_EQ(ret[1], 3);

    ret = flat_mv_block_index_unravel(1'000'000, data_host.offsets());
    EXPECT_EQ(ret[1], num_octs_1);
  }

  const uint32_t                          num_octs_2 = num_octs_1 + 2;
  Kokkos::View<uint32_t *, DefaultDevice> num_vars_2("Num vars per octant 2", num_octs_2);

  {
    auto host_num_vars_2 = Kokkos::create_mirror_view(num_vars_2);
    host_num_vars_2(0) = 0; //
    host_num_vars_2(1) = 0; // +---+---+
    host_num_vars_2(2) = 2; // | 2 | 2 |
    host_num_vars_2(3) = 0; // +---+---+
    host_num_vars_2(4) = 0; //
    host_num_vars_2(5) = 0; //
    host_num_vars_2(6) = 0; //
    Kokkos::deep_copy(num_vars_2, host_num_vars_2);
  }

  data.reorganize(num_vars_2);

  { // Metadata checks
    auto storage = data.storage();
    auto offsets = data.offsets();

    auto offsets_host =
      DataArrayBlockMultiVar_t::Offsets_t::create_host_mirror_view_and_copy(offsets);
    EXPECT_EQ(offsets.size(), num_octs_2 + 1);
    EXPECT_EQ(offsets_host(0), 0); // <- always 0
    EXPECT_EQ(offsets_host(1), 0);
    EXPECT_EQ(offsets_host(2), 0);
    EXPECT_EQ(offsets_host(3), 2);
    EXPECT_EQ(offsets_host(4), 2);
    EXPECT_EQ(offsets_host(5), 2);
    EXPECT_EQ(offsets_host(6), 2);
    EXPECT_EQ(offsets_host(7), 2); // <- always the number of var blocks
    EXPECT_EQ(storage.num_quadrants(), 2);
    EXPECT_EQ(storage.num_vars(), 1); // <- always 1
    EXPECT_EQ(storage.num_cells(), Kokkos::dim_prod(block_size));
    EXPECT_EQ(storage.block_size(), block_size);
  }

  { // Flat indices
    auto data_host = DataArrayBlockMultiVar_t::create_host_mirror_view_and_copy(data);

    EXPECT_EQ(flat_mv_block_index(0, 2, data_host.offsets()), 0);
    EXPECT_EQ(flat_mv_block_index(1, 2, data_host.offsets()), 1);

    MVBlockIndex_t ret;

    ret = flat_mv_block_index_unravel(0, data_host.offsets());
    EXPECT_EQ(ret[0], 0);
    EXPECT_EQ(ret[1], 2);

    ret = flat_mv_block_index_unravel(1, data_host.offsets());
    EXPECT_EQ(ret[0], 1);
    EXPECT_EQ(ret[1], 2);

    ret = flat_mv_block_index_unravel(1'000'000, data_host.offsets());
    EXPECT_EQ(ret[1], num_octs_2);
  }

  data.reorganize(num_vars_1);

  { // Setup data
    const auto nb_cells = data.num_cells() * data.storage().num_quadrants();
    Kokkos::RangePolicy<typename DefaultDevice::execution_space> policy(0, nb_cells);

    Kokkos::parallel_for(
      "fill", policy, KOKKOS_LAMBDA(const size_t flat_index) {
        const auto index = data.flat_index_unravel(flat_index);
        data(index[IX], index[IY], index[IZ], index[DIM], index[DIM + 1]) = flat_index;
      });
  }

  { // Data checks
    auto data_host = DataArrayBlockMultiVar_t::create_host_mirror_view_and_copy(data);

    EXPECT_EQ(data_host(3, 2, 1, 1, 1), 47);
    EXPECT_EQ(data_host(0, 1, 1, 0, 2), 88);
    EXPECT_EQ(data_host(3, 1, 0, 2, 3), 151);
  }

  data.reorganize_and_reset(num_vars_1);

  { // Data checks
    auto data_host = DataArrayBlockMultiVar_t::create_host_mirror_view_and_copy(data);

    EXPECT_EQ(data_host(3, 2, 1, 1, 1), 0);
    EXPECT_EQ(data_host(0, 1, 1, 0, 2), 0);
    EXPECT_EQ(data_host(3, 1, 0, 2, 3), 0);
  }

  block_size_t<5 - DIM>                               other_block_size{ 2, 2 };
  DataArrayBlockMultiVar<5 - DIM, int, DefaultDevice> other_data("other data", other_block_size);
  other_data.align_with(data);

  { // Metadata checks
    auto storage = other_data.storage();
    auto offsets = other_data.offsets();

    auto offsets_host =
      DataArrayBlockMultiVar_t::Offsets_t::create_host_mirror_view_and_copy(offsets);
    EXPECT_EQ(offsets.size(), num_octs_1 + 1);
    EXPECT_EQ(offsets_host(0), 0); // <- always 0
    EXPECT_EQ(offsets_host(1), 0);
    EXPECT_EQ(offsets_host(2), 3);
    EXPECT_EQ(offsets_host(3), 4);
    EXPECT_EQ(offsets_host(4), 8);
    EXPECT_EQ(offsets_host(5), 8); // <- always the number of var blocks
    EXPECT_EQ(storage.num_quadrants(), 8);
    EXPECT_EQ(storage.num_vars(), 1); // <- always 1
    EXPECT_EQ(storage.num_cells(), Kokkos::dim_prod(other_block_size));
    EXPECT_EQ(storage.block_size(), other_block_size);
  }
}

// ================================================================================================
// ================================================================================================
TEST(kalypsso_shared_test, DataArrayBlockMultiVar_2d)
{
  run_data_array_block_multi_var_2d();
}

// ================================================================================================
// ================================================================================================
TEST(kalypsso_shared_test, DataArrayBlockMultiVar_3d)
{
  run_data_array_block_multi_var_3d();
}

} // namespace kalypsso
