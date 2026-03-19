// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat à l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define KALYPSSO_CORE_USE_NEW_DATA_ARRAY_GHOSTED_BLOCK_IMPL
#include <kalypsso/core/kalypsso_data_container.h>

#include <cstdint>

#include "gtest/gtest.h"

#include "kalypsso_unittest_utils.h"

namespace kalypsso
{

// ====================================================================
// ====================================================================
void
check_DataArray_resize()
{
  {
    using device_t = HostDevice;
    using exec_space = Kokkos::DefaultHostExecutionSpace;

    using DataArray_t = DataArray<int64_t, device_t>;

    DataArrayUtils::set_growth_rate(KALYPSSO_NUM(1.2));
    const size_t num_elts = 18;
    auto         data = DataArray_t("data", num_elts);

    EXPECT_EQ(data.num_elements(), 18);
    EXPECT_EQ(data.allocated_size_in_bytes(), 21 * sizeof(int64_t));

    data.resize(32);
    EXPECT_EQ(data.allocated_size_in_bytes(), 38 * sizeof(int64_t));

    DataArrayUtils::set_growth_rate(KALYPSSO_NUM(1.0));
  }
}

// ====================================================================
// ====================================================================
void
check_DataArrayBlock_resize_2d()
{
  constexpr int dim = 2;

  // using device_t = DefaultDevice;
  // using exec_space = Kokkos::DefaultExecutionSpace;
  using device_t = HostDevice;
  using exec_space = Kokkos::DefaultHostExecutionSpace;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  const auto     bSize = block_size_t<dim>{ 3, 5 };
  const uint32_t num_vars = 2;
  const uint32_t num_octs = 4;
  auto           data = DataArrayBlock_t("data", bSize, num_vars, num_octs);

  // check that capacity growth rate is equal to default
  EXPECT_REAL_EQ(DataArrayUtils::m_capacity_growth_rate,
                 DataArrayUtils::m_capacity_growth_rate_default);

  const auto num_elements = Kokkos::dim_prod(bSize) * num_vars * num_octs;
  uint64_t   num_bytes = num_elements * sizeof(real_t);

  // assuming DataArrayUtils::m_capacity_growth_rate_default is 1.0
  EXPECT_EQ(data.allocated_size_in_bytes(), num_bytes);

  {
    data.resize(8);
    num_bytes = Kokkos::dim_prod(bSize) * num_vars * 8 * sizeof(real_t);
    EXPECT_EQ(data.allocated_size_in_bytes(), num_bytes);
  }

  DataArrayUtils::set_growth_rate(KALYPSSO_NUM(1.2));

  {
    data.resize(9);
    // 3*5*2*9*1.2 = 324
    num_bytes = 324 * sizeof(real_t);
    EXPECT_EQ(data.allocated_size_in_bytes(), num_bytes);
  }

  {
    // capacity should not change because new size is smaller than capacity
    data.resize(10);
    // 3*5*2*10 = 300 < 324
    num_bytes = 324 * sizeof(real_t);
    EXPECT_EQ(data.allocated_size_in_bytes(), num_bytes);
  }

  {
    // capacity should change because new is larger than capacity
    data.resize(13);
    // 3*5*2*13*1.2 = 468 > 324
    num_bytes = 468 * sizeof(real_t);
    EXPECT_EQ(data.allocated_size_in_bytes(), num_bytes);
  }

  DataArrayUtils::set_growth_rate(DataArrayUtils::m_capacity_growth_rate_default);
}

// ====================================================================
// ====================================================================
void
run_kalypsso_data_container_DataArrayBlock_2d()
{
  constexpr int dim = 2;

  // using device_t = DefaultDevice;
  // using exec_space = Kokkos::DefaultExecutionSpace;
  using device_t = HostDevice;
  using exec_space = Kokkos::DefaultHostExecutionSpace;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  const auto     bSize = block_size_t<dim>{ 3, 5 };
  const uint32_t num_vars = 2;
  const uint32_t num_octs = 4;
  auto           data = DataArrayBlock_t("data", bSize, num_vars, num_octs);

  EXPECT_EQ(bSize, data.block_size());
  EXPECT_EQ(data.num_cells(), 15);

  const auto nbCellsTotal = data.num_cells() * data.num_vars() * data.num_quadrants();
  EXPECT_EQ(nbCellsTotal, 120);

  {
    // 3 * (3*5*2) + 0 * (3*5) + 1*3 + 2
    EXPECT_EQ(data.flat_index(2, 1, 0, 3), 95);

    // 1 * (3*5*2) + 1 * (3*5) + 1*3 + 1
    EXPECT_EQ(data.flat_index(1, 1, 1, 1), 49);
  }

  {
    size_t flat_index = 95;
    auto   mindex = flat_index_unravel<dim>(flat_index, bSize, num_vars);
    EXPECT_EQ(mindex[dim + 1], 3);
    EXPECT_EQ(mindex[dim], 0);
    EXPECT_EQ(mindex[IY], 1);
    EXPECT_EQ(mindex[IX], 2);
  }

  {
    size_t flat_index = 49;
    auto   mindex = flat_index_unravel<dim>(flat_index, bSize, num_vars);
    EXPECT_EQ(mindex[dim + 1], 1);
    EXPECT_EQ(mindex[dim], 1);
    EXPECT_EQ(mindex[IY], 1);
    EXPECT_EQ(mindex[IX], 1);
  }

  // change shape (equal size)
  {
    const auto new_shape = block_size_t<dim>{ 5, 3 };
    data.reshape(new_shape);

    // 3 * (5*3*2) + 0 * (5*3) + 1*5 + 0
    EXPECT_EQ(data.flat_index(0, 1, 0, 3), 95);

    // 1 * (5*3*2) + 1 * (5*3) + 0*5 + 4
    EXPECT_EQ(data.flat_index(4, 0, 1, 1), 49);
  }

  // change shape (smaller size)
  {
    const auto new_shape = block_size_t<dim>{ 2, 3 };
    data.reshape(new_shape);

    EXPECT_EQ(data.shape()[IX], 2);
    EXPECT_EQ(data.shape()[IY], 3);

    // 7 * (2*3*2) + 1 * (2*3) + 2*2 + 1
    EXPECT_EQ(data.flat_index(1, 2, 1, 7), 95);
  }

  // change shape (larger size => should be no-op)
  {
    const auto old_shape = data.shape();
    const auto new_shape = block_size_t<dim>{ 12, 3 };
    data.reshape(new_shape);

    EXPECT_EQ(data.shape(), old_shape);
  }

  data.shape_reset();
  auto nbCellsTotal2 = data.num_cells() * data.num_vars() * data.num_quadrants();

  Kokkos::parallel_for(
    "fill", Kokkos::RangePolicy<exec_space>(0, nbCellsTotal2), KOKKOS_LAMBDA(uint32_t flat_index) {
      const auto mindex = flat_index_unravel<dim>(flat_index, data.block_size(), data.num_vars());
      data(mindex[IX], mindex[IY], mindex[dim], mindex[dim + 1]) = flat_index;
    });

  auto data_h = DataArrayBlock_t::create_host_mirror_view_and_copy(data);

  auto ij = coord_t<2>{ 2, 4 };
  EXPECT_EQ(data_h(14, 0, 0), 14);
  EXPECT_EQ(data_h(2, 4, 0, 0), 14);
  EXPECT_EQ(data_h(ij, 0, 0), 14);

  ij[IX] = 1;
  EXPECT_EQ(data_h(13, 1, 3), 118);
  EXPECT_EQ(data_h(1, 4, 1, 3), 118);
  EXPECT_EQ(data_h(ij, 1, 3), 118);

  // resize with 5 octants
  data.resize_and_reset(5);
  auto data_h2 = DataArrayBlock_t::create_host_mirror_view_and_copy(data);
  EXPECT_EQ(data_h2(14, 0, 0), 0);
  EXPECT_EQ(data_h2(13, 1, 3), 0);

} // run_kalypsso_data_container_DataArrayBlock_2d

// ====================================================================
// ====================================================================
void
run_kalypsso_data_container_DataArrayBlock_3d()
{
  constexpr int dim = 3;

  // using device_t = DefaultDevice;
  // using exec_space = Kokkos::DefaultExecutionSpace;
  using device_t = HostDevice;
  using exec_space = Kokkos::DefaultHostExecutionSpace;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  const auto     bSize = block_size_t<dim>{ 4, 3, 2 };
  const uint32_t num_vars = 2;
  const uint32_t num_octs = 4;
  auto           data = DataArrayBlock_t("data", bSize, num_vars, num_octs);

  EXPECT_EQ(bSize, data.block_size());
  EXPECT_EQ(data.num_cells(), 24);

  const auto nbCellsTotal = data.num_cells() * data.num_vars() * data.num_quadrants();
  EXPECT_EQ(nbCellsTotal, 192);

  {
    // 1 * (4*3*2*2) + 1 * (4*3*2) + 1 * (4*3) + 2 * (4) + 3
    EXPECT_EQ(data.flat_index(3, 2, 1, 1, 1), 95);

    // 1 * (4*3*2*2) + 0 * (4*3*2) + 0 * (4*3) + 0 * (4) + 1
    EXPECT_EQ(data.flat_index(1, 0, 0, 0, 1), 49);

    // 2 * (4*3*2*2) + 1 * (4*3*2) + 0 * (4*3) + 1 * (4) + 3
    EXPECT_EQ(data.flat_index(3, 1, 0, 1, 2), 127);
  }

  {
    size_t flat_index = 95;
    auto   mindex = flat_index_unravel<dim>(flat_index, bSize, num_vars);
    EXPECT_EQ(mindex[dim + 1], 1);
    EXPECT_EQ(mindex[dim], 1);
    EXPECT_EQ(mindex[IZ], 1);
    EXPECT_EQ(mindex[IY], 2);
    EXPECT_EQ(mindex[IX], 3);
  }

  {
    size_t flat_index = 49;
    auto   mindex = flat_index_unravel<dim>(flat_index, bSize, num_vars);
    EXPECT_EQ(mindex[dim + 1], 1);
    EXPECT_EQ(mindex[dim], 0);
    EXPECT_EQ(mindex[IZ], 0);
    EXPECT_EQ(mindex[IY], 0);
    EXPECT_EQ(mindex[IX], 1);
  }

  {
    size_t flat_index = 127;
    auto   mindex = flat_index_unravel<dim>(flat_index, bSize, num_vars);
    EXPECT_EQ(mindex[dim + 1], 2);
    EXPECT_EQ(mindex[dim], 1);
    EXPECT_EQ(mindex[IZ], 0);
    EXPECT_EQ(mindex[IY], 1);
    EXPECT_EQ(mindex[IX], 3);
  }

  // change shape (equal size)
  {
    const auto new_shape = block_size_t<dim>{ 2, 3, 4 };
    data.reshape(new_shape);

    // 1 * (2*3*4*2) + 1 * (2*3*4) + 3 * (2*3) + 2 * (2) + 1
    EXPECT_EQ(data.flat_index(1, 2, 3, 1, 1), 95);

    // 1 * (2*3*4*2) + 0 * (2*3*4) + 0 * (2*3) + 1 * (2) + 1
    EXPECT_EQ(data.flat_index(1, 1, 0, 0, 1), 51);
  }

  // change shape (smaller size)
  {
    const auto new_shape = block_size_t<dim>{ 2, 3, 1 };
    data.reshape(new_shape);

    EXPECT_EQ(data.shape()[IX], 2);
    EXPECT_EQ(data.shape()[IY], 3);
    EXPECT_EQ(data.shape()[IZ], 1);

    // 7 * (2*3*1*2) + 1 * (2*3*1) + 0 * (2*3) + 2 * (2) + 1
    EXPECT_EQ(data.flat_index(1, 2, 0, 1, 7), 95);
  }

  // change shape (larger size => should be no-op)
  {
    const auto old_shape = data.shape();
    const auto new_shape = block_size_t<dim>{ 12, 13, 14 };
    data.reshape(new_shape);

    EXPECT_EQ(data.shape(), old_shape);
  }

  data.shape_reset();
  auto nbCellsTotal2 = data.num_cells() * data.num_vars() * data.num_quadrants();

  Kokkos::parallel_for(
    "fill", Kokkos::RangePolicy<exec_space>(0, nbCellsTotal2), KOKKOS_LAMBDA(uint32_t flat_index) {
      const auto mindex = flat_index_unravel<dim>(flat_index, data.block_size(), data.num_vars());
      data(mindex[IX], mindex[IY], mindex[IZ], mindex[3], mindex[4]) = flat_index;
    });

  auto data_h = DataArrayBlock_t::create_host_mirror_view_and_copy(data);

  auto ijk = coord_t<3>{ 2, 0, 1 };
  EXPECT_EQ(data_h(14, 0, 0), 14);
  EXPECT_EQ(data_h(ijk, 0, 0), 14);

  ijk[IX] = 1;
  EXPECT_EQ(data_h(13, 1, 3), 181);
  EXPECT_EQ(data_h(ijk, 1, 3), 181);

  // resize with 5 octants
  data.resize_and_reset(5);
  auto data_h2 = DataArrayBlock_t::create_host_mirror_view_and_copy(data);
  EXPECT_EQ(data_h2(14, 0, 0), 0);
  EXPECT_EQ(data_h2(13, 1, 3), 0);

} // run_kalypsso_data_container_DataArrayBlock_3d

// ====================================================================
// ====================================================================
void
run_kalypsso_data_container_FaceDataArrayBlock_2d_noghost()
{
  constexpr int dim = 2;

  // using device_t = DefaultDevice;
  // using exec_space = Kokkos::DefaultExecutionSpace;
  using device_t = HostDevice;
  using exec_space = Kokkos::DefaultHostExecutionSpace;

  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  const auto     bSize = block_size_t<dim>{ 3, 5 };
  const auto     gSize = 0;
  const uint32_t num_octs = 4;
  auto           facedata = FaceDataArrayBlock_t("facedata", bSize, gSize, num_octs);

  EXPECT_EQ(bSize, facedata.cell_block_size_inner());
  EXPECT_EQ(facedata.num_elements_per_octant(), 53); // (3+1)*5 + 3*(5+1) + 3*5

  EXPECT_EQ(facedata.shift()[IX], 0);
  EXPECT_EQ(facedata.shift()[IY], 0);

  EXPECT_EQ(facedata.cell_block_size()[IX], 3);
  EXPECT_EQ(facedata.cell_block_size()[IY], 5);

  {
    // 0 + 2 + 1*(3+1) + 53*3
    EXPECT_EQ(facedata.flat_index<IX>(2, 1, 3), 165);

    // (3+1)*5 + 1 + 1*(3) + 53
    EXPECT_EQ(facedata.flat_index<IY>(1, 1, 1), 77);
  }

  {
    size_t flat_index = 165 - 3 * 53;
    auto   mindex =
      face_flat_index_unravel<dim>(flat_index, bSize, facedata.offsets(), facedata.shift());
    EXPECT_EQ(mindex[IX], 2);
    EXPECT_EQ(mindex[IY], 1);
    EXPECT_EQ(mindex[dim], 0);
  }

  {
    size_t flat_index = 77 - 53;
    auto   mindex =
      face_flat_index_unravel<dim>(flat_index, bSize, facedata.offsets(), facedata.shift());
    EXPECT_EQ(mindex[IX], 1);
    EXPECT_EQ(mindex[IY], 1);
    EXPECT_EQ(mindex[dim], 1);
  }

  const auto nbFacesPerLeaf = facedata.num_elements_per_octant();
  const auto nbFacesTotal = nbFacesPerLeaf * facedata.num_quadrants();

  Kokkos::parallel_for(
    "fill", Kokkos::RangePolicy<exec_space>(0, nbFacesTotal), KOKKOS_LAMBDA(uint32_t global_index) {
      const auto iOct = global_index / nbFacesPerLeaf;
      const auto face_flat_index = global_index - iOct * nbFacesPerLeaf;

      const auto mindex = face_flat_index_unravel<dim>(
        face_flat_index, facedata.cell_block_size(), facedata.offsets(), facedata.shift());

      facedata(mindex[IX], mindex[IY], mindex[dim], iOct) = face_flat_index;
    });

  const auto facedata_x_h = FaceDataArrayBlock_t::to_DataArrayBlockCentered(facedata, IX);
  const auto facedata_y_h = FaceDataArrayBlock_t::to_DataArrayBlockCentered(facedata, IY);

  EXPECT_EQ(facedata_x_h(0, 0, 0), 0.5);
  EXPECT_EQ(facedata_y_h(0, 0, 0), 21.5); // (20+23)/2

} // run_kalypsso_data_container_FaceDataArrayBlock_2d_noghost

// ====================================================================
// ====================================================================
void
run_kalypsso_data_container_FaceDataArrayBlock_2d_ghost()
{
  constexpr int dim = 2;

  // using device_t = DefaultDevice;
  // using exec_space = Kokkos::DefaultExecutionSpace;
  using device_t = HostDevice;
  using exec_space = Kokkos::DefaultHostExecutionSpace;

  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  const auto     bSize_inner = block_size_t<dim>{ 3, 5 };
  const auto     gSize = 2;
  const auto     bSize = bSize_inner + 2 * gSize;
  const uint32_t num_octs = 4;
  auto           facedata = FaceDataArrayBlock_t("facedata", bSize_inner, gSize, num_octs);

  EXPECT_EQ(bSize_inner[IX], facedata.cell_block_size_inner()[IX]);
  EXPECT_EQ(bSize_inner[IY], facedata.cell_block_size_inner()[IY]);
  EXPECT_EQ(facedata.num_elements_per_octant(), 205); // (7+1)*9 + 7*(9+1) + 7*9

  EXPECT_EQ(facedata.shift()[IX], -2);
  EXPECT_EQ(facedata.shift()[IY], -2);

  {
    // 0 + 2 + 1*(7+1) + 205*3
    EXPECT_EQ(facedata.flat_index<IX>(2, 1, 3), 625);

    // (7+1)*9 + 1 + 1*(7) + 205
    EXPECT_EQ(facedata.flat_index<IY>(1, 1, 1), 285);
  }

  {
    size_t flat_index = 625 - 3 * 205;
    auto   mindex =
      face_flat_index_unravel<dim>(flat_index, bSize, facedata.offsets(), facedata.shift());
    EXPECT_EQ(mindex[IX] + gSize, 2);
    EXPECT_EQ(mindex[IY] + gSize, 1);
    EXPECT_EQ(mindex[dim], 0);
  }

  {
    size_t flat_index = 285 - 205;
    auto   mindex =
      face_flat_index_unravel<dim>(flat_index, bSize, facedata.offsets(), facedata.shift());
    EXPECT_EQ(mindex[IX] + gSize, 1);
    EXPECT_EQ(mindex[IY] + gSize, 1);
    EXPECT_EQ(mindex[dim], 1);
  }

  const auto nbFacesPerLeaf = facedata.num_elements_per_octant();
  const auto nbFacesTotal = nbFacesPerLeaf * facedata.num_quadrants();

  Kokkos::parallel_for(
    "fill", Kokkos::RangePolicy<exec_space>(0, nbFacesTotal), KOKKOS_LAMBDA(uint32_t global_index) {
      const auto iOct = global_index / nbFacesPerLeaf;
      const auto face_flat_index = global_index - iOct * nbFacesPerLeaf;

      const auto mindex = face_flat_index_unravel<dim>(
        face_flat_index, facedata.cell_block_size(), facedata.offsets(), facedata.shift());

      facedata(mindex[IX], mindex[IY], mindex[dim], iOct) = face_flat_index;
    });

  const auto facedata_x_h = FaceDataArrayBlock_t::to_DataArrayBlockCentered(facedata, IX);
  const auto facedata_y_h = FaceDataArrayBlock_t::to_DataArrayBlockCentered(facedata, IY);

  EXPECT_EQ(facedata_x_h(0, 0, 0), 0.5);
  EXPECT_EQ(facedata_y_h(0, 0, 0), 75.5); // (72+79)/2

} // run_kalypsso_data_container_FaceDataArrayBlock_2d_ghost

// ====================================================================
// ====================================================================
void
check_FaceDataArrayBlock_resize_2d()
{
  constexpr int dim = 2;

  // using device_t = DefaultDevice;
  // using exec_space = Kokkos::DefaultExecutionSpace;
  using device_t = HostDevice;
  using exec_space = Kokkos::DefaultHostExecutionSpace;

  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  const auto     bSize = block_size_t<dim>{ 3, 5 };
  const auto     gSize = 0;
  const uint32_t num_octs = 4;
  auto           facedata = FaceDataArrayBlock_t("facedata", bSize, gSize, num_octs);

  const auto num_elements = facedata.num_elements_per_octant() * num_octs;
  uint64_t   num_bytes = num_elements * sizeof(real_t);

  // assuming DataArrayUtils::m_capacity_growth_rate_default is 1.0
  EXPECT_EQ(facedata.allocated_size_in_bytes(), num_bytes);

  {
    facedata.resize(8);
    num_bytes = ((bSize[0] + 1) * bSize[1] + bSize[0] * (bSize[1] + 1) + bSize[0] * bSize[1]) * 8 *
                sizeof(real_t);
    EXPECT_EQ(facedata.allocated_size_in_bytes(), num_bytes);
  }

  DataArrayUtils::set_growth_rate(KALYPSSO_NUM(1.2));

  {
    facedata.resize(9);
    // (4*5+3*6+3*5)*9*1.2 = 572
    num_bytes = 572 * sizeof(real_t);
    EXPECT_EQ(facedata.allocated_size_in_bytes(), num_bytes);
  }

  {
    // capacity should not change because new size is smaller than capacity
    facedata.resize(10);
    // (4*5+3*6+3*5)*10 = 530 < 572
    num_bytes = 572 * sizeof(real_t);
    EXPECT_EQ(facedata.allocated_size_in_bytes(), num_bytes);
  }

  {
    // capacity should change because new is larger than capacity
    facedata.resize(13);
    // (4*5+3*6+3*5)*13*1.2 = 826 > 572
    num_bytes = 826 * sizeof(real_t);
    EXPECT_EQ(facedata.allocated_size_in_bytes(), num_bytes);
  }

  DataArrayUtils::set_growth_rate(DataArrayUtils::m_capacity_growth_rate_default);
}

// ====================================================================
// ====================================================================
void
run_kalypsso_data_container_EdgeDataArrayBlock_2d_noghost()
{
  constexpr int dim = 2;

  // using device_t = DefaultDevice;
  // using exec_space = Kokkos::DefaultExecutionSpace;
  using device_t = HostDevice;
  // using exec_space = Kokkos::DefaultHostExecutionSpace;

  using EdgeDataArrayBlock_t = EdgeDataArrayBlock<dim, real_t, device_t>;

  const auto     bSize = block_size_t<dim>{ 3, 5 };
  const auto     gSize = 0;
  const uint32_t num_octs = 4;
  auto           edgedata = EdgeDataArrayBlock_t("edgedata", bSize, gSize, num_octs);

  EXPECT_EQ(bSize, edgedata.cell_block_size_inner());
  EXPECT_EQ(edgedata.num_elements_per_octant(), 62); // 3*(5+1) + (3+1)*5 + (3+1)*(5+1)

  {
    // 0*3*(5+1) + 2 + 1*3 + 62*3 = 191
    EXPECT_EQ(edgedata.flat_index<IX>(2, 1, 3), 191);

    // 3*(5+1)   + 1 + 1*(3+1) + 62 = 85
    EXPECT_EQ(edgedata.flat_index<IY>(1, 1, 1), 85);
  }

  {
    size_t flat_index = 191 - 3 * 62;
    auto   mindex = edge_flat_index_unravel<dim>(flat_index, bSize, edgedata.offsets());
    EXPECT_EQ(mindex[IX], 2);
    EXPECT_EQ(mindex[IY], 1);
    EXPECT_EQ(mindex[dim], 0);
  }

  {
    size_t flat_index = 85 - 62;
    auto   mindex = edge_flat_index_unravel<dim>(flat_index, bSize, edgedata.offsets());
    EXPECT_EQ(mindex[IX], 1);
    EXPECT_EQ(mindex[IY], 1);
    EXPECT_EQ(mindex[dim], 1);
  }

} // run_kalypsso_data_container_EdgeDataArrayBlock_2d_noghost

// ====================================================================
// ====================================================================
TEST(kalypsso_shared_test, kalypsso_data_container_DataArrayBlock)
{
  run_kalypsso_data_container_DataArrayBlock_2d();
  run_kalypsso_data_container_DataArrayBlock_3d();
  run_kalypsso_data_container_FaceDataArrayBlock_2d_noghost();
  run_kalypsso_data_container_FaceDataArrayBlock_2d_ghost();
  run_kalypsso_data_container_EdgeDataArrayBlock_2d_noghost();
  check_DataArray_resize();
  check_DataArrayBlock_resize_2d();
  check_FaceDataArrayBlock_resize_2d();

} // TEST: kalypsso_shared_test/kalypsso_data_container_DataArrayBlock

#ifdef KALYPSSO_CORE_USE_NEW_DATA_ARRAY_GHOSTED_BLOCK_IMPL
// ====================================================================
// ====================================================================
TEST(kalypsso_shared_test, kalypsso_data_container_DataArrayGhostedBlock2d)
{
  using device_t = HostDevice;

  {
    auto       bSize = block_size_t<2>{ 12, 7 };
    auto       gSize = block_size_t<2>{ 16, 13 };
    shift_t<2> shift{ -2, -3 };
    int        num_vars = 4;
    int        num_quadrants = 10;

    auto ghosted_data = DataArrayGhostedBlock<2, real_t, device_t>(
      bSize, gSize, shift, "data", num_vars, num_quadrants);

    EXPECT_EQ(ghosted_data.num_vars(), 4)
      << "DataArrayGhostedBlock<2, device_t>::num_vars() failed";
    EXPECT_EQ(ghosted_data.ghosted_block_size()[0], 16)
      << "DataArrayGhostedBlock<2, device_t> wrong total block size";
    EXPECT_EQ(ghosted_data.ghosted_block_size()[1], 13)
      << "DataArrayGhostedBlock<2, device_t> wrong total block size";

    EXPECT_EQ(ghosted_data.block_size()[0], 12)
      << "DataArrayGhostedBlock<2, device_t> wrong block size";
    EXPECT_EQ(ghosted_data.block_size()[1], 7)
      << "DataArrayGhostedBlock<2, device_t> wrong block size";

    EXPECT_EQ(Kokkos::dim_prod(ghosted_data.get_block_size_overlap()), 84)
      << "wrong overlap block size";

    auto block_data = ghosted_data.data();

    EXPECT_EQ(block_data.num_cells(), 208) << "Wrong view size";
  }

  {
    using DirectAccess = DataArrayGhostedBlock<2, real_t, device_t>::DirectAccess;

    auto       bSize = block_size_t<2>{ 12, 7 };
    auto       gSize = block_size_t<2>{ 14, 7 };
    shift_t<2> shift{ -1, 0 };
    int        num_vars = 2;
    int        num_quadrants = 6;

    auto ghosted_data = DataArrayGhostedBlock<2, real_t, device_t>(
      bSize, gSize, shift, "data", num_vars, num_quadrants);

    EXPECT_EQ(ghosted_data.num_cells(), 98)
      << "DataArrayGhostedBlock<2, device_t>::num_cells() failed";
    EXPECT_EQ(ghosted_data.num_cells_inner(), 84)
      << "DataArrayGhostedBlock<2, device_t>::num_cells() failed";

    ghosted_data.reshape({ 12, 8 }, { 0, -1 });
    EXPECT_EQ(ghosted_data.num_cells(), 96)
      << "DataArrayGhostedBlock<2, device_t>::num_cells() failed";

    for (int j = -1; j < 7; ++j)
      for (int i = 0; i < 12; ++i)
        ghosted_data(i, j, 0, 3) = i * i + j;

    EXPECT_NEAR(ghosted_data(0, -1, 0, 3), -1.0, 1e-14);
    EXPECT_NEAR(ghosted_data(8, 5, 0, 3), 69.0, 1e-14);

    auto ij = coord_t<2>{ 4, 2 };
    EXPECT_NEAR(ghosted_data(ij, 0, 3), 18.0, 1e-14);

    EXPECT_NEAR(ghosted_data(0, 0, 0, 3, DirectAccess{}), -1.0, 1e-14);
    EXPECT_NEAR(ghosted_data(8, 6, 0, 3, DirectAccess{}), 69.0, 1e-14);
    EXPECT_NEAR(ghosted_data(ij, 0, 3, DirectAccess{}), 17.0, 1e-14);

    ghosted_data.reshape({ 10, 5 }, { -1, -2 });
    EXPECT_EQ(Kokkos::dim_prod(ghosted_data.get_block_size_overlap()), 27)
      << "wrong overlap block size";

    ghosted_data.reshape({ 10, 5 }, { 0, 3 });
    EXPECT_EQ(Kokkos::dim_prod(ghosted_data.get_block_size_overlap()), 40)
      << "wrong overlap block size";
  }

} // TEST: kalypsso_shared_test/kalypsso_data_container_DataArrayGhostedBlock2d

// ====================================================================
// ====================================================================
TEST(kalypsso_shared_test, kalypsso_data_container_DataArrayGhostedBlock3d)
{
  using device_t = HostDevice;

  {
    auto       bSize = block_size_t<3>{ 4, 6, 2 };
    auto       gSize = block_size_t<3>{ 7, 9, 3 };
    shift_t<3> shift{ -2, 0, 1 };
    int        num_vars = 2;
    int        num_quadrants = 4;

    auto ghosted_data = DataArrayGhostedBlock<3, real_t, device_t>(
      bSize, gSize, shift, "data", num_vars, num_quadrants);

    EXPECT_EQ(ghosted_data.num_vars(), 2)
      << "DataArrayGhostedBlock<3, device_t>::num_vars() failed";
    EXPECT_EQ(ghosted_data.ghosted_block_size()[0], 7)
      << "DataArrayGhostedBlock<3, device_t> wrong total block size";
    EXPECT_EQ(ghosted_data.ghosted_block_size()[1], 9)
      << "DataArrayGhostedBlock<3, device_t> wrong total block size";
    EXPECT_EQ(ghosted_data.ghosted_block_size()[2], 3)
      << "DataArrayGhostedBlock<3, device_t> wrong total block size";

    EXPECT_EQ(ghosted_data.block_size()[0], 4)
      << "DataArrayGhostedBlock<3, device_t> wrong block size";
    EXPECT_EQ(ghosted_data.block_size()[1], 6)
      << "DataArrayGhostedBlock<3, device_t> wrong block size";
    EXPECT_EQ(ghosted_data.block_size()[2], 2)
      << "DataArrayGhostedBlock<3, device_t> wrong block size";

    auto block_data = ghosted_data.data();

    EXPECT_EQ(block_data.num_cells(), 189) << "Wrong view size";
  }

  {
    using DirectAccess = DataArrayGhostedBlock<3, real_t, device_t>::DirectAccess;

    auto       bSize = block_size_t<3>{ 4, 6, 8 };
    auto       gSize = block_size_t<3>{ 6, 7, 5 };
    shift_t<3> shift{ -1, -1, 0 };
    int        num_vars = 2;
    int        num_quadrants = 6;

    auto ghosted_data = DataArrayGhostedBlock<3, real_t, device_t>(
      bSize, gSize, shift, "data", num_vars, num_quadrants);

    EXPECT_EQ(ghosted_data.num_cells(), 210)
      << "DataArrayGhostedBlock<3, device_t>::num_cells() failed";
    EXPECT_EQ(ghosted_data.num_cells_inner(), 192)
      << "DataArrayGhostedBlock<3, device_t>::num_cells() failed";

    auto new_shape = block_size_t<3>{ 4, 8, 5 };
    auto new_shift = shift_t<3>{ 0, -1, 1 };
    auto accepted = ghosted_data.reshape(new_shape, new_shift);
    EXPECT_EQ(accepted, true);

    EXPECT_EQ(ghosted_data.num_cells(), 160)
      << "DataArrayGhostedBlock<3, device_t>::num_cells() failed";
    EXPECT_EQ(ghosted_data.num_cells_inner(), 192)
      << "DataArrayGhostedBlock<3, device_t>::num_cells() failed";


    for (int k = new_shift[IZ]; k < new_shift[IZ] + static_cast<int>(new_shape[IZ]); ++k)
      for (int j = new_shift[IY]; j < new_shift[IY] + static_cast<int>(new_shape[IY]); ++j)
        for (int i = new_shift[IX]; i < new_shift[IX] + static_cast<int>(new_shape[IX]); ++i)
        {
          ghosted_data(i, j, k, 0, 3) = i * i + j - k * k * k;
        }

    EXPECT_NEAR(ghosted_data(0, -1, 1, 0, 3), -2.0, 1e-14);
    EXPECT_NEAR(ghosted_data(2, 2, 2, 0, 3), -2.0, 1e-14);

    auto ijk = coord_t<3>{ 2, 0, 3 };
    EXPECT_NEAR(ghosted_data(ijk, 0, 3), -23.0, 1e-14);

    EXPECT_NEAR(ghosted_data(0, 0, 0, 0, 3, DirectAccess{}), -2.0, 1e-14);
    EXPECT_NEAR(ghosted_data(2, 3, 4, 0, 3, DirectAccess{}), -119.0, 1e-14);
    EXPECT_NEAR(ghosted_data(ijk, 0, 3, DirectAccess{}), -61.0, 1e-14);
  }

  {
    using DirectAccess = DataArrayGhostedBlock<3, real_t, device_t>::DirectAccess;

    auto       bSize = block_size_t<3>{ 4, 5, 4 };
    auto       gSize = block_size_t<3>{ 4, 9, 4 };
    shift_t<3> shift{ 0, -2, 0 };
    int        num_vars = 2;
    int        num_quadrants = 6;

    auto ghosted_data = DataArrayGhostedBlock<3, real_t, device_t>(
      bSize, gSize, shift, "data", num_vars, num_quadrants);

    EXPECT_EQ(ghosted_data.num_cells(), 144)
      << "DataArrayGhostedBlock<3, device_t>::num_cells() failed";
    EXPECT_EQ(ghosted_data.num_cells_inner(), 80)
      << "DataArrayGhostedBlock<3, device_t>::num_cells() failed";

    auto new_inner_shape = block_size_t<3>{ 4, 4, 5 };
    auto new_shape = block_size_t<3>{ 4, 4, 9 };
    auto new_shift = shift_t<3>{ 0, 0, -2 };
    auto accepted = ghosted_data.reshape(new_inner_shape, new_shape, new_shift);
    EXPECT_EQ(accepted, true);

    EXPECT_EQ(ghosted_data.num_cells(), 144)
      << "DataArrayGhostedBlock<3, device_t>::num_cells() failed";
    EXPECT_EQ(ghosted_data.num_cells_inner(), 80)
      << "DataArrayGhostedBlock<3, device_t>::num_cells() failed";

    EXPECT_EQ(ghosted_data.shape()[IX], 4);
    EXPECT_EQ(ghosted_data.shape()[IY], 4);
    EXPECT_EQ(ghosted_data.shape()[IZ], 9);
  }
} // TEST: kalypsso_shared_test/kalypsso_data_container_DataArrayGhostedBlock3d

#else
// ====================================================================
// ====================================================================
TEST(kalypsso_shared_test, kalypsso_data_container_DataArrayGhostedBlock)
{
  // using device_t = DefaultDevice;
  using device_t = HostDevice;
  auto bSize2 = block_size_t<2>{ 12, 7 };
  auto gw2 = block_size_t<2>{ 2, 3 };
  auto gdata = DataArrayGhostedBlock<2, real_t, device_t>(bSize2, gw2, "data", 4, 10);

  EXPECT_EQ(gdata.num_vars(), 4) << "DataArrayGhostedBlock<2, device_t>::num_vars() failed";
  EXPECT_EQ(gdata.total_block_size()[0], 16)
    << "DataArrayGhostedBlock<2, device_t> compute total block size failed";
  EXPECT_EQ(gdata.total_block_size()[1], 13)
    << "DataArrayGhostedBlock<2, device_t> compute total block size failed";

  EXPECT_EQ(gdata.block_size()[0], 12) << "DataArrayGhostedBlock<2, device_t> wrong block size";
  EXPECT_EQ(gdata.block_size()[1], 7) << "DataArrayGhostedBlock<2, device_t> wrong block size";

  auto block_data = gdata.data();

  EXPECT_EQ(block_data.num_cells(), 208) << "Wrong view size";

} // TEST: kalypsso_shared_test/kalypsso_data_container_DataArrayGhostedBlock
#endif

} // namespace kalypsso
