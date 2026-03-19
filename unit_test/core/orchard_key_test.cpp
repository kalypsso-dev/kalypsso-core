// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/orchard_key.h>

#include <cstdint>

#include <gtest/gtest.h>

namespace kalypsso
{

//=================================================================
//
// basic test
//
//=================================================================

TEST(kalypsso_shared_orchard_key_test, base_2d)
{

  {
    auto key = kalypsso::orchard_key_t<2>::encode_orchard(0, { 1 << 13, 1 << 13 }, 5);

    auto oct_coords = kalypsso::orchard_key_t<2>::get_octant_coords(key);

    EXPECT_EQ(oct_coords[IX], 8192) << "kalypsso::orchard_key_t<2>::get_octant_coords";
    EXPECT_EQ(oct_coords[IY], 8192) << "kalypsso::orchard_key_t<2>::get_octant_coords";
  }

  {
    auto key = kalypsso::orchard_key_t<2>::encode_orchard(0, { 1 << 21, 1 << 10 }, 5);

    auto oct_coords = kalypsso::orchard_key_t<2>::get_octant_coords(key);

    EXPECT_EQ(oct_coords[IX], 2097152) << "kalypsso::orchard_key_t<2>::get_octant_coords";
    EXPECT_EQ(oct_coords[IY], 1024) << "kalypsso::orchard_key_t<2>::get_octant_coords";
  }

  // this one should "fail" (overflow, because max level is 22), so the extracted coord should be 0
  {
    auto key = kalypsso::orchard_key_t<2>::encode_orchard(0, { 1 << 22, 1 << 10 }, 5);

    auto oct_coords = kalypsso::orchard_key_t<2>::get_octant_coords(key);

    EXPECT_EQ(oct_coords[IX], 0) << "kalypsso::orchard_key_t<2>::get_octant_coords";
    EXPECT_EQ(oct_coords[IY], 1024) << "kalypsso::orchard_key_t<2>::get_octant_coords";
  }

  // this one should "fail" (overflow, because max level is 22), so the extracted coord should be 0
  {
    auto key = kalypsso::orchard_key_t<2>::encode_orchard(0, { 1 << 23, 1 << 10 }, 5);

    auto oct_coords = kalypsso::orchard_key_t<2>::get_octant_coords(key);

    EXPECT_EQ(oct_coords[IX], 0) << "kalypsso::orchard_key_t<2>::get_octant_coords";
    EXPECT_EQ(oct_coords[IY], 1024) << "kalypsso::orchard_key_t<2>::get_octant_coords";
  }
}

TEST(kalypsso_shared_orchard_key_test, base_3d)
{

  {
    auto key = kalypsso::orchard_key_t<3>::encode_orchard(0, { 1 << 11, 1 << 11, 1 << 11 }, 5);

    auto oct_coords = kalypsso::orchard_key_t<3>::get_octant_coords(key);

    EXPECT_EQ(oct_coords[IX], 2048) << "kalypsso::orchard_key_t<3>::get_octant_coords";
    EXPECT_EQ(oct_coords[IY], 2048) << "kalypsso::orchard_key_t<3>::get_octant_coords";
    EXPECT_EQ(oct_coords[IZ], 2048) << "kalypsso::orchard_key_t<3>::get_octant_coords";
  }

  {
    auto key = kalypsso::orchard_key_t<3>::encode_orchard(0, { 1 << 12, 1 << 11, 1 << 11 }, 5);

    auto oct_coords = kalypsso::orchard_key_t<3>::get_octant_coords(key);

    EXPECT_EQ(oct_coords[IX], 4096) << "kalypsso::orchard_key_t<3>::get_octant_coords";
    EXPECT_EQ(oct_coords[IY], 2048) << "kalypsso::orchard_key_t<3>::get_octant_coords";
    EXPECT_EQ(oct_coords[IZ], 2048) << "kalypsso::orchard_key_t<3>::get_octant_coords";
  }

  // this one should "fail" (overflow, because max level is 14), i.e. 2**13 is max
  // so the extracted coord should be 0
  {
    auto key = kalypsso::orchard_key_t<3>::encode_orchard(0, { 1 << 14, 1 << 10, 1 << 5 }, 5);

    auto oct_coords = kalypsso::orchard_key_t<3>::get_octant_coords(key);

    EXPECT_EQ(oct_coords[IX], 0) << "kalypsso::orchard_key_t<3>::get_octant_coords";
    EXPECT_EQ(oct_coords[IY], 1024) << "kalypsso::orchard_key_t<3>::get_octant_coords";
    EXPECT_EQ(oct_coords[IZ], 32) << "kalypsso::orchard_key_t<3>::get_octant_coords";
  }

  // this one should "fail" (overflow, because max level is 14), i.e. 2**13 is max
  // so the extracted coord should be 0
  {
    auto key = kalypsso::orchard_key_t<3>::encode_orchard(0, { 1 << 10, 1 << 14, 1 << 5 }, 3);

    auto oct_coords = kalypsso::orchard_key_t<3>::get_octant_coords(key);

    EXPECT_EQ(oct_coords[IX], 1024) << "kalypsso::orchard_key_t<3>::get_octant_coords";
    EXPECT_EQ(oct_coords[IY], 0) << "kalypsso::orchard_key_t<3>::get_octant_coords";
    EXPECT_EQ(oct_coords[IZ], 32) << "kalypsso::orchard_key_t<3>::get_octant_coords";
  }

  // this on should "fail" (overflow, because max level is 14), i.e. 2**13 is max
  // so the extracted coord should be 0
  {
    auto key = kalypsso::orchard_key_t<3>::encode_orchard(0, { 1 << 2, 1 << 3, 1 << 14 }, 5);

    auto oct_coords = kalypsso::orchard_key_t<3>::get_octant_coords(key);

    EXPECT_EQ(oct_coords[IX], 4) << "kalypsso::orchard_key_t<3>::get_octant_coords";
    EXPECT_EQ(oct_coords[IY], 8) << "kalypsso::orchard_key_t<3>::get_octant_coords";
    EXPECT_EQ(oct_coords[IZ], 0) << "kalypsso::orchard_key_t<3>::get_octant_coords";
  }
}

//=================================================================
//
// Father and child
//
//=================================================================

TEST(kalypsso_shared_orchard_key_test, father_child_2d)
{

  auto key = kalypsso::orchard_key_t<2>::encode_orchard(0, { 0, 0 }, 5);

  auto length = kalypsso::orchard_key_t<2>::octantLength(key);

  auto father = kalypsso::orchard_key_t<2>::father(key);


  auto key2 = kalypsso::orchard_key_t<2>::encode_orchard(0, { length, 0 }, 5);

  EXPECT_EQ(kalypsso::orchard_key_t<2>::child_id(key2), 1)
    << "kalypsso::orchard_key_t<2>::child_id";

  key2 = kalypsso::orchard_key_t<2>::encode_orchard(0, { 0, length }, 5);

  EXPECT_EQ(kalypsso::orchard_key_t<2>::child_id(key2), 2)
    << "kalypsso::orchard_key_t<2>::child_id";

  key2 = kalypsso::orchard_key_t<2>::encode_orchard(0, { length, length }, 5);

  EXPECT_EQ(kalypsso::orchard_key_t<2>::child_id(key2), 3)
    << "kalypsso::orchard_key_t<2>::child_id";

  key2 = kalypsso::orchard_key_t<2>::encode_orchard(0, { 2 * length, length }, 5);

  EXPECT_EQ(kalypsso::orchard_key_t<2>::child_id(key2), 2)
    << "kalypsso::orchard_key_t<2>::child_id";

  key2 = kalypsso::orchard_key_t<2>::encode_orchard(0, { 0, length }, 5);

  auto father2 = kalypsso::orchard_key_t<2>::father(key2);

  EXPECT_EQ(father, father2) << "kalypsso::orchard_key_t<2>::father failed";

  key2 = kalypsso::orchard_key_t<2>::encode_orchard(0, { 0, 2 * length }, 5);
  father2 = kalypsso::orchard_key_t<2>::father(key2);

  EXPECT_TRUE(father != father2) << "kalypsso::orchard_key_t<2>::father failed";
}

TEST(kalypsso_shared_orchard_key_test, father_child_3d)
{

  auto key = kalypsso::orchard_key_t<3>::encode_orchard(0, { 0, 0, 0 }, 6);

  auto length = kalypsso::orchard_key_t<3>::octantLength(key);

  // auto father = kalypsso::orchard_key_t<3>::father(key);


  auto key2 = kalypsso::orchard_key_t<3>::encode_orchard(0, { 0, length, 0 }, 5);

  EXPECT_EQ(kalypsso::orchard_key_t<3>::child_id(key2), 2)
    << "kalypsso::orchard_key_t<3>::child_id";

  key2 = kalypsso::orchard_key_t<3>::encode_orchard(0, { 0, length, length }, 6);

  EXPECT_EQ(kalypsso::orchard_key_t<3>::child_id(key2), 6)
    << "kalypsso::orchard_key_t<3>::child_id";
}

//=================================================================
//
// neighbors
//
//=================================================================
TEST(kalypsso_shared_orchard_key_test, neighbors)
{
  // 2D

  EXPECT_EQ(kalypsso::orchard_key_t<2>::get_face_neighbor_smallest_child_id(Face::XMIN), 1);
  EXPECT_EQ(kalypsso::orchard_key_t<2>::get_face_neighbor_smallest_child_id(Face::XMAX), 0);
  EXPECT_EQ(kalypsso::orchard_key_t<2>::get_face_neighbor_smallest_child_id(Face::YMIN), 2);
  EXPECT_EQ(kalypsso::orchard_key_t<2>::get_face_neighbor_smallest_child_id(Face::YMAX), 0);

  EXPECT_EQ(kalypsso::orchard_key_t<2>::get_edge_neighbor_smallest_child_id(Face::XMIN, Face::YMIN),
            3);
  EXPECT_EQ(kalypsso::orchard_key_t<2>::get_edge_neighbor_smallest_child_id(Face::XMAX, Face::YMIN),
            2);
  EXPECT_EQ(kalypsso::orchard_key_t<2>::get_edge_neighbor_smallest_child_id(Face::XMIN, Face::YMAX),
            1);
  EXPECT_EQ(kalypsso::orchard_key_t<2>::get_edge_neighbor_smallest_child_id(Face::XMAX, Face::YMAX),
            0);

  // 3D

  // faces
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_face_neighbor_smallest_child_id(Face::XMIN), 1);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_face_neighbor_smallest_child_id(Face::XMAX), 0);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_face_neighbor_smallest_child_id(Face::YMIN), 2);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_face_neighbor_smallest_child_id(Face::YMAX), 0);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_face_neighbor_smallest_child_id(Face::ZMIN), 4);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_face_neighbor_smallest_child_id(Face::ZMAX), 0);

  // edges
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::XMIN, Face::YMIN),
            3);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::XMAX, Face::YMIN),
            2);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::XMIN, Face::YMAX),
            1);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::XMAX, Face::YMAX),
            0);

  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::YMIN, Face::ZMIN),
            6);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::YMAX, Face::ZMIN),
            4);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::YMIN, Face::ZMAX),
            2);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::YMAX, Face::ZMAX),
            0);

  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::XMIN, Face::ZMIN),
            5);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::XMAX, Face::ZMIN),
            4);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::XMIN, Face::ZMAX),
            1);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_edge_neighbor_smallest_child_id(Face::XMAX, Face::ZMAX),
            0);

  // corners
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_corner_neighbor_smallest_child_id(
              Face::XMIN, Face::YMIN, Face::ZMIN),
            7);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_corner_neighbor_smallest_child_id(
              Face::XMAX, Face::YMIN, Face::ZMIN),
            6);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_corner_neighbor_smallest_child_id(
              Face::XMIN, Face::YMAX, Face::ZMIN),
            5);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_corner_neighbor_smallest_child_id(
              Face::XMAX, Face::YMAX, Face::ZMIN),
            4);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_corner_neighbor_smallest_child_id(
              Face::XMIN, Face::YMIN, Face::ZMAX),
            3);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_corner_neighbor_smallest_child_id(
              Face::XMAX, Face::YMIN, Face::ZMAX),
            2);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_corner_neighbor_smallest_child_id(
              Face::XMIN, Face::YMAX, Face::ZMAX),
            1);
  EXPECT_EQ(kalypsso::orchard_key_t<3>::get_corner_neighbor_smallest_child_id(
              Face::XMAX, Face::YMAX, Face::ZMAX),
            0);
}

//=================================================================
//
// encode / decode
//
//=================================================================

TEST(kalypsso_shared_orchard_key_test, encode_decode_2d)
{

  auto key = kalypsso::orchard_key_t<2>::encode_orchard({ 5, 6 }, { 7777, 3333 }, 8);
  kalypsso::orchard_key_t<2>::set_is_touching_face_X(key, true);

  EXPECT_EQ(key, 256705187183600417) << "kalypsso::orchard_key_t<2>::encode_orchard error";

  EXPECT_EQ(orchard_key_t<2>::template get_tree_coord<IX>(key), 5)
    << "kalypsso::orchard_key_t<2>::decode_orchard error";
  EXPECT_EQ(orchard_key_t<2>::template get_tree_coord<IY>(key), 6)
    << "kalypsso::orchard_key_t<2>::decode_orchard error";

  EXPECT_EQ(orchard_key_t<2>::template get_octant_coord<IX>(key), 7777)
    << "kalypsso::orchard_key_t<2>::decode_orchard error";
  EXPECT_EQ(orchard_key_t<2>::template get_octant_coord<IY>(key), 3333)
    << "kalypsso::orchard_key_t<2>::decode_orchard error";

  EXPECT_EQ(orchard_key_t<2>::level(key), 8)
    << "kalypsso::orchard_key_t<2>::decode_orchard error decoding level";
}

TEST(kalypsso_shared_orchard_key_test, encode_decode_3d)
{

  auto key = kalypsso::orchard_key_t<3>::encode_orchard({ 2, 3, 4 }, { 2048, 16383, 4096 }, 5);
  kalypsso::orchard_key_t<3>::set_is_touching_face_Y(key, true);

  EXPECT_EQ(key, 158949013592361258) << "kalypsso::orchard_key_t<3>::encode_orchard error";

  EXPECT_EQ(orchard_key_t<3>::template get_tree_coord<IX>(key), 2)
    << "kalypsso::orchard_key_t<3>::decode_orchard error";
  EXPECT_EQ(orchard_key_t<3>::template get_tree_coord<IY>(key), 3)
    << "kalypsso::orchard_key_t<3>::decode_orchard error";
  EXPECT_EQ(orchard_key_t<3>::template get_tree_coord<IZ>(key), 4)
    << "kalypsso::orchard_key_t<3>::decode_orchard error";

  EXPECT_EQ(orchard_key_t<3>::template get_octant_coord<IX>(key), 2048)
    << "kalypsso::orchard_key_t<3>::decode_orchard error";
  EXPECT_EQ(orchard_key_t<3>::template get_octant_coord<IY>(key), 16383)
    << "kalypsso::orchard_key_t<3>::decode_orchard error";
  EXPECT_EQ(orchard_key_t<3>::template get_octant_coord<IZ>(key), 4096)
    << "kalypsso::orchard_key_t<3>::decode_orchard error";

  EXPECT_EQ(orchard_key_t<3>::level(key), 5)
    << "kalypsso::orchard_key_t<3>::decode_orchard error decoding level";
}

} // namespace kalypsso
