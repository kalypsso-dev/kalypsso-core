// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_orchard.cpp
 *
 * binary print borrowed from stackoverflow :
 * https://stackoverflow.com/questions/111928/is-there-a-printf-converter-to-print-in-binary-format
 */
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/orchard_key.h>

#include <cstdlib>

// clang-format off
#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0')
// clang-format on

namespace kalypsso
{

// ================================================================================
// ================================================================================
template <int dim>
void
print_morton_tree(Kokkos::Array<uint16_t, dim> treeCoord)
{
  uint16_t morton_tree = kalypsso::orchard_key_t<dim>::encode_morton_tree(treeCoord);

  if constexpr (dim == 2)
  {
    printf("treeCoord=(%8d,%8d)=(0x%04x,0x%04x) | morton key = 0x%016x = 0b" BYTE_TO_BINARY_PATTERN
           "|" BYTE_TO_BINARY_PATTERN "\n",
           treeCoord[IX],
           treeCoord[IY],
           treeCoord[IX],
           treeCoord[IY],
           morton_tree,
           BYTE_TO_BINARY(morton_tree >> 8),
           BYTE_TO_BINARY(morton_tree));
  }

  if constexpr (dim == 3)
  {
    printf("treeCoord=(%5d,%5d,%5d)=(0x%04x,0x%04x,0x%04x) | morton key = 0x%016x = "
           "0b" BYTE_TO_BINARY_PATTERN "|" BYTE_TO_BINARY_PATTERN "\n",
           treeCoord[IX],
           treeCoord[IY],
           treeCoord[IZ],
           treeCoord[IX],
           treeCoord[IY],
           treeCoord[IZ],
           morton_tree,
           BYTE_TO_BINARY(morton_tree >> 8),
           BYTE_TO_BINARY(morton_tree));
  }

  // build full orchard key
  uint64_t full_orchard_key = morton_tree;

  // for cross-check, decode morton
  if constexpr (dim == 2)
  {
    uint16_t iTreeX = kalypsso::orchard_key_t<dim>::template get_tree_coord<IX>(full_orchard_key);
    uint16_t iTreeY = kalypsso::orchard_key_t<dim>::template get_tree_coord<IY>(full_orchard_key);
    printf("decode orchard key : tree=(%d,%d)\n", iTreeX, iTreeY);
  }
  if constexpr (dim == 3)
  {
    uint16_t iTreeX = kalypsso::orchard_key_t<dim>::template get_tree_coord<IX>(full_orchard_key);
    uint16_t iTreeY = kalypsso::orchard_key_t<dim>::template get_tree_coord<IY>(full_orchard_key);
    uint16_t iTreeZ = kalypsso::orchard_key_t<dim>::template get_tree_coord<IZ>(full_orchard_key);
    printf("decode orchard key : tree=(%d,%d,%d)\n", iTreeX, iTreeY, iTreeZ);
  }

} // print_morton_tree

// ================================================================================
// ================================================================================
template <int dim>
void
print_morton_octant(Kokkos::Array<uint32_t, dim> octCoord)
{
  uint64_t key = 0;

  kalypsso::orchard_key_t<dim>::encode_morton_octant(key, octCoord);
  uint64_t morton_oct = kalypsso::orchard_key_t<dim>::morton_octant(key);

  if constexpr (dim == 2)
  {
    printf("octCoord=(%4d,%4d)=(0x%04x,0x%04x) | morton key = 0x%016lx = 0b" BYTE_TO_BINARY_PATTERN
           "|" BYTE_TO_BINARY_PATTERN "||" BYTE_TO_BINARY_PATTERN "|" BYTE_TO_BINARY_PATTERN
           "||" BYTE_TO_BINARY_PATTERN "|" BYTE_TO_BINARY_PATTERN "\n",
           octCoord[IX],
           octCoord[IY],
           octCoord[IX],
           octCoord[IY],
           morton_oct,
           BYTE_TO_BINARY(morton_oct >> 40),
           BYTE_TO_BINARY(morton_oct >> 32),
           BYTE_TO_BINARY(morton_oct >> 24),
           BYTE_TO_BINARY(morton_oct >> 16),
           BYTE_TO_BINARY(morton_oct >> 8),
           BYTE_TO_BINARY(morton_oct));
  }
  else if constexpr (dim == 3)
  {
    printf("octCoord=(%5d,%5d,%5d)=(0x%04x,0x%04x,0x%04x) | morton key = 0x%016lx = "
           "0b" BYTE_TO_BINARY_PATTERN "|" BYTE_TO_BINARY_PATTERN "||" BYTE_TO_BINARY_PATTERN
           "|" BYTE_TO_BINARY_PATTERN "||" BYTE_TO_BINARY_PATTERN "|" BYTE_TO_BINARY_PATTERN "\n",
           octCoord[IX],
           octCoord[IY],
           octCoord[IZ],
           octCoord[IX],
           octCoord[IY],
           octCoord[IZ],
           morton_oct,
           BYTE_TO_BINARY(morton_oct >> 40),
           BYTE_TO_BINARY(morton_oct >> 32),
           BYTE_TO_BINARY(morton_oct >> 24),
           BYTE_TO_BINARY(morton_oct >> 16),
           BYTE_TO_BINARY(morton_oct >> 8),
           BYTE_TO_BINARY(morton_oct));
  }

  // build full orchard key
  uint64_t full_orchard_key = morton_oct << kalypsso::orchard_key_t<dim>::MORTON_OCTANT_OFFSET;

  // for cross-check, decode morton octant
  if constexpr (dim == 2)
  {
    uint32_t iOctX = kalypsso::orchard_key_t<dim>::template get_octant_coord<IX>(full_orchard_key);
    uint32_t iOctY = kalypsso::orchard_key_t<dim>::template get_octant_coord<IY>(full_orchard_key);

    printf("decode orchard key : octant=(%d,%d)\n", iOctX, iOctY);
  }
  else if constexpr (dim == 3)
  {
    uint32_t iOctX = kalypsso::orchard_key_t<dim>::template get_octant_coord<IX>(full_orchard_key);
    uint32_t iOctY = kalypsso::orchard_key_t<dim>::template get_octant_coord<IY>(full_orchard_key);
    uint32_t iOctZ = kalypsso::orchard_key_t<dim>::template get_octant_coord<IZ>(full_orchard_key);

    printf("decode orchard key : octant=(%d,%d,%d)\n", iOctX, iOctY, iOctZ);
  }
} // print_morton_octant

// ================================================================================
// ================================================================================
template <int dim>
void
run_test_tree()
{

  if constexpr (dim == 2)
  {
    printf("=================================================\n");
    printf("Cross-checking morton encoding for tree - 2d case\n");
    printf("=================================================\n");
    // input is the tree coordinate in the brick connectivity
    // output is the morton tree key, here printed in binary for eye check

    // use brick connectivity - max number of trees per dim is 2^7 = 128
    // uint16_t nTreeX = 128;
    // uint16_t nTreeY = 128;

    Kokkos::Array<uint16_t, dim> iTree{ 0, 0 };

    iTree[IX] = 0x0000;
    iTree[IY] = 0x007F;
    print_morton_tree<dim>(iTree);

    iTree[IX] = 0x007F;
    iTree[IY] = 0x0000;
    print_morton_tree<dim>(iTree);

    iTree[IX] = 0x0005;
    iTree[IY] = 0x0002;
    print_morton_tree<dim>(iTree);

    iTree[IX] = 0x0002;
    iTree[IY] = 0x0005;
    print_morton_tree<dim>(iTree);

    iTree[IX] = 32;
    iTree[IY] = 17;
    print_morton_tree<dim>(iTree);

    iTree[IX] = 77;
    iTree[IY] = 111;
    print_morton_tree<dim>(iTree);
  }

  if constexpr (dim == 3)
  {
    printf("=================================================\n");
    printf("Cross-checking morton encoding for tree - 3d case\n");
    printf("=================================================\n");
    // input is the tree coordinate in the brick connectivity
    // output is the morton tree key, here printed in binary for eye check

    // use brick connectivity - max number of trees per dim is 2^5 = 32

    Kokkos::Array<uint16_t, dim> iTree{ 0, 0, 0 };

    iTree[IX] = 0x001F;
    iTree[IY] = 0x0000;
    iTree[IZ] = 0x0000;
    print_morton_tree<dim>(iTree);

    iTree[IX] = 0x0000;
    iTree[IY] = 0x001F;
    iTree[IZ] = 0x0000;
    print_morton_tree<dim>(iTree);

    iTree[IX] = 0x0000;
    iTree[IY] = 0x0000;
    iTree[IZ] = 0x001F;
    print_morton_tree<dim>(iTree);

    iTree[IX] = 18;
    iTree[IY] = 4;
    iTree[IZ] = 1;
    print_morton_tree<dim>(iTree);

    iTree[IX] = 8;
    iTree[IY] = 17;
    iTree[IZ] = 29;
    print_morton_tree<dim>(iTree);

    // this one should fail
    iTree[IX] = 32;
    iTree[IY] = 1;
    iTree[IZ] = 2;
    print_morton_tree<dim>(iTree);
  }
} // run_test_tree

// ================================================================================
// ================================================================================
template <int dim>
void
run_test_octant()
{

  if constexpr (dim == 2)
  {
    printf("====================================================\n");
    printf("Cross-checking morton encoding for octant - 2d case \n");
    printf("====================================================\n");
    // input is the octant coordinates in the brick connectivity
    // output is the morton octant key, here printed in binary for eye check

    // octant coordinates are unsigned integer on 22 bits

    Kokkos::Array<uint32_t, dim> iOct{ 0, 0 };

    iOct[IX] = 0x0000;
    iOct[IY] = 0x007F;
    print_morton_octant<dim>(iOct);

    iOct[IX] = 0x007F;
    iOct[IY] = 0x0000;
    print_morton_octant<dim>(iOct);

    iOct[IX] = 0x0005;
    iOct[IY] = 0x0002;
    print_morton_octant<dim>(iOct);

    iOct[IX] = 0x0002;
    iOct[IY] = 0x0005;
    print_morton_octant<dim>(iOct);

    iOct[IX] = 320;
    iOct[IY] = 1777;
    print_morton_octant<dim>(iOct);

    iOct[IX] = 77;
    iOct[IY] = 111;
    print_morton_octant<dim>(iOct);

    iOct[IX] = 2048;
    iOct[IY] = 1024;
    print_morton_octant<dim>(iOct);

    iOct[IX] = 1024;
    iOct[IY] = 2048;
    print_morton_octant<dim>(iOct);

    iOct[IX] = 1 << 16;
    iOct[IY] = 1 << 18;
    print_morton_octant<dim>(iOct);

    // this one should fail, since 1^22 is 1 too large
    iOct[IX] = 1 << 22;
    iOct[IY] = 1 << 18;
    print_morton_octant<dim>(iOct);

    // this one should fail, since 6544213 is >= 4194304 (2^22)
    iOct[IX] = 1000000;
    iOct[IY] = 6544213;
    print_morton_octant<dim>(iOct);
  }
  else if constexpr (dim == 3)
  {
    printf("====================================================\n");
    printf("Cross-checking morton encoding for octant - 3d case \n");
    printf("====================================================\n");
    // input is the octant coordinates in the brick connectivity
    // output is the morton octant key, here printed in binary for eye check

    // octant coordinates are unsigned integer on 15 bits

    Kokkos::Array<uint32_t, dim> iOct{ 0, 0, 0 };

    iOct[IX] = 0x0000;
    iOct[IY] = 0x001F;
    iOct[IZ] = 0x0000;
    print_morton_octant<dim>(iOct);

    iOct[IX] = 0x0100;
    iOct[IY] = 0x001F;
    iOct[IZ] = 0x4000;
    print_morton_octant<dim>(iOct);

    iOct[IX] = 0x1000;
    iOct[IY] = 0x001F;
    iOct[IZ] = 0x7FFF;
    print_morton_octant<dim>(iOct);

    // this one should fail
    iOct[IX] = 0x1000;
    iOct[IY] = 0x001F;
    iOct[IZ] = 0x8000; // should fail here
    print_morton_octant<dim>(iOct);
  }

} // run_test_octant

} // namespace kalypsso

// ================================================================================
// ================================================================================
// ================================================================================
int
main()
{

  kalypsso::run_test_tree<2>();
  kalypsso::run_test_octant<2>();

  kalypsso::run_test_tree<3>();
  kalypsso::run_test_octant<3>();

  return EXIT_SUCCESS;
}
