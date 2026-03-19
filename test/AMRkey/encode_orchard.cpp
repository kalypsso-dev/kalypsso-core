// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file decode_orchard.cpp
 *
 */
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/cmdline_utils.h>

#include <cstdlib>
#include <vector>
#include <iostream>
#include <array>
#include <algorithm>
#include <boost/program_options.hpp>

namespace kalypsso
{

using tree_coord_v_t = std::vector<uint16_t>;
using tree_coord_t = std::array<uint16_t, 3>;
using oct_coord_v_t = std::vector<uint32_t>;
using oct_coord_t = std::array<uint32_t, 3>;

// ================================================================================
// ================================================================================
template <int dim>
void
encode_orchard_key(tree_coord_t              tree_coord,
                   oct_coord_t               oct_coord,
                   uint8_t                   level,
                   Kokkos ::Array<bool, dim> outside_status)
{

  std::cout << "Encode a " << dim << "D orchard key:\n";

  Kokkos::Array<uint16_t, dim> tree_coord_kokkos;
  tree_coord_kokkos[0] = tree_coord[0];
  tree_coord_kokkos[1] = tree_coord[1];
  if constexpr (dim == 3)
    tree_coord_kokkos[2] = tree_coord[2];

  Kokkos::Array<uint32_t, dim> oct_coord_kokkos;
  oct_coord_kokkos[0] = oct_coord[0];
  oct_coord_kokkos[1] = oct_coord[1];
  if constexpr (dim == 3)
    oct_coord_kokkos[2] = oct_coord[2];

  std::cout << "root length : " << orchard_key_t<dim>::ROOT_LENGTH << " : 0x" << std::hex
            << orchard_key_t<dim>::ROOT_LENGTH << std::dec << "\n";

  auto key = orchard_key_t<dim>::encode_orchard(tree_coord_kokkos, oct_coord_kokkos, level);

  orchard_key_t<dim>::set_is_touching_face_X(key, outside_status[IX]);
  orchard_key_t<dim>::set_is_touching_face_Y(key, outside_status[IY]);
  if constexpr (dim == 3)
    orchard_key_t<dim>::set_is_touching_face_Z(key, outside_status[IZ]);

  if constexpr (dim == 2)
  {
    // clang-format off
    std::cout << "tree coord           : "
              << tree_coord_kokkos[0] << " "
              << tree_coord_kokkos[1] << "\n";
    std::cout << "oct  coord (input)   : "
              << oct_coord_kokkos[0] << " "
              << oct_coord_kokkos[1] << "\n";
    std::cout << "oct coords (encoded) : "
              << orchard_key_t<dim>::template get_octant_coord<IX>(key) << " "
              << orchard_key_t<dim>::template get_octant_coord<IY>(key) << "\n";
    // clang-format on
  }
  else
  {
    // clang-format off
    std::cout << "tree coord        : "
              << tree_coord_kokkos[0] << " "
              << tree_coord_kokkos[1] << " "
              << tree_coord_kokkos[2] << "\n";
    std::cout << "oct coord (input) : "
              << oct_coord_kokkos[0] << " "
              << oct_coord_kokkos[1] << " "
              << oct_coord_kokkos[2] << "\n";
    std::cout << "oct coords (Logical space) : "
              << orchard_key_t<dim>::template get_octant_coord<IX>(key) << " "
              << orchard_key_t<dim>::template get_octant_coord<IY>(key) << " "
              << orchard_key_t<dim>::template get_octant_coord<IZ>(key) << "\n";
    // clang-format on
  }
  std::cout << "level      : " << (int)level << "\n";


  std::cout << "key        : " << key << "\n";
  std::cout << "key (hex)  : 0x" << std::hex << key << std::dec << "\n";

} // encode_orchard_key

} // namespace kalypsso

// ================================================================================
// ================================================================================
// ================================================================================
int
main(int argc, char * argv[])
{
  using tree_coord_v_t = kalypsso::tree_coord_v_t;
  using tree_coord_t = kalypsso::tree_coord_t;
  using oct_coord_v_t = kalypsso::oct_coord_v_t;
  using oct_coord_t = kalypsso::oct_coord_t;

  namespace po = boost::program_options;

  bool           dim3 = false;
  tree_coord_v_t tree_coord{ 0, 0, 0 };
  oct_coord_v_t  oct_coord{ 0, 0, 0 };
  uint16_t       level_u16{ 0 };
  bool           is_outside_x = false;
  bool           is_outside_y = false;
  bool           is_outside_z = false;

  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()
    ("help", "produce help message")
    ("3d",  po::bool_switch(&dim3)->default_value(false), "decode a 3d key")
    ("tree_coord", po::value<tree_coord_v_t>(&tree_coord)->multitoken(), "set tree coords")
    ("oct_coord", po::value<oct_coord_v_t>(&oct_coord)->multitoken(), "set octant coords")
    ("level", po::value<uint16_t>(&level_u16), "set leaf quadrant level")
    ("is_outside_x", po::bool_switch(&is_outside_x)->default_value(false), "leaf quadrant is touching a X face")
    ("is_outside_y", po::bool_switch(&is_outside_y)->default_value(false), "leaf quadrant is touching a Y face")
    ("is_outside_z", po::bool_switch(&is_outside_z)->default_value(false), "leaf quadrant is touching a Z face")
    ;
  // clang-format-on

  po::variables_map vm;

  try{
    po::store(
      po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);
  } catch(po::error & e) {
    std::cout << "ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    std::cout << "Example: ./encode_orchard --level 5 --oct_coord $((2**4)) $((2**4)) \n";
    return EXIT_SUCCESS;
  }

  tree_coord_t tree_coord_array;
  std::copy_n(tree_coord.begin(), 3, tree_coord_array.begin());
  oct_coord_t oct_coord_array;
  std::copy_n(oct_coord.begin(), 3, oct_coord_array.begin());

  uint8_t level = static_cast<uint8_t>(level_u16);

  if (dim3) {
    Kokkos::Array<bool, 3> is_outside{is_outside_x, is_outside_y, is_outside_z};
    kalypsso::encode_orchard_key<3>(tree_coord_array, oct_coord_array, level, is_outside);
  } else {
    Kokkos::Array<bool, 2> is_outside{is_outside_x, is_outside_y};
    kalypsso::encode_orchard_key<2>(tree_coord_array, oct_coord_array, level, is_outside);
  }

  return EXIT_SUCCESS;

} // main
