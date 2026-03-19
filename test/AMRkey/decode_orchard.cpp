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
#include <boost/program_options.hpp>

namespace kalypsso
{

// ================================================================================
// ================================================================================
template <int dim>
void
decode_orchard_key(uint64_t key)
{

  std::cout << "Decode a " << dim << "D orchard key:\n";

  std::cout << "key        : " << key << "\n";

  std::cout << "level      : " << (uint)orchard_key_t<dim>::level(key) << "\n";

  std::cout << "tree id    : " << orchard_key_t<dim>::morton_tree(key) << "\n";

  // clang-format off
  if constexpr (dim == 2)
    std::cout << "tree coord : "
              << orchard_key_t<dim>::template get_tree_coord<IX>(key) << " "
              << orchard_key_t<dim>::template get_tree_coord<IY>(key) << "\n";
  else if constexpr (dim == 3)
    std::cout << "tree coord : "
              << orchard_key_t<dim>::template get_tree_coord<IX>(key) << " "
              << orchard_key_t<dim>::template get_tree_coord<IY>(key) << " "
              << orchard_key_t<dim>::template get_tree_coord<IZ>(key) << "\n";
  // clang-format on

  std::cout << "oct id     : " << orchard_key_t<dim>::morton_octant(key) << "\n";

  // real space lower left corner of leaf block identified by key
  auto vertex_coord = orchard_key_to_vertex_coord<dim>(key, false);

  // clang-format off
  if constexpr (dim == 2)
  {
    std::cout << "oct coords (Logical space) : "
              << orchard_key_t<dim>::template get_octant_coord<IX>(key) << " "
              << orchard_key_t<dim>::template get_octant_coord<IY>(key) << "\n"
      ;
    std::cout << "oct coords (Vertex  space) : "
              << vertex_coord[IX] << " "
              << vertex_coord[IY] << "\n";
  }
  else if constexpr (dim == 3)
  {
    std::cout << "oct coords (Logical space) : "
              << orchard_key_t<dim>::template get_octant_coord<IX>(key) << " "
              << orchard_key_t<dim>::template get_octant_coord<IY>(key) << " "
              << orchard_key_t<dim>::template get_octant_coord<IZ>(key) << "\n"
      ;
    std::cout << "oct coords (Vertex  space) : "
              << vertex_coord[IX] << " "
              << vertex_coord[IY] << " "
              << vertex_coord[IZ] << "\n";
  }
  // clang-format on

} // decode_orchard_key

} // namespace kalypsso

// ================================================================================
// ================================================================================
// ================================================================================
int
main(int argc, char * argv[])
{
  namespace po = boost::program_options;

  bool dim3 = false;

  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()
    ("help", "produce help message")
    ("3d", po::bool_switch(&dim3)->default_value(false), "decode a 3d key")
    ("key", po::value<uint64_t>()->default_value(0), "key to decode")
    ;
  // clang-format-on

  // Positional arguments don't need a parameter flag
  po::positional_options_description pos_desc;
  pos_desc.add("key", 1);

  po::variables_map vm;

  try{
    po::store(
      po::command_line_parser(argc, argv).options(desc).positional(pos_desc).run(), vm);
    po::notify(vm);
  } catch(po::error & e) {
    std::cout << "ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }

  auto key { vm["key"].as<uint64_t>() };

  if (dim3)
    kalypsso::decode_orchard_key<3>(key);
  else
    kalypsso::decode_orchard_key<2>(key);

  return EXIT_SUCCESS;

} // main
