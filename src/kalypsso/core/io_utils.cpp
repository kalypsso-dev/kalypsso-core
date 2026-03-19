// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file io_utils.cpp
 */
#include <kalypsso/core/io_utils.h>

namespace kalypsso
{

// =======================================================
// =======================================================
size_t
build_var_to_write_map(names2id_t &       map,
                       const names2id_t & avail_names,
                       const ConfigMap &  config_map)
{

  // Read parameter file and get the list of variable field names
  // we wish to write
  // variable names must be comma-separated (no quotes !)
  // e.g. write_variables=rho,vx,vy,unknown
  std::string write_variables = config_map.getString("output", "write_variables", "rho,rho_vx");

  // now tokenize
  std::istringstream iss(write_variables);
  std::string        token;
  while (std::getline(iss, token, ','))
  {

    // check if token is valid, i.e. present in avail_names
    auto got = avail_names.find(token);
    // std::cout << " " << token << " " << got->second << "\n";

    // if token is valid, we insert it into map
    if (got != avail_names.end())
    {
      map[token] = got->second;
    }
  }

  return map.size();

} // build_var_to_write_map

} // namespace kalypsso
