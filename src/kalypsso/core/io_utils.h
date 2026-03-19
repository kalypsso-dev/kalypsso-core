// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file io_utils.h
 */
#ifndef KALYPSSO_CORE_IO_UTILS_H_
#define KALYPSSO_CORE_IO_UTILS_H_

#include <kalypsso/core/HydroParams.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/FieldMap.h>

#include <fstream>

namespace kalypsso
{

/**
 * Use config_map information to retrieve a list of scalar field to write,
 * compute the their id the access them in DataArray. This routine returns
 * a map with this information.
 *
 * \param[in,out] map          this is the map to fill
 * \param[in]     avail_names  list of available variable names
 * \param[in]     config_map    parameters settings
 *
 * \return the map size (i.e. the number of valid variable names)
 */
size_t
build_var_to_write_map(names2id_t &       map,
                       const names2id_t & avail_names,
                       const ConfigMap &  config_map);

/**
 * Check if file exists.
 *
 * \param[in] filename file name to check.
 */
inline bool
file_exists(const std::string & filename)
{
  std::ifstream f(filename.c_str());
  return f.good();
}

} // namespace kalypsso

#endif // KALYPSSO_CORE_IO_UTILS_H_
