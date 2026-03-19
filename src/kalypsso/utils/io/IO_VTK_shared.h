// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KALYPSSO_UTILS_IO_IOVTKSHARED_H_
#define KALYPSSO_UTILS_IO_IOVTKSHARED_H_

#include <kalypsso/core/kalypsso_core_config.h>

#include <kalypsso/core/real_type.h>
#include <kalypsso/core/misc_utils.h>
#include <kalypsso/utils/config/ConfigMap.h>

namespace kalypsso
{
namespace io
{

/**
 * Write VTK unstructured grid header.
 */
void
write_vtu_header(std::ostream & outFile, ConfigMap & config_map);

/**
 * Write VTK unstructured grid metadata (date and time).
 */
void
write_vtk_metadata(std::ostream & outFile, int iStep, real_t time);

/**
 * Write closing VTK unstructured grid statement.
 */
void
close_vtu_grid(std::ostream & outFile);

/**
 * Write closing VTK file statement.
 */
void
write_vtu_footer(std::ostream & outFile);

#ifdef KALYPSSO_CORE_USE_MPI
/**
 * write pvtu header in a separate file.
 */
void
write_pvtu_header(std::string                        headerFilename,
                  std::string                        outputPrefix,
                  const int                          nProcs,
                  ConfigMap &                        config_map,
                  const std::map<int, std::string> & varNames,
                  const int                          iStep);
#endif // KALYPSSO_CORE_USE_MPI

} // namespace io

} // namespace kalypsso

#endif // KALYPSSO_UTILS_IO_IOVTKSHARED_H_
