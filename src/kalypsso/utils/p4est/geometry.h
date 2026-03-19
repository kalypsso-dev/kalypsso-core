// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * @file geometry.h
 * @date 17 octobre 2020
 * @author pkestene
 * @note
 *
 * This file is part of the Kalypsso software project.
 *
 */
#ifndef KALYPSSO_UTILS_P4EST_GEOMETRY_H_
#define KALYPSSO_UTILS_P4EST_GEOMETRY_H_

#include <p4est_connectivity.h>
#include <p4est_geometry.h>

#include <p8est_connectivity.h>
#include <p8est_geometry.h>

namespace kalypsso
{
class ConfigMap;
} // namespace kalypsso

/**
 * cylindrical geometry for computation associated to cylindrical connectivity.
 *
 * \param[in] R0 radius of the inner border
 * \param[in] R1 radius of the outer border
 * \param[in] nbTrees_theta number of tree along theta (orthoradial)
 * \param[in] nbTrees_z     number of tree along z direction
 *
 */
p4est_geometry_t *
p4est_geometry_new_cylindrical_compute(p4est_connectivity_t * conn,
                                       double                 R0,
                                       double                 R1,
                                       int                    nbTrees_theta,
                                       int                    nbTrees_z);

/**
 * cylindrical geometry used only for IO associated to cylindrical connectivity.
 *
 * \param[in] R0 radius of the inner border
 * \param[in] R1 radius of the outer border
 * \param[in] nbTrees_theta number of tree along theta (orthoradial)
 * \param[in] nbTrees_z     number of tree along z direction
 *
 */
p4est_geometry_t *
p4est_geometry_new_cylindrical_io(p4est_connectivity_t * conn,
                                  double                 R0,
                                  double                 R1,
                                  int                    nbTrees_theta,
                                  int                    nbTrees_z);

/**
 * \brief Create a geometry by name.
 *
 * This is optional. If you don't do anything, i.e. geometry parameter is not set, the
 * default geometry is cartesian.
 *
 * Allowed values for 2D:
 * shell2d
 * disk2d
 * icosahedron (map the sphere)
 *
 * Allowed values for 3D:
 * shell
 * sphere
 *
 * \param[in] name A geometry name.
 * \param[in] conn A connectivity already created.
 * \param[in] cfg A config reader.
 * \return A fully allocated geometry.
 */
p4est_geometry_t *
geometry_2d_new_byname(const char *                name,
                       p4est_connectivity_t *      conn,
                       const kalypsso::ConfigMap & cfg);

p8est_geometry_t *
geometry_3d_new_byname(const char *                name,
                       p8est_connectivity_t *      conn,
                       const kalypsso::ConfigMap & cfg);

#endif // KALYPSSO_UTILS_P4EST_GEOMETRY_H_
