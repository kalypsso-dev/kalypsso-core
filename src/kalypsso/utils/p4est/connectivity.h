// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * @file connectivity.h
 * @date 17 octobre 2020
 * @author pkestene
 * @note
 *
 * This file is part of the Kalypsso software project.
 *
 */
#ifndef KALYPSSO_UTILS_P4EST_CONNECTIVITY_H_
#define KALYPSSO_UTILS_P4EST_CONNECTIVITY_H_

// #include "p4est_wrapper.h"

#include <p4est_connectivity.h>
#include <p8est_connectivity.h>

// class ConfigMap;
#include <kalypsso/utils/config/ConfigMap.h>

typedef enum
{
  CONNECTIVITY_PERIODIC_FALSE = 0,
  CONNECTIVITY_PERIODIC_TRUE = 1
} connectivity_periodic_t;


/**
 * \brief Connectivity with two side-by-side trees along direction x.
 *
 * The trees can be periodic in any direction.
 */
p4est_connectivity_t *
p4est_connectivity_new_two(connectivity_periodic_t is_periodic_x,
                           connectivity_periodic_t is_periodic_y);


/**
 * \brief Connectivity with two side-by-side trees along direction x.
 *
 * The trees can be periodic in any direction.
 */
p8est_connectivity_t *
p8est_connectivity_new_two(connectivity_periodic_t is_periodic_x,
                           connectivity_periodic_t is_periodic_y,
                           connectivity_periodic_t is_periodic_z);


/**
 * \brief 2D connectivity with two side-by-side trees.
 */
p4est_connectivity_t *
p4est_connectivity_new_two_simple(void);

/**
 * \brief 3D connectivity with two side-by-side trees.
 */
p8est_connectivity_t *
p8est_connectivity_new_two_simple(void);

/**
 * \brief 2D connectivity with 10 side-by-side trees.
 *
 * The trees are periodic in the y direction.
 */
p4est_connectivity_t *
p4est_connectivity_new_shock_tube(void);

/**
 * \brief 3D connectivity with 10 side-by-side trees.
 *
 * The trees are periodic in the y direction.
 */
p8est_connectivity_t *
p8est_connectivity_new_shock_tube(void);

/**
 * \brief 2D connectivity with five trees in a tetris-like shape.
 *
 *   _ _ _ _
 *  |_|_|_|_|
 *    |_|
 *
 * Tree numbering:
 *  0 1 2 3
 *    4
 *
 * Vertex numbering:
 * 0  1  2  3  4
 * 5  6  7  8  9
 *   10 11
 *
 * Note: the 3D equivalent is possible, just use the abaqus input file "tetris3d.inp"
 *
 */
p4est_connectivity_t *
p4est_connectivity_new_tetris(void);

/**
 * \brief 2D connectivity with ring shape (cylindrical geometry).
 *
 * No geometry required.
 *
 * \param[in] num_trees_radial number of trees in the radial direction
 * \param[in] num_trees_orthoradial number of trees in the orthoradial direction
 * \param[in] rMin radius of the inner border
 * \param[in] rMax radius of the outer border
 */
p4est_connectivity_t *
p4est_connectivity_new_ring(int    num_trees_radial,
                            int    num_trees_orthoradial,
                            double rMin,
                            double rMax);

/**
 * \brief 2D connectivity for forward facing step test (small).
 *
 *
 * 12 trees.
 * 21 vertices.
 *  6 corners.
 * _ _ _ _ _
 *|_|_|_|_|_|
 *|_|_|_|_|_|
 *|_|_|
 *
 */
p4est_connectivity_t *
p4est_connectivity_new_forward_facing_step_small(void);

/**
 * \brief 2D connectivity for forward facing step test.
 *
 *
 * 15 trees along x direction and 4 rows + 3 extra trees (below left).
 * That is 63 trees in total.
 * _ _ _ _ _ _ _ _ _ _
 *|_|_|_|_|_|_|_|_|_|_|
 *|_|_|_|_|_|_|_|_|_|_|
 *|_|_|_|_|_|_|_|_|_|_|
 *|_|_|_|_|_|_|_|_|_|_|
 *|_|_|_|
 *
 */
p4est_connectivity_t *
p4est_connectivity_new_forward_facing_step(void);

/**
 * \brief 2D connectivity for backward facing step test.
 *
 *
 * 3 trees
 * _ _
 *|_|_|
 *  |_|
 *
 */
p4est_connectivity_t *
p4est_connectivity_new_backward_facing_step(void);

/**
 * \brief Print all the information in the 2d connectivity struct.
 */
void
connectivity_print(p4est_connectivity_t * c);

/**
 * \brief Print all the information in the 3d connectivity struct.
 */
void
connectivity_print(p8est_connectivity_t * c);

/**
 * \brief Get a connectivity by name.
 *
 * Allowed p4est values for 2D:
 * brick, corner, cubed, disk, disk2d, icosahedron, moebius, periodic, pillow, rotwrap, shell2d,
 * star, unit
 *
 * Allowed p4est values for 3D:
 * brick, periodic, rotcubes, rotwrap, shell, sphere, twocubes, twowrap, unit
 *
 * Besides the p4est names found in p4est_connectivity.h, we have:
 *      two         connectivity_new_two
 *      two_simple  connectivity_new_two_simple
 *      shock_tube  connectivity_new_shock_tube
 *      tetris, backward_facing_step, ring
 *
 * This function first looks for p4est provided connectivities, if the name
 * is not found there, it falls back to our own and as a last resort, tries
 * to read the connectivity with the filename "name" using
 * p4est_connectivity_read_inp.
 *
 * \param[in] name A connectivity name.
 * \param[in] cfg A config reader.
 * \return A fully allocated connectivity.
 */
p4est_connectivity_t *
connectivity_2d_new_byname(const char * name, const kalypsso::ConfigMap & cfg);

p8est_connectivity_t *
connectivity_3d_new_byname(const char * name, const kalypsso::ConfigMap & cfg);

#endif // KALYPSSO_UTILS_P4EST_CONNECTIVITY_H_
