// Copyright (C) 2010 The University of Texas System
// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 * Usage: test_p4est_wrapper <configuration> <level>
 *
 *        possible 2D configurations:
 *        o unit       Refinement on the unit square.
 *        o three      Refinement on a forest with three trees.
 *        o brick      Refinement on a regular grid of trees (orchard)
 *        o evil       Check second round of refinement with np=5 level=7
 *        o evil3      Check second round of refinement on three trees
 *        o pillow     Refinement on a 2-tree pillow-shaped domain.
 *        o moebius    Refinement on a 5-tree Moebius band.
 *        o star       Refinement on a 6-tree star shaped domain.
 *        o cubed      Refinement on a 6-tree cubed sphere surface.
 *        o disk       Refinement on a 5-tree spherical disk.
 *        o periodic   Refinement on the unit square with all-periodic b.c.
 *        o rotwrap    Refinement on the unit square with weird periodic b.c.
 *        o icosahedron   Refinement on the sphere
 *        o shell2d       Refinement on a 2d shell with geometry.
 *        o disk2d        Refinement on a 2d disk with geometry.
 *
 *        possible 3D configurations:
 *        o unit3      The unit cube.
 *        o brick3      Refinement on a regular grid of trees (orchard)
 *        o periodic3  The unit cube with all-periodic boundary conditions.
 *        o rotwrap3   The unit cube with various self-periodic b.c.
 *        o twocubes3  Two connected cubes.
 *        o twowrap3   Two cubes with periodically identified far ends.
 *        o rotcubes3  A collection of six connected rotated cubes.
 *        o shell3     A 24-tree discretization of a hollow sphere.
 *        o sphere3    A 13-tree discretization of a solid sphere.
 *        o torus      A configurable discretization of a solid torus.
 */

/*
 * Additional 2d geometry defined in kalypsso / CanoP:
 *        o disk2d    Refinement of a 5-tree mapping the unit disk.
 *        o shell2d   Similar to 3d shell
 *
 * Refinement based on radius.
 *
 */

#include <kalypsso/utils/p4est/p4est_wrapper.h>

#include <cstdlib>
#include <vector>
#include <string>

#include <kalypsso/utils/p4est/connectivity.h>
#include <kalypsso/utils/p4est/geometry.h>


// default parameters for p4est_vtk_write_file
static const int p4est_vtk_write_tree = 1;
static const int p4est_vtk_write_level = 1;
static const int p4est_vtk_write_rank = 1;
static const int p4est_vtk_wrap_rank = 0;

enum simple_config_t
{
  // 2D
  P4EST_CONFIG_NULL,
  P4EST_CONFIG_UNIT,
  P4EST_CONFIG_THREE,
  P4EST_CONFIG_BRICK,
  P4EST_CONFIG_EVIL,
  P4EST_CONFIG_EVIL3,
  P4EST_CONFIG_PILLOW,
  P4EST_CONFIG_MOEBIUS,
  P4EST_CONFIG_STAR,
  P4EST_CONFIG_CUBED,
  P4EST_CONFIG_DISK,
  P4EST_CONFIG_PERIODIC,
  P4EST_CONFIG_ROTWRAP,
  P4EST_CONFIG_ICOSAHEDRON,
  P4EST_CONFIG_SHELL2D,
  P4EST_CONFIG_DISK2D,
  // 3D
  P8EST_CONFIG_NULL,
  P8EST_CONFIG_UNIT,
  P8EST_CONFIG_PERIODIC,
  P8EST_CONFIG_BRICK,
  P8EST_CONFIG_ROTWRAP,
  P8EST_CONFIG_TWOCUBES,
  P8EST_CONFIG_TWOWRAP,
  P8EST_CONFIG_ROTCUBES,
  P8EST_CONFIG_SHELL,
  P8EST_CONFIG_SPHERE,
  P8EST_CONFIG_TORUS
};

struct simple_regression_t
{
  simple_config_t config;
  int             mpisize;
  int             level;
  unsigned        checksum;
};

struct user_data_t
{
  double x; // physical space
};

struct mpi_context_t
{
  sc_MPI_Comm mpicomm;
  int         mpisize;
  int         mpirank;
};

static int refine_level = 0;

template <int dim>
void
quadrant_center_vertex(typename kalypsso::p4est::Wrapper<dim>::connectivity_t * connectivity,
                       typename kalypsso::p4est::Wrapper<dim>::geometry_t *     geom,
                       typename kalypsso::p4est::topidx_t                       tree,
                       typename kalypsso::p4est::Wrapper<dim>::quadrant_t *     quad,
                       double                                                   xyz[3])
{

  using namespace kalypsso::p4est;

  uint32_t root_len = Wrapper<dim>::ROOT_LEN;
  uint32_t quad_len = Wrapper<dim>::QUADRANT_LEN(quad->level);

  qcoord_t half_length = quad_len / 2;

  double       h2 = 0.5 * quad_len / root_len;
  const double intsize = 1.0 / root_len;

  if (geom != nullptr)
  {

    double xyz_logic[3] = { 0., 0., 0. };

    /*
     * get coordinates at cell center
     */
    xyz_logic[0] = intsize * get_x(quad) + h2;
    xyz_logic[1] = intsize * get_y(quad) + h2;
    xyz_logic[2] = dim == 3 ? intsize * get_z(quad) + h2 : 0.0;

    // from logical coordinates to physical coordinates
    geom->X(geom, tree, xyz_logic, xyz);
  }
  else
  { // regular cartesian geometry

    qcoord_t xyz_logic[3] = { get_x(quad) + half_length,
                              get_y(quad) + half_length,
                              get_z(quad) + half_length };

    Wrapper<dim>::qcoord_to_vertex(connectivity, tree, xyz_logic, xyz);

  } // end cartesian geometry

} // quadrant_center_vertex

template <int dim>
static void
init_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   p4est,
        typename kalypsso::p4est::topidx_t                   which_tree,
        typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{

  using namespace kalypsso::p4est;

  using connectivity_t = typename Wrapper<dim>::connectivity_t;
  using geometry_t = typename Wrapper<dim>::geometry_t;

  user_data_t * data = (user_data_t *)quadrant->p.user_data;

  // compute physical space coordinates, and put "x" into user_data

  connectivity_t * conn = (connectivity_t *)p4est->connectivity;
  geometry_t *     geom = (geometry_t *)p4est->user_pointer;

  double XYZ[3];

  quadrant_center_vertex<dim>(conn, geom, which_tree, quadrant, XYZ);

  data->x = XYZ[0];
}

template <int dim>
static int
refine_normal_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   p4est,
                 typename kalypsso::p4est::topidx_t                   which_tree,
                 typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  using namespace kalypsso::p4est;

  if ((int)quadrant->level >= (refine_level - (int)(which_tree % 3)))
  {
    return 0;
  }
  if (quadrant->level == 1 && Wrapper<dim>::quadrant_child_id(quadrant) == 3)
  {
    return 1;
  }
  if (quadrant->x == P4EST_LAST_OFFSET(2) && quadrant->y == P4EST_LAST_OFFSET(2))
  {
    return 1;
  }

  if (dim == 2)
  {
    if (get_x(quadrant) >= (int)Wrapper<dim>::QUADRANT_LEN(2))
    {
      return 0;
    }
  }
  else
  {
    if (get_z(quadrant) >= (int)Wrapper<dim>::QUADRANT_LEN(2))
    {
      return 0;
    }
  }

  return 1;
}

template <int dim>
static int
refine_sparse_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
                 typename kalypsso::p4est::topidx_t                   which_tree,
                 typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  using namespace kalypsso::p4est;
  // uint32_t quad_len = Wrapper<dim>::QUADRANT_LEN (quadrant->level);

  if (which_tree != 0)
  {
    return 0;
  }
  if ((int)quadrant->level >= refine_level)
  {
    return 0;
  }
  if (quadrant->level == 0)
  {
    return 1;
  }
  if (dim == 2)
  {
    if (get_x(quadrant) < (int)Wrapper<dim>::QUADRANT_LEN(2) && get_y(quadrant) > 0)
    {
      return 1;
    }
  }
  else
  {
    if (get_x(quadrant) < (int)Wrapper<dim>::QUADRANT_LEN(2) && get_y(quadrant) > 0 and
        get_z(quadrant) < (int)Wrapper<dim>::QUADRANT_LEN(2))
    {
      return 1;
    }
  }

  return 0;
}

template <int dim>
static int
refine_evil_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
               typename kalypsso::p4est::topidx_t                   which_tree,
               typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  if ((int)quadrant->level >= refine_level)
  {
    return 0;
  }
  if (forest->mpirank <= 1)
  {
    return 1;
  }

  return 0;
}

template <int dim>
static int
refine_evil3_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   p4est,
                typename kalypsso::p4est::topidx_t                   which_tree,
                typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{

  using namespace kalypsso::p4est;

  qcoord_t                          u2;
  typename Wrapper<dim>::quadrant_t ref;

  P4EST_QUADRANT_INIT(&ref);

  u2 = Wrapper<dim>::QUADRANT_LEN(2);

  if (which_tree == 0)
  {
    ref.x = 3 * u2;
    ref.y = 2 * u2;
  }
  else if (which_tree == 1)
  {
    ref.x = 2 * u2;
    ref.y = 3 * u2;
  }
  ref.level = 2;

  if ((int)quadrant->level >= refine_level)
  {
    return 0;
  }
  if ((which_tree == 0 || which_tree == 1) && (Wrapper<dim>::quadrant_is_equal(&ref, quadrant) ||
                                               Wrapper<dim>::quadrant_is_ancestor(&ref, quadrant)))
  {
    return 1;
  }

  return 0;
}

template <int dim>
static int
coarsen_evil_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
                typename kalypsso::p4est::topidx_t                   which_tree,
                typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant[])
{
  if (forest->mpirank >= 2)
  {
    return 1;
  }

  return 0;
}

template <int dim>
static int
refine_radius_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
                 typename kalypsso::p4est::topidx_t                   which_tree,
                 typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{

  using namespace kalypsso::p4est;
  using geometry_t = typename Wrapper<dim>::geometry_t;

  uint32_t root_len = Wrapper<dim>::ROOT_LEN;
  uint32_t quad_len = Wrapper<dim>::QUADRANT_LEN(quadrant->level);

  // stop criterion
  if ((int)quadrant->level >= refine_level)
  {
    return 0;
  }

  geometry_t * geom = (geometry_t *)forest->user_pointer;

  // logical coordinates
  double xyz[3] = { 0, 0, 0 };

  // physical coordinates (after geometry mapping )
  double XYZ[3] = { 0, 0, 0 };

  // half-size of the cell in logical coordinate
  double h2 = 0.5 * quad_len / root_len;

  const double intsize = 1.0 / root_len;

  /*
   * get coordinates at cell center
   */
  xyz[0] = intsize * get_x(quadrant) + h2;
  xyz[1] = intsize * get_y(quadrant) + h2;
  if (dim == 3)
    xyz[2] = intsize * get_z(quadrant) + h2;

  // apply mapping from logical coordinates to physical coordinates
  geom->X(geom, which_tree, xyz, XYZ);

  double radius = 0;
  if (dim == 3)
    radius = sqrt(XYZ[0] * XYZ[0] + XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]);
  else
    radius = sqrt(XYZ[0] * XYZ[0] + XYZ[1] * XYZ[1]);

  if (dim == 2)
  {
    if (radius < 0.7 and quadrant->level > 4)
      return 0;
    if (radius < 0.8 and quadrant->level > 5)
      return 0;
    if (radius < 0.9 and quadrant->level > 6)
      return 0;
  }
  else
  {
    if (radius < 0.7 and quadrant->level > 3)
      return 0;
    if (radius < 0.85 and quadrant->level > 4)
      return 0;
    if (radius < 0.95 and quadrant->level > 5)
      return 0;
    if (radius < 0.99 and quadrant->level > 6)
      return 0;
  }

  return 1;
}

template <int dim>
void
getdata_cb(typename kalypsso::p4est::Wrapper<dim>::volume_info_t * info, void * user_data)
{

  using namespace kalypsso::p4est;

  using forest_t = typename Wrapper<dim>::forest_t;
  using tree_t = typename Wrapper<dim>::tree_t;
  using quadrant_t = typename Wrapper<dim>::quadrant_t;

  /* we passed the array of values to fill as the user_data in the call
     to p4est_iterate */
  sc_array_t * scalar_data = (sc_array_t *)user_data;
  double *     this_data;
  forest_t *   forest = info->p4est;
  quadrant_t * q = info->quad;
  topidx_t     which_tree = info->treeid;
  locidx_t     local_id =
    info->quadid; /* this is the index of q *within its tree's numbering*.  We want to convert it
                     its index for all the quadrants on this process, which we do below */
  tree_t *      tree;
  user_data_t * udata = (user_data_t *)q->p.user_data;
  locidx_t      arrayoffset;

  tree = (tree_t *)(forest->trees->array + sizeof(tree_t) * (size_t)which_tree);

  /* compute id relative inside current MPI process */
  local_id += tree->quadrants_offset;
  arrayoffset = local_id;

  this_data = (double *)sc_array_index(scalar_data, arrayoffset);
  this_data[0] = udata->x;

} // getdata_cb


/*
 * A slightly modified output routine, to be able to change the scale parameter.
 */
template <int dim>
void
my_p4est_vtk_write_file(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
                        typename kalypsso::p4est::Wrapper<dim>::geometry_t * geom,
                        std::string                                          filename,
                        double                                               scale)
{
  using namespace kalypsso::p4est;

  int                                    retval;
  typename Wrapper<dim>::vtk_context_t * cont;

  // allocate context and set parameters
  cont = Wrapper<dim>::vtk_context_new(forest, filename.c_str());

  Wrapper<dim>::vtk_context_set_scale(cont, scale);

  Wrapper<dim>::vtk_context_set_geom(cont, geom);

  // We do not write point data, so it is safe to set continuous to true.
  // This will not save any space though since the default scale is < 1.
  Wrapper<dim>::vtk_context_set_continuous(cont, 1);

  // write header, that is, vertex positions and quadrant-to-vertex map
  cont = Wrapper<dim>::vtk_write_header(cont);
  SC_CHECK_ABORT(cont != NULL, P4EST_STRING "_vtk: Error writing header");

  // fill scalar array
  sc_array_t * scalar_data;
  scalar_data = sc_array_new_size(sizeof(double), forest->local_num_quadrants);

  // Use the iterator to visit every cell and fill scalar_data vector
  Wrapper<dim>::iterate_volume(forest, nullptr, (void *)scalar_data, getdata_cb<dim>);

  // write the tree/level/rank data
  cont = Wrapper<dim>::vtk_write_cell_dataf(cont,
                                            p4est_vtk_write_tree,
                                            p4est_vtk_write_level,
                                            p4est_vtk_write_rank,
                                            p4est_vtk_wrap_rank,
                                            1, // writing one scalar field
                                            0, // not writing any vector field
                                            "xxx",
                                            scalar_data,
                                            cont);
  SC_CHECK_ABORT(cont != nullptr, P4EST_STRING "_vtk: Error writing cell data");

  // properly write rest of the files' contents
  retval = Wrapper<dim>::vtk_write_footer(cont);
  SC_CHECK_ABORT(!retval, P4EST_STRING "_vtk: Error writing footer");

  sc_array_destroy(scalar_data);

  // !!! context is destroyed in write_footer !!!
  // Wrapper<dim>::vtk_context_destroy(cont);
}

/* ================================================================= */
template <int dim>
typename kalypsso::p4est::Wrapper<dim>::connectivity_t *
create_connectivity(simple_config_t config, int brick_dim[3]);

template <>
kalypsso::p4est::Wrapper<2>::connectivity_t *
create_connectivity<2>(simple_config_t config, int brick_dim[3])
{

  typename kalypsso::p4est::Wrapper<2>::connectivity_t * connectivity = nullptr;

  if (config == P4EST_CONFIG_UNIT)
  {
    connectivity = p4est_connectivity_new_unitsquare();
  }
  else if (config == P4EST_CONFIG_THREE || config == P4EST_CONFIG_EVIL3)
  {
    connectivity = p4est_connectivity_new_corner();
  }
  else if (config == P4EST_CONFIG_BRICK)
  {
    connectivity = p4est_connectivity_new_brick(brick_dim[0], brick_dim[1], 1, 1);
  }
  else if (config == P4EST_CONFIG_PILLOW)
  {
    connectivity = p4est_connectivity_new_pillow();
  }
  else if (config == P4EST_CONFIG_MOEBIUS)
  {
    connectivity = p4est_connectivity_new_moebius();
  }
  else if (config == P4EST_CONFIG_STAR)
  {
    connectivity = p4est_connectivity_new_star();
  }
  else if (config == P4EST_CONFIG_CUBED)
  {
    connectivity = p4est_connectivity_new_cubed();
  }
  else if (config == P4EST_CONFIG_DISK)
  {
#if defined(KALYPSSO_CORE_USE_OLD_P4EST_API)
    connectivity = p4est_connectivity_new_disk();
#else
    // non periodic disk
    connectivity = p4est_connectivity_new_disk(0, 0);
#endif // USE_OLD_P4EST_API
  }
  else if (config == P4EST_CONFIG_PERIODIC)
  {
    connectivity = p4est_connectivity_new_periodic();
  }
  else if (config == P4EST_CONFIG_ROTWRAP)
  {
    connectivity = p4est_connectivity_new_rotwrap();
  }
  else if (config == P4EST_CONFIG_ICOSAHEDRON)
  {
    connectivity = p4est_connectivity_new_icosahedron();
  }
  else if (config == P4EST_CONFIG_SHELL2D)
  {
    connectivity = p4est_connectivity_new_shell2d();
  }
  else if (config == P4EST_CONFIG_DISK2D)
  {
    connectivity = p4est_connectivity_new_disk2d();
  }

  return connectivity;
}

template <>
kalypsso::p4est::Wrapper<3>::connectivity_t *
create_connectivity<3>(simple_config_t config, int brick_dim[3])
{

  kalypsso::p4est::Wrapper<3>::connectivity_t * connectivity = nullptr;

  if (config == P8EST_CONFIG_UNIT)
  {
    connectivity = p8est_connectivity_new_unitcube();
  }
  else if (config == P8EST_CONFIG_PERIODIC)
  {
    connectivity = p8est_connectivity_new_periodic();
  }
  else if (config == P8EST_CONFIG_BRICK)
  {
    connectivity = p8est_connectivity_new_brick(brick_dim[0], brick_dim[1], brick_dim[2], 1, 1, 1);
  }
  else if (config == P8EST_CONFIG_ROTWRAP)
  {
    connectivity = p8est_connectivity_new_rotwrap();
  }
  else if (config == P8EST_CONFIG_TWOCUBES)
  {
    connectivity = p8est_connectivity_new_twocubes();
  }
  else if (config == P8EST_CONFIG_TWOWRAP)
  {
    connectivity = p8est_connectivity_new_twowrap();
  }
  else if (config == P8EST_CONFIG_ROTCUBES)
  {
    connectivity = p8est_connectivity_new_rotcubes();
  }
  else if (config == P8EST_CONFIG_SHELL)
  {
    connectivity = p8est_connectivity_new_shell();
  }
  else if (config == P8EST_CONFIG_SPHERE)
  {
    connectivity = p8est_connectivity_new_sphere();
  }
  else if (config == P8EST_CONFIG_TORUS)
  {
    connectivity = p8est_connectivity_new_torus(8);
  }

  return connectivity;
}

/* ================================================================= */
template <int dim>
typename kalypsso::p4est::Wrapper<dim>::geometry_t *
create_geometry(simple_config_t                                          config,
                typename kalypsso::p4est::Wrapper<dim>::connectivity_t * connectivity,
                double                                                   R0,
                double                                                   R1);

template <>
kalypsso::p4est::Wrapper<2>::geometry_t *
create_geometry<2>(simple_config_t                               config,
                   kalypsso::p4est::Wrapper<2>::connectivity_t * connectivity,
                   double                                        R0,
                   double                                        R1)
{

  kalypsso::p4est::Wrapper<2>::geometry_t * geom = nullptr;

  if (config == P4EST_CONFIG_ICOSAHEDRON)
  {
    geom = p4est_geometry_new_icosahedron(connectivity, R0);
  }
  else if (config == P4EST_CONFIG_SHELL2D)
  {
    geom = p4est_geometry_new_shell2d(connectivity, R0, R1);
  }
  else if (config == P4EST_CONFIG_DISK2D)
  {
    geom = p4est_geometry_new_disk2d(connectivity, R0, R1);
  }

  return geom;
}

template <>
kalypsso::p4est::Wrapper<3>::geometry_t *
create_geometry<3>(simple_config_t                               config,
                   kalypsso::p4est::Wrapper<3>::connectivity_t * connectivity,
                   double                                        R0,
                   double                                        R1)
{

  kalypsso::p4est::Wrapper<3>::geometry_t * geom = nullptr;

  if (config == P8EST_CONFIG_SHELL)
  {
    geom = p8est_geometry_new_shell(connectivity, 1., .44);
  }
  else if (config == P8EST_CONFIG_SPHERE)
  {
    geom = p8est_geometry_new_sphere(connectivity, 1., 0.7, 0.5);
  }
  else if (config == P8EST_CONFIG_TORUS)
  {
    geom = p8est_geometry_new_torus(connectivity, 0.44, 1.0, 3.0);
  }

  return geom;
}

/* ============================================================ */
/* ============================================================ */
/* ============================================================ */
template <int dim>
void
run_test(simple_config_t config,
         std::string     config_name,
         mpi_context_t * mpi,
         int             argc,
         char *          argv[])
{

  if (dim == 2 and mpi->mpirank == 0)
    printf("Running a 2D test\n");
  if (dim == 3 and mpi->mpirank == 0)
    printf("Running a 3D test\n");

  using namespace kalypsso::p4est;

  typename Wrapper<dim>::forest_t *       forest;
  typename Wrapper<dim>::connectivity_t * connectivity = nullptr;
  typename Wrapper<dim>::geometry_t *     geom = nullptr;
  typename Wrapper<dim>::refine_cb_t      refine_fn;
  typename Wrapper<dim>::coarsen_cb_t     coarsen_fn;
  using balance_type_t = typename Wrapper<dim>::balance_type_t;
  unsigned crc;

  // only meaningful when using brick connectivity
  int brick_dim[3];
  brick_dim[0] = argc > 3 ? atoi(argv[3]) : 2;
  brick_dim[1] = argc > 4 ? atoi(argv[4]) : 3;
  brick_dim[2] = argc > 5 ? atoi(argv[5]) : 4;

  // create connectivity and forest structures
  connectivity = create_connectivity<dim>(config, brick_dim);

  // create geometry
  double R0 = 0.44;
  double R1 = 1.0;
  if (argc >= 4)
    R0 = atof(argv[3]);
  if (argc >= 5)
    R1 = atof(argv[4]);
  geom = create_geometry<dim>(config, connectivity, R0, R1);

  // assign refine_fn, coarsen_fn
  refine_fn = refine_normal_fn<dim>;
  coarsen_fn = nullptr;

  if (config == P4EST_CONFIG_EVIL)
  {
    refine_fn = refine_evil_fn<dim>;
    coarsen_fn = coarsen_evil_fn<dim>;
  }
  else if (config == P4EST_CONFIG_EVIL3)
  {
    refine_fn = refine_evil3_fn<dim>;
    coarsen_fn = nullptr;
  }
  else if (config == P4EST_CONFIG_ICOSAHEDRON)
  {
    // refine_fn = refine_icosahedron_fn;
    coarsen_fn = nullptr;
  }
  else if (config == P4EST_CONFIG_DISK2D)
  {
    // refine_fn = refine_radius_fn<dim>;
    coarsen_fn = nullptr;
  }
  else if (config == P4EST_CONFIG_SHELL2D)
  {
    // refine_fn = refine_radius_fn<dim>;
    coarsen_fn = nullptr;
  }
  else if (config == P8EST_CONFIG_TWOCUBES)
  {
    refine_fn = refine_sparse_fn<dim>;
  }
  else if (config == P8EST_CONFIG_TWOWRAP)
  {
    refine_fn = refine_sparse_fn<dim>;
  }
  else
  {
    refine_fn = refine_normal_fn<dim>;
    coarsen_fn = nullptr;
  }

  /*
   * set geometry as p4est user pointer so that we can
   * retrieve it in the callbacks.
   */
  forest = Wrapper<dim>::new_forest(
    mpi->mpicomm, connectivity, dim == 2 ? 15 : 8, 0, 0, sizeof(user_data_t), init_fn<dim>, geom);

  double scale = 1.0;

  my_p4est_vtk_write_file<dim>(forest, geom, config_name + std::string("_new"), scale);

  // refinement and coarsening
  Wrapper<dim>::refine(forest, 1, refine_fn, init_fn<dim>);
  if (coarsen_fn != nullptr)
  {
    Wrapper<dim>::coarsen(forest, 1, coarsen_fn, init_fn<dim>);
  }
  my_p4est_vtk_write_file<dim>(forest, geom, config_name + std::string("_refined"), scale);

  // balance
  Wrapper<dim>::balance(
    forest,
    (dim == 2 ? balance_type_t(P4EST_CONNECT_FULL) : balance_type_t(P8EST_CONNECT_FULL)),
    init_fn<dim>);
  my_p4est_vtk_write_file<dim>(forest, geom, config_name + std::string("_balanced"), scale);
  crc = Wrapper<dim>::checksum(forest);
  (void)crc;

  // partition
  Wrapper<dim>::partition(forest, 0, nullptr);
  my_p4est_vtk_write_file<dim>(forest, geom, config_name + std::string("_partition"), scale);

#ifdef P4EST_ENABLE_DEBUG
  // rebalance should not change checksum
  Wrapper<dim>::balance(
    forest,
    (dim == 2 ? balance_type_t(P4EST_CONNECT_FULL) : balance_type_t(P8EST_CONNECT_FULL)),
    init_fn<dim>);
  P4EST_ASSERT(Wrapper<dim>::checksum(forest) == crc);
#endif

  // destroy the p4est and its connectivity structure
  Wrapper<dim>::destroy(forest);
  if (geom != nullptr)
  {
    Wrapper<dim>::geometry_destroy(geom);
  }
  Wrapper<dim>::connectivity_destroy(connectivity);

} // run p4est test


// ======================================================
// ======================================================
// ======================================================
int
main(int argc, char ** argv)
{

  int             mpiret;
  mpi_context_t   mpi_context, *mpi = &mpi_context;
  simple_config_t config;
  int             wrongusage;
  const char *    usage;

  // initialize MPI and p4est internals
  mpiret = sc_MPI_Init(&argc, &argv);
  SC_CHECK_MPI(mpiret);
  mpi->mpicomm = sc_MPI_COMM_WORLD;
  mpiret = sc_MPI_Comm_size(mpi->mpicomm, &mpi->mpisize);
  SC_CHECK_MPI(mpiret);
  mpiret = sc_MPI_Comm_rank(mpi->mpicomm, &mpi->mpirank);
  SC_CHECK_MPI(mpiret);

  sc_init(mpi->mpicomm, 1, 1, nullptr, SC_LP_DEFAULT);
  p4est_init(nullptr, SC_LP_DEFAULT);

  // Process command line arguments
  usage =
    "Arguments: <configuration> <level>\n"
    "   2D configuration can be any of\n"
    "      unit|three|brick|evil|evil3|pillow|moebius|\n"
    "      star|cubed|disk|periodic|rotwrap|isosahedron|disk2d|shell2d\n"
    "   3D configuration can be any of\n"
    "      unit3|periodic3|brick3|rotwrap3|twocubes3|twowrap3|rotcubes3|shell3|sphere3|torus\n"
    "   Level controls the maximum depth of refinement\n"
    "\n"
    "Example run:\n"
    "  mpirun -np 3 ./test_p4est_wrapper unit 6\n"
    "  mpirun -np 6 ./test_p4est_wrapper brick 5 5 3\n"
    "  mpirun -np 4 ./test_p4est_wrapper torus 5\n";
  wrongusage = 0;
  config = P4EST_CONFIG_NULL;
  if (!wrongusage && argc < 3)
  {
    wrongusage = 1;
  }
  if (!wrongusage)
  {
    if (!strcmp(argv[1], "unit"))
    {
      config = P4EST_CONFIG_UNIT;
    }
    else if (!strcmp(argv[1], "three"))
    {
      config = P4EST_CONFIG_THREE;
    }
    else if (!strcmp(argv[1], "brick"))
    {
      config = P4EST_CONFIG_BRICK;
    }
    else if (!strcmp(argv[1], "evil"))
    {
      config = P4EST_CONFIG_EVIL;
    }
    else if (!strcmp(argv[1], "evil3"))
    {
      config = P4EST_CONFIG_EVIL3;
    }
    else if (!strcmp(argv[1], "pillow"))
    {
      config = P4EST_CONFIG_PILLOW;
    }
    else if (!strcmp(argv[1], "moebius"))
    {
      config = P4EST_CONFIG_MOEBIUS;
    }
    else if (!strcmp(argv[1], "star"))
    {
      config = P4EST_CONFIG_STAR;
    }
    else if (!strcmp(argv[1], "cubed"))
    {
      config = P4EST_CONFIG_CUBED;
    }
    else if (!strcmp(argv[1], "disk"))
    {
      config = P4EST_CONFIG_DISK;
    }
    else if (!strcmp(argv[1], "periodic"))
    {
      config = P4EST_CONFIG_PERIODIC;
    }
    else if (!strcmp(argv[1], "rotwrap"))
    {
      config = P4EST_CONFIG_ROTWRAP;
    }
    else if (!strcmp(argv[1], "icosahedron"))
    {
      config = P4EST_CONFIG_ICOSAHEDRON;
    }
    else if (!strcmp(argv[1], "shell2d"))
    {
      config = P4EST_CONFIG_SHELL2D;
    }
    else if (!strcmp(argv[1], "disk2d"))
    {
      config = P4EST_CONFIG_DISK2D;
    }
    // 3D config
    else if (!strcmp(argv[1], "unit3"))
    {
      config = P8EST_CONFIG_UNIT;
    }
    else if (!strcmp(argv[1], "periodic3"))
    {
      config = P8EST_CONFIG_PERIODIC;
    }
    else if (!strcmp(argv[1], "brick3"))
    {
      config = P8EST_CONFIG_BRICK;
    }
    else if (!strcmp(argv[1], "rotwrap3"))
    {
      config = P8EST_CONFIG_ROTWRAP;
    }
    else if (!strcmp(argv[1], "twocubes3"))
    {
      config = P8EST_CONFIG_TWOCUBES;
    }
    else if (!strcmp(argv[1], "twowrap3"))
    {
      config = P8EST_CONFIG_TWOWRAP;
    }
    else if (!strcmp(argv[1], "rotcubes3"))
    {
      config = P8EST_CONFIG_ROTCUBES;
    }
    else if (!strcmp(argv[1], "shell3"))
    {
      config = P8EST_CONFIG_SHELL;
    }
    else if (!strcmp(argv[1], "sphere3"))
    {
      config = P8EST_CONFIG_SPHERE;
    }
    else if (!strcmp(argv[1], "torus"))
    {
      config = P8EST_CONFIG_TORUS;
    }
    else
    {
      wrongusage = 1;
    }
  }
  if (wrongusage)
  {
    P4EST_GLOBAL_LERROR(usage);
    sc_abort_collective("Usage error");
  }

  // assign variables based on configuration
  refine_level = atoi(argv[2]);


  // run some tests
  if (config <= P4EST_CONFIG_DISK2D)
    run_test<2>(config, std::string(argv[1]), mpi, argc, argv);
  else
    run_test<3>(config, std::string(argv[1]), mpi, argc, argv);

  // clean up and exit
  sc_finalize();

  mpiret = sc_MPI_Finalize();
  SC_CHECK_MPI(mpiret);

  return 0;
}
