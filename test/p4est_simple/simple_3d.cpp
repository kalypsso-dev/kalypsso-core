// Copyright (C) 2010 The University of Texas System
// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 * Usage: p8est_simple <configuration> <level>
 *        possible configurations:
 *        o unit      The unit cube.
 *        o periodic  The unit cube with all-periodic boundary conditions.
 *        o rotwrap   The unit cube with various self-periodic b.c.
 *        o twocubes  Two connected cubes.
 *        o twowrap   Two cubes with periodically identified far ends.
 *        o rotcubes  A collection of six connected rotated cubes.
 *        o shell     A 24-tree discretization of a hollow sphere.
 *        o sphere    A 13-tree discretization of a solid sphere.
 */

#define VTK_OUTPUT 1

#include <p8est_bits.h>
#include <p8est_extended.h>

#ifdef VTK_OUTPUT
#  include <p8est_vtk.h>
#endif

/* default parameters for p8est_vtk_write_file */
static const int p8est_vtk_write_tree = 1;
static const int p8est_vtk_write_level = 1;
static const int p8est_vtk_write_rank = 1;
static const int p8est_vtk_wrap_rank = 0;


typedef enum
{
  P8EST_CONFIG_NULL,
  P8EST_CONFIG_UNIT,
  P8EST_CONFIG_PERIODIC,
  P8EST_CONFIG_ROTWRAP,
  P8EST_CONFIG_TWOCUBES,
  P8EST_CONFIG_TWOWRAP,
  P8EST_CONFIG_ROTCUBES,
  P8EST_CONFIG_SHELL,
  P8EST_CONFIG_SPHERE
} simple_config_t;

typedef struct
{
  simple_config_t config;
  int             mpisize;
  int             level;
  unsigned        checksum;
} simple_regression_t;

typedef struct
{
  p4est_topidx_t a;
} user_data_t;

typedef struct
{
  sc_MPI_Comm mpicomm;
  int         mpisize;
  int         mpirank;
} mpi_context_t;

static int refine_level = 0;

/* *INDENT-OFF* */
// clang-format off
static const simple_regression_t regression[] =
{ { P8EST_CONFIG_UNIT, 1, 7, 0x88fc2229U },
  { P8EST_CONFIG_UNIT, 3, 6, 0xce19fee3U },
  { P8EST_CONFIG_TWOCUBES, 1, 4, 0xd9e96b31U },
  { P8EST_CONFIG_TWOCUBES, 3, 5, 0xe8b16b4aU },
  { P8EST_CONFIG_TWOWRAP, 1, 4, 0xd3e06e2fU },
  { P8EST_CONFIG_TWOWRAP, 5, 5, 0x920ecd43U },
  { P8EST_CONFIG_PERIODIC, 1, 4, 0x28304c83U },
  { P8EST_CONFIG_PERIODIC, 7, 4, 0x28304c83U },
  { P8EST_CONFIG_PERIODIC, 3, 5, 0xe4d123b2U },
  { P8EST_CONFIG_PERIODIC, 6, 6, 0x81c22cc6U },
  { P8EST_CONFIG_ROTWRAP, 1, 5, 0xe4d123b2U },
  { P8EST_CONFIG_ROTWRAP, 3, 5, 0xe4d123b2U },
  { P8EST_CONFIG_ROTWRAP, 5, 6, 0x81c22cc6U },
  { P8EST_CONFIG_ROTCUBES, 1, 5, 0x5c497bdaU },
  { P8EST_CONFIG_ROTCUBES, 3, 5, 0x5c497bdaU },
  { P8EST_CONFIG_ROTCUBES, 5, 6, 0x00530556U },
  { P8EST_CONFIG_ROTCUBES, 7, 1, 0x47f00071U },
  { P8EST_CONFIG_ROTCUBES, 7, 6, 0x00530556U },
  { P8EST_CONFIG_ROTCUBES, 7, 7, 0x84730f31U },
  { P8EST_CONFIG_ROTCUBES, 9, 1, 0x00600001U },
  { P8EST_CONFIG_NULL, 0, 0, 0 } };
/* *INDENT-ON* */
// clang-format on

static void
init_fn(p8est_t * p8est, p4est_topidx_t which_tree, p8est_quadrant_t * quadrant)
{
  user_data_t * data = (user_data_t *)quadrant->p.user_data;

  data->a = which_tree;
}

static int
refine_sparse_fn(p8est_t * p8est, p4est_topidx_t which_tree, p8est_quadrant_t * quadrant)
{
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
  if (quadrant->x < P8EST_QUADRANT_LEN(2) && quadrant->y > 0 && quadrant->z < P8EST_QUADRANT_LEN(2))
  {
    return 1;
  }

  return 0;
}

static int
refine_normal_fn(p8est_t * p8est, p4est_topidx_t which_tree, p8est_quadrant_t * quadrant)
{
  if ((int)quadrant->level >= (refine_level - (int)(which_tree % 3)))
  {
    return 0;
  }
  if (quadrant->level == 1 && p8est_quadrant_child_id(quadrant) == 3)
  {
    return 1;
  }
  if (quadrant->x == P8EST_LAST_OFFSET(2) && quadrant->y == P8EST_LAST_OFFSET(2))
  {
    return 1;
  }
  if (quadrant->z >= P8EST_QUADRANT_LEN(2))
  {
    return 0;
  }

  return 1;
}

static int
refine_radius_fn(p8est_t * p8est, p4est_topidx_t which_tree, p8est_quadrant_t * quadrant)
{

  /* stop criterion */
  if ((int)quadrant->level >= refine_level)
  {
    return 0;
  }

  p8est_geometry_t * geom = (p8est_geometry_t *)p8est->user_pointer;

  /* logical coordinates */
  double xyz[3] = { 0, 0, 0 };

  /* physical coordinates (after geometry mapping )*/
  double XYZ[3] = { 0, 0, 0 };

  /* half-size of the cell in logical coordinate */
  double       h2 = 0.5 * P8EST_QUADRANT_LEN(quadrant->level) / P8EST_ROOT_LEN;
  const double intsize = 1.0 / P8EST_ROOT_LEN;

  /*
   * get coordinates at cell center
   */
  xyz[0] = intsize * quadrant->x + h2;
  xyz[1] = intsize * quadrant->y + h2;
  xyz[2] = intsize * quadrant->z + h2;

  // apply mapping from logical coordinates to physical coordinates
  geom->X(geom, which_tree, xyz, XYZ);

  double radius = sqrt(XYZ[0] * XYZ[0] + XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]);


  /*if (radius<0.5)
    return 0;*/
  if (radius < 0.7 and quadrant->level > 3)
    return 0;
  if (radius < 0.85 and quadrant->level > 4)
    return 0;
  if (radius < 0.95 and quadrant->level > 5)
    return 0;
  if (radius < 0.99 and quadrant->level > 6)
    return 0;

  return 1;
}

/*
 * A slightly modified output routine, to be able to change the scale parameter.
 */
void
my_p8est_vtk_write_file(p8est_t *          p8est,
                        p8est_geometry_t * geom,
                        const char *       filename,
                        double             scale)
{
  int                   retval;
  p8est_vtk_context_t * cont;

  /* allocate context and set parameters */
  cont = p8est_vtk_context_new(p8est, filename);

  p8est_vtk_context_set_scale(cont, scale);

  p8est_vtk_context_set_geom(cont, geom);

  /* We do not write point data, so it is safe to set continuous to true.
   * This will not save any space though since the default scale is < 1. */
  p8est_vtk_context_set_continuous(cont, 1);

  /* write header, that is, vertex positions and quadrant-to-vertex map */
  cont = p8est_vtk_write_header(cont);
  SC_CHECK_ABORT(cont != NULL, P8EST_STRING "_vtk: Error writing header");

  /* write the tree/level/rank data */
  cont = p8est_vtk_write_cell_dataf(cont,
                                    p8est_vtk_write_tree,
                                    p8est_vtk_write_level,
                                    p8est_vtk_write_rank,
                                    p8est_vtk_wrap_rank,
                                    0,
                                    0,
                                    cont);
  SC_CHECK_ABORT(cont != NULL, P8EST_STRING "_vtk: Error writing cell data");

  /* properly write rest of the files' contents */
  retval = p8est_vtk_write_footer(cont);
  SC_CHECK_ABORT(!retval, P8EST_STRING "_vtk: Error writing footer");
}

int
main(int argc, char ** argv)
{
  int                         mpiret;
  int                         wrongusage;
  unsigned                    crc;
  const char *                usage;
  mpi_context_t               mpi_context, *mpi = &mpi_context;
  p8est_t *                   p8est;
  p8est_connectivity_t *      connectivity;
  p8est_geometry_t *          geom;
  p8est_refine_t              refine_fn;
  p8est_coarsen_t             coarsen_fn;
  simple_config_t             config;
  const simple_regression_t * r;

  /* initialize MPI and p4est internals */
  mpiret = sc_MPI_Init(&argc, &argv);
  SC_CHECK_MPI(mpiret);
  mpi->mpicomm = sc_MPI_COMM_WORLD;
  mpiret = sc_MPI_Comm_size(mpi->mpicomm, &mpi->mpisize);
  SC_CHECK_MPI(mpiret);
  mpiret = sc_MPI_Comm_rank(mpi->mpicomm, &mpi->mpirank);
  SC_CHECK_MPI(mpiret);

  sc_init(mpi->mpicomm, 1, 1, NULL, SC_LP_DEFAULT);
  p4est_init(NULL, SC_LP_DEFAULT);

  /* process command line arguments */
  usage = "Arguments: <configuration> <level>\n"
          "   Configuration can be any of\n"
          "      unit|periodic|rotwrap|twocubes|twowrap|rotcubes|shell|sphere\n"
          "   Level controls the maximum depth of refinement\n";
  wrongusage = 0;
  config = P8EST_CONFIG_NULL;
  if (!wrongusage && argc != 3)
  {
    wrongusage = 1;
  }
  if (!wrongusage)
  {
    if (!strcmp(argv[1], "unit"))
    {
      config = P8EST_CONFIG_UNIT;
    }
    else if (!strcmp(argv[1], "periodic"))
    {
      config = P8EST_CONFIG_PERIODIC;
    }
    else if (!strcmp(argv[1], "rotwrap"))
    {
      config = P8EST_CONFIG_ROTWRAP;
    }
    else if (!strcmp(argv[1], "twocubes"))
    {
      config = P8EST_CONFIG_TWOCUBES;
    }
    else if (!strcmp(argv[1], "twowrap"))
    {
      config = P8EST_CONFIG_TWOWRAP;
    }
    else if (!strcmp(argv[1], "rotcubes"))
    {
      config = P8EST_CONFIG_ROTCUBES;
    }
    else if (!strcmp(argv[1], "shell"))
    {
      config = P8EST_CONFIG_SHELL;
    }
    else if (!strcmp(argv[1], "sphere"))
    {
      config = P8EST_CONFIG_SPHERE;
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

  /* assign variables based on configuration */
  refine_level = atoi(argv[2]);
  refine_fn = refine_normal_fn;
  coarsen_fn = NULL;

  /* create connectivity and forest structures */
  geom = NULL;
  if (config == P8EST_CONFIG_PERIODIC)
  {
    connectivity = p8est_connectivity_new_periodic();
  }
  else if (config == P8EST_CONFIG_ROTWRAP)
  {
    connectivity = p8est_connectivity_new_rotwrap();
  }
  else if (config == P8EST_CONFIG_TWOCUBES)
  {
    connectivity = p8est_connectivity_new_twocubes();
    refine_fn = refine_sparse_fn;
  }
  else if (config == P8EST_CONFIG_TWOWRAP)
  {
    connectivity = p8est_connectivity_new_twowrap();
    refine_fn = refine_sparse_fn;
  }
  else if (config == P8EST_CONFIG_ROTCUBES)
  {
    connectivity = p8est_connectivity_new_rotcubes();
  }
  else if (config == P8EST_CONFIG_SHELL)
  {
    connectivity = p8est_connectivity_new_shell();
    geom = p8est_geometry_new_shell(connectivity, 1., .44);
    refine_fn = refine_radius_fn;
    coarsen_fn = NULL;
  }
  else if (config == P8EST_CONFIG_SPHERE)
  {
    connectivity = p8est_connectivity_new_sphere();
    // geom = p8est_geometry_new_sphere (connectivity, 1., 0.191728, 0.039856);
    geom = p8est_geometry_new_sphere(connectivity, 1., 0.7, 0.5);
    refine_fn = refine_radius_fn;
    coarsen_fn = NULL;
  }
  else
  {
    connectivity = p8est_connectivity_new_unitcube();
  }

  /*
   * set geometry as p4est user pointer so that we can
   * retrieve it in the callbacks.
   */
  p8est = p8est_new_ext(mpi->mpicomm, connectivity, 4, 0, 0, sizeof(user_data_t), init_fn, geom);

  double scale = 1.0;

#ifdef VTK_OUTPUT
  my_p8est_vtk_write_file(p8est, geom, "simple_3d_new", scale);
#endif

  /* refinement and coarsening */
  p8est_refine(p8est, 1, refine_fn, init_fn);
  if (coarsen_fn != NULL)
  {
    p8est_coarsen(p8est, 1, coarsen_fn, init_fn);
  }
#ifdef VTK_OUTPUT
  my_p8est_vtk_write_file(p8est, geom, "simple_3d_refined", scale);
#endif

  /* balance */
  p8est_balance(p8est, P8EST_CONNECT_FULL, init_fn);
#ifdef VTK_OUTPUT
  my_p8est_vtk_write_file(p8est, geom, "simple_3d_balanced", scale);
#endif

  crc = p8est_checksum(p8est);

  /* partition */
  p8est_partition(p8est, 0, NULL);
#ifdef VTK_OUTPUT
  my_p8est_vtk_write_file(p8est, geom, "simple_3d_partition", scale);
#endif

#ifdef P4EST_ENABLE_DEBUG
  /* rebalance should not change checksum */
  p8est_balance(p8est, P8EST_CONNECT_FULL, init_fn);
  P4EST_ASSERT(p8est_checksum(p8est) == crc);
#endif

  /* print and verify forest checksum */
  P4EST_GLOBAL_STATISTICSF("Tree checksum 0x%08x\n", crc);
  if (mpi->mpirank == 0)
  {
    for (r = regression; r->config != P8EST_CONFIG_NULL; ++r)
    {
      if (r->config != config || r->mpisize != mpi->mpisize || r->level != refine_level)
        continue;
      SC_CHECK_ABORT(crc == r->checksum, "Checksum mismatch");
      P4EST_GLOBAL_INFO("Checksum regression OK\n");
      break;
    }
  }

  /* destroy the p8est and its connectivity structure */
  p8est_destroy(p8est);
  if (geom != NULL)
  {
    p8est_geometry_destroy(geom);
  }
  p8est_connectivity_destroy(connectivity);

  /* clean up and exit */
  sc_finalize();

  mpiret = sc_MPI_Finalize();
  SC_CHECK_MPI(mpiret);

  return 0;
}
