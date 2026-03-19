// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * @file geometry.cpp
 * @date 17 octobre 2020
 * @author pkestene
 * @note
 *
 * This file is part of the Kalypsso software project.
 *
 */
#include "geometry.h"

#include <kalypsso/utils/config/ConfigMap.h>

/*
 * basic types - 2d and 3d.
 */
enum p4est_geometry_builtin_type_t
{
  P4EST_GEOMETRY_BUILTIN_MAGIC =
    0x65F2F8DE, /* should be different from P8EST_GEOMETRY_BUILTIN_MAGIC ? */
  P4EST_GEOMETRY_BUILTIN_CYLINDRICAL_COMPUTE,
  P4EST_GEOMETRY_BUILTIN_CYLINDRICAL_IO,
};


/* CYLINDRICAL_COMPUTE 2D and 3D */
struct p4est_geometry_builtin_cylindrical_compute_t
{
  p4est_geometry_builtin_type_t type;
  double                        R0, R1;
  int                           nbTrees_theta;
  int                           nbTrees_z;
};


/* CYLINDRICAL_IO 2D and 3D */
struct p4est_geometry_builtin_cylindrical_io_t
{
  p4est_geometry_builtin_type_t type;
  double                        R0, R1;
  int                           nbTrees_theta;
  int                           nbTrees_z;
};


/*
 * This is one the uglyest thing that can be done in C,
 * for now, just keep it as it is done in p4est...
 */
struct p4est_geometry_builtin_t
{
  /** The geom member needs to come first; we cast to p4est_geometry_t * */
  p4est_geometry_t geom;
  union
  {
    p4est_geometry_builtin_type_t                type;
    p4est_geometry_builtin_cylindrical_compute_t cylindrical_compute;
    p4est_geometry_builtin_cylindrical_io_t      cylindrical_io;
  } p;
};

struct p8est_geometry_builtin_t
{
  /** The geom member needs to come first; we cast to p8est_geometry_t * */
  p8est_geometry_t geom;
  union
  {
    p4est_geometry_builtin_type_t                type;
    p4est_geometry_builtin_cylindrical_compute_t cylindrical_compute;
    p4est_geometry_builtin_cylindrical_io_t      cylindrical_io;
  } p;
};

/****************************************************
 * CYLINDRICAL_COMPUTE
 ****************************************************/
/**
 * geometric coordinate transformation for cylindrical_compute geometry.
 *
 * Define the geometric transformation from logical space (where AMR
 * is performed) to the physical space.
 *
 * \param[in]  p4est      the forest
 * \param[in]  which_tree tree id inside forest
 * \param[in]  rst        coordinates in AMR space : [0,1]^3
 * \param[out] xyz        cartesian coordinates in physical space after geometry
 *
 * Note abc[3] contains cartesian coordinates in logical
 * vertex space (before geometry).
 */
static void
p4est_geometry_cylindrical_compute_X(p4est_geometry_t * geom,
                                     p4est_topidx_t     which_tree,
                                     const double       rst[3],
                                     double             xyz[3])
{
  const p4est_geometry_builtin_cylindrical_compute_t * cylindrical_compute =
    &(reinterpret_cast<p4est_geometry_builtin_t *>(geom))->p.cylindrical_compute;
  // double              x, y, z;
  double R0 = cylindrical_compute->R0;
  double R1 = cylindrical_compute->R1;

  int                  nbTrees_theta = cylindrical_compute->nbTrees_theta;
  [[maybe_unused]] int nbTrees_z = cylindrical_compute->nbTrees_z;
  double               abc[3];

  /* transform from the reference cube [0,1]^3 into logical vertex space
     using bi/trilinear transformation */
  p4est_geometry_connectivity_X(geom, which_tree, rst, abc);

  /*
   * assert that input points are in the expected range
   * Note: maybe we should remove these assert, this would allow
   * ghost quadrant at external boundary to call this routine ?
   */
  P4EST_ASSERT(cylindrical_compute->type == P4EST_GEOMETRY_BUILTIN_CYLINDRICAL_COMPUTE);
  P4EST_ASSERT(0 <= which_tree && which_tree < nbTrees_theta * nbTrees_z);
  P4EST_ASSERT(abc[0] < 1.0 + SC_1000_EPS && abc[0] > 0.0 - SC_1000_EPS);
  P4EST_ASSERT(abc[1] < nbTrees_theta + SC_1000_EPS && abc[1] > 0.0 - SC_1000_EPS);

  // abc[0] is in range [0..1]
  // radius is in range [R0..R1]
  xyz[0] = R0 + abc[0] * (R1 - R0);

  // abc[1] is in range [0..nbTrees_theta]
  // theta  is in range [0, 2 M_PI]
  xyz[1] = abc[1] / nbTrees_theta * 2 * M_PI;

  // abc[2] is in range [0..nbTrees_z]
  // just make it zero-centered
  xyz[2] = 0.0;

} // p4est_geometry_cylindrical_compute_X

static void
p8est_geometry_cylindrical_compute_X(p8est_geometry_t * geom,
                                     p4est_topidx_t     which_tree,
                                     const double       rst[3],
                                     double             xyz[3])
{
  const p4est_geometry_builtin_cylindrical_compute_t * cylindrical_compute =
    &(reinterpret_cast<p8est_geometry_builtin_t *>(geom))->p.cylindrical_compute;
  // double              x, y, z;
  double R0 = cylindrical_compute->R0;
  double R1 = cylindrical_compute->R1;

  int                  nbTrees_theta = cylindrical_compute->nbTrees_theta;
  [[maybe_unused]] int nbTrees_z = cylindrical_compute->nbTrees_z;
  double               abc[3];

  /* transform from the reference cube [0,1]^3 into logical vertex space
     using bi/trilinear transformation */
  p8est_geometry_connectivity_X(geom, which_tree, rst, abc);

  /*
   * assert that input points are in the expected range
   * Note: maybe we should remove these assert, this would allow
   * ghost quadrant at external boundary to call this routine ?
   */
  P4EST_ASSERT(cylindrical_compute->type == P4EST_GEOMETRY_BUILTIN_CYLINDRICAL_COMPUTE);
  P4EST_ASSERT(0 <= which_tree && which_tree < nbTrees_theta * nbTrees_z);
  P4EST_ASSERT(abc[0] < 1.0 + SC_1000_EPS && abc[0] > 0.0 - SC_1000_EPS);
  P4EST_ASSERT(abc[1] < nbTrees_theta + SC_1000_EPS && abc[1] > 0.0 - SC_1000_EPS);

  P4EST_ASSERT(abc[2] < nbTrees_z + SC_1000_EPS && abc[2] > 0.0 - SC_1000_EPS);


  // abc[0] is in range [0..1]
  // radius is in range [R0..R1]
  xyz[0] = R0 + abc[0] * (R1 - R0);

  // abc[1] is in range [0..nbTrees_theta]
  // theta  is in range [0, 2 M_PI]
  xyz[1] = abc[1] / nbTrees_theta * 2 * M_PI;

  // abc[2] is in range [0..nbTrees_z]
  // just make it zero-centered

  xyz[2] = abc[2] - 0.5 * nbTrees_z;

} // p8est_geometry_cylindrical_compute_X

p4est_geometry_t *
p4est_geometry_new_cylindrical_compute(p4est_connectivity_t * conn,
                                       double                 R0,
                                       double                 R1,
                                       int                    nbTrees_theta,
                                       int                    nbTrees_z)
{

  p4est_geometry_builtin_t *                     builtin;
  p4est_geometry_builtin_cylindrical_compute_t * cylindrical_compute;

  builtin = P4EST_ALLOC_ZERO(p4est_geometry_builtin_t, 1);

  cylindrical_compute = &builtin->p.cylindrical_compute;
  cylindrical_compute->type = P4EST_GEOMETRY_BUILTIN_CYLINDRICAL_COMPUTE;
  cylindrical_compute->R0 = R0;
  cylindrical_compute->R1 = R1;
  cylindrical_compute->nbTrees_theta = nbTrees_theta;
  cylindrical_compute->nbTrees_z = nbTrees_z;

  builtin->geom.name = "p4est_cylindrical_compute";
  builtin->geom.user = conn;
  builtin->geom.X = p4est_geometry_cylindrical_compute_X;

  return reinterpret_cast<p4est_geometry_t *>(builtin);

} /* p4est_geometry_new_cylindrical_compute */

p8est_geometry_t *
p8est_geometry_new_cylindrical_compute(p8est_connectivity_t * conn,
                                       double                 R0,
                                       double                 R1,
                                       int                    nbTrees_theta,
                                       int                    nbTrees_z)
{

  p8est_geometry_builtin_t *                     builtin;
  p4est_geometry_builtin_cylindrical_compute_t * cylindrical_compute;

  builtin = P4EST_ALLOC_ZERO(p8est_geometry_builtin_t, 1);

  cylindrical_compute = &builtin->p.cylindrical_compute;
  cylindrical_compute->type = P4EST_GEOMETRY_BUILTIN_CYLINDRICAL_COMPUTE;
  cylindrical_compute->R0 = R0;
  cylindrical_compute->R1 = R1;
  cylindrical_compute->nbTrees_theta = nbTrees_theta;
  cylindrical_compute->nbTrees_z = nbTrees_z;

  builtin->geom.name = "p4est_cylindrical_compute";
  builtin->geom.user = conn;
  builtin->geom.X = p8est_geometry_cylindrical_compute_X;

  return reinterpret_cast<p8est_geometry_t *>(builtin);

} /* p8est_geometry_new_cylindrical_compute */

/****************************************************
 * CYLINDRICAL_IO
 ****************************************************/
/**
 * geometric coordinate transformation for cylindrical_io geometry.
 *
 * Define the geometric transformation from logical space (where AMR
 * is performed) to the physical space.
 *
 * \param[in]  p4est      the forest
 * \param[in]  which_tree tree id inside forest
 * \param[in]  rst        coordinates in AMR space : [0,1]^3
 * \param[out] xyz        cartesian coordinates in physical space after geometry
 *
 * Note abc[3] contains cartesian coordinates in logical
 * vertex space (before geometry).
 */
static void
p4est_geometry_cylindrical_io_X(p4est_geometry_t * geom,
                                p4est_topidx_t     which_tree,
                                const double       rst[3],
                                double             xyz[3])
{
  const p4est_geometry_builtin_cylindrical_io_t * cylindrical_io =
    &(reinterpret_cast<p4est_geometry_builtin_t *>(geom))->p.cylindrical_io;
  // double              x, y, z;
  double R0 = cylindrical_io->R0;
  double R1 = cylindrical_io->R1;

  int                  nbTrees_theta = cylindrical_io->nbTrees_theta;
  [[maybe_unused]] int nbTrees_z = cylindrical_io->nbTrees_z;
  double               abc[3];

  double r, theta;

  /* transform from the reference cube [0,1]^3 into logical vertex space
     using bi/trilinear transformation */
  p4est_geometry_connectivity_X(geom, which_tree, rst, abc);

  /*
   * assert that input points are in the expected range
   * Note: maybe we should remove these assert, this would allow
   * ghost quadrant at external boundary to call this routine ?
   */
  P4EST_ASSERT(cylindrical_io->type == P4EST_GEOMETRY_BUILTIN_CYLINDRICAL_IO);
  P4EST_ASSERT(0 <= which_tree && which_tree < nbTrees_theta * nbTrees_z);
  P4EST_ASSERT(abc[0] < 1.0 + SC_1000_EPS && abc[0] > 0.0 - SC_1000_EPS);
  P4EST_ASSERT(abc[1] < nbTrees_theta + SC_1000_EPS && abc[1] > 0.0 - SC_1000_EPS);

  // abc[0] is in range [0..1]
  // r      is in range [R0..R1]
  r = R0 + abc[0] * (R1 - R0);

  // abc[1] is in range [0..nbTrees_theta]
  // theta  is in range [0, 2 M_PI]
  theta = abc[1] / nbTrees_theta * 2 * M_PI;

  // return cartesian coordinates for plotting
  xyz[0] = r * cos(theta);
  xyz[1] = r * sin(theta);

  // abc[2] is in range [0..nbTrees_z]
  // just make it zero-centered
  xyz[2] = 0.0;

} /* p4est_geometry_cylindrical_io_X */

static void
p8est_geometry_cylindrical_io_X(p8est_geometry_t * geom,
                                p4est_topidx_t     which_tree,
                                const double       rst[3],
                                double             xyz[3])
{
  const p4est_geometry_builtin_cylindrical_io_t * cylindrical_io =
    &(reinterpret_cast<p8est_geometry_builtin_t *>(geom))->p.cylindrical_io;
  // double              x, y, z;
  double R0 = cylindrical_io->R0;
  double R1 = cylindrical_io->R1;

  int                  nbTrees_theta = cylindrical_io->nbTrees_theta;
  [[maybe_unused]] int nbTrees_z = cylindrical_io->nbTrees_z;
  double               abc[3];

  double r, theta;

  /* transform from the reference cube [0,1]^3 into logical vertex space
     using bi/trilinear transformation */
  p8est_geometry_connectivity_X(geom, which_tree, rst, abc);

  /*
   * assert that input points are in the expected range
   * Note: maybe we should remove these assert, this would allow
   * ghost quadrant at external boundary to call this routine ?
   */
  P4EST_ASSERT(cylindrical_io->type == P4EST_GEOMETRY_BUILTIN_CYLINDRICAL_IO);
  P4EST_ASSERT(0 <= which_tree && which_tree < nbTrees_theta * nbTrees_z);
  P4EST_ASSERT(abc[0] < 1.0 + SC_1000_EPS && abc[0] > 0.0 - SC_1000_EPS);
  P4EST_ASSERT(abc[1] < nbTrees_theta + SC_1000_EPS && abc[1] > 0.0 - SC_1000_EPS);

  P4EST_ASSERT(abc[2] < nbTrees_z + SC_1000_EPS && abc[2] > 0.0 - SC_1000_EPS);

  // abc[0] is in range [0..1]
  // r      is in range [R0..R1]
  r = R0 + abc[0] * (R1 - R0);

  // abc[1] is in range [0..nbTrees_theta]
  // theta  is in range [0, 2 M_PI]
  theta = abc[1] / nbTrees_theta * 2 * M_PI;

  // return cartesian coordinates for plotting
  xyz[0] = r * cos(theta);
  xyz[1] = r * sin(theta);

  // abc[2] is in range [0..nbTrees_z]
  // just make it zero-centered
  xyz[2] = abc[2] - 0.5 * nbTrees_z;

} /* p8est_geometry_cylindrical_io_X */

p4est_geometry_t *
p4est_geometry_new_cylindrical_io(p4est_connectivity_t * conn,
                                  double                 R0,
                                  double                 R1,
                                  int                    nbTrees_theta,
                                  int                    nbTrees_z)
{

  p4est_geometry_builtin_t *                builtin;
  p4est_geometry_builtin_cylindrical_io_t * cylindrical_io;

  builtin = P4EST_ALLOC_ZERO(p4est_geometry_builtin_t, 1);

  cylindrical_io = &builtin->p.cylindrical_io;
  cylindrical_io->type = P4EST_GEOMETRY_BUILTIN_CYLINDRICAL_IO;
  cylindrical_io->R0 = R0;
  cylindrical_io->R1 = R1;
  cylindrical_io->nbTrees_theta = nbTrees_theta;
  cylindrical_io->nbTrees_z = nbTrees_z;

  builtin->geom.name = "p4est_cylindrical_io";
  builtin->geom.user = conn;
  builtin->geom.X = p4est_geometry_cylindrical_io_X;

  return reinterpret_cast<p4est_geometry_t *>(builtin);

} /* p4est_geometry_new_cylindrical_io */

p8est_geometry_t *
p8est_geometry_new_cylindrical_io(p8est_connectivity_t * conn,
                                  double                 R0,
                                  double                 R1,
                                  int                    nbTrees_theta,
                                  int                    nbTrees_z)
{

  p8est_geometry_builtin_t *                builtin;
  p4est_geometry_builtin_cylindrical_io_t * cylindrical_io;

  builtin = P4EST_ALLOC_ZERO(p8est_geometry_builtin_t, 1);

  cylindrical_io = &builtin->p.cylindrical_io;
  cylindrical_io->type = P4EST_GEOMETRY_BUILTIN_CYLINDRICAL_IO;
  cylindrical_io->R0 = R0;
  cylindrical_io->R1 = R1;
  cylindrical_io->nbTrees_theta = nbTrees_theta;
  cylindrical_io->nbTrees_z = nbTrees_z;

  builtin->geom.name = "p4est_cylindrical_io";
  builtin->geom.user = conn;
  builtin->geom.X = p8est_geometry_cylindrical_io_X;

  return reinterpret_cast<p8est_geometry_t *>(builtin);

} /* p8est_geometry_new_cylindrical_io */


p4est_geometry_t *
geometry_2d_new_byname(const char *                name,
                       p4est_connectivity_t *      conn,
                       const kalypsso::ConfigMap & cfg)
{
  p4est_geometry_t * geom = nullptr;

  if (strcmp(name, "cartesian") == 0 or strcmp(name, "no_geometry") == 0)
  {
    geom = nullptr;
  }

  else if (strcmp(name, "cylindrical_compute") == 0 || strcmp(name, "cylindrical_io") == 0)
  {

    double R0, R1;
    R0 = cfg.getDouble("p4est_geometry", "cylindrical.rMin", 0.1);
    R1 = cfg.getDouble("p4est_geometry", "cylindrical.rMax", 1.0);
    int nbTrees_theta, nbTrees_z;
    nbTrees_theta = cfg.getInteger("p4est_geometry", "cylindrical.nbTrees_theta", 8);
    nbTrees_z = cfg.getInteger("p4est_geometry", "cylindrical.nbTrees_z", 1);

    if (strcmp(name, "cylindrical_compute") == 0)
      geom = p4est_geometry_new_cylindrical_compute(conn, R0, R1, nbTrees_theta, nbTrees_z);
    else
      geom = p4est_geometry_new_cylindrical_io(conn, R0, R1, nbTrees_theta, nbTrees_z);
  }

  else if (strcmp(name, "disk2d") == 0)
  {
    double R0, R1;
    R0 = cfg.getDouble("p4est_geometry", "disk2d.R0", 0.5);
    R1 = cfg.getDouble("p4est_geometry", "disk2d.R1", 1.0);

    geom = p4est_geometry_new_disk2d(conn, R0, R1);
  }

  else if (strcmp(name, "shell2d") == 0)
  {
    double R1, R2;
    R1 = cfg.getDouble("p4est_geometry", "shell2d.R1", 0.5);
    R2 = cfg.getDouble("p4est_geometry", "shell2d.R2", 4.0);

    geom = p4est_geometry_new_shell2d(conn, R2, R1);
  }

  else if (strcmp(name, "icosahedron") == 0)
  {
    double a;
    a = cfg.getDouble("p4est_geometry", "icosahedron.a", 1.0);

    geom = p4est_geometry_new_icosahedron(conn, a);
  }

  else
  {

    P4EST_GLOBAL_LERRORF("###### Unrecognized geometry \"%s\".\n", name);
    P4EST_GLOBAL_LERROR("###### Defaulting to nullptr.\n");
    geom = nullptr;
  }

  return geom;

} // geometry_2d_new_byname

p8est_geometry_t *
geometry_3d_new_byname(const char *                name,
                       p8est_connectivity_t *      conn,
                       const kalypsso::ConfigMap & cfg)
{
  p8est_geometry_t * geom = nullptr;

  if (strcmp(name, "cartesian") == 0 or strcmp(name, "no_geometry") == 0)
  {
    geom = nullptr;
  }

  else if (strcmp(name, "sphere") == 0)
  {

    double R0, R1, R2;
    R2 = cfg.getDouble("p4est_geometry", "sphere.R2", 1.0);
    R1 = cfg.getDouble("p4est_geometry", "sphere.R1", 0.7);
    R0 = cfg.getDouble("p4est_geometry", "sphere.R0", 0.5);

    geom = p8est_geometry_new_sphere(conn, R2, R1, R0);
  }

  else if (strcmp(name, "shell") == 0)
  {

    double R1, R2;
    R2 = cfg.getDouble("p4est_geometry", "shell.R2", 1.0);
    R1 = cfg.getDouble("p4est_geometry", "shell.R1", 0.44);

    geom = p8est_geometry_new_shell(conn, R2, R1);
  }

  else if (strcmp(name, "cylindrical_compute") == 0 || strcmp(name, "cylindrical_io") == 0)
  {

    double R0, R1;
    R0 = cfg.getDouble("p4est_geometry", "cylindrical.rMin", 0.1);
    R1 = cfg.getDouble("p4est_geometry", "cylindrical.rMax", 1.0);
    int nbTrees_theta, nbTrees_z;
    nbTrees_theta = cfg.getInteger("p4est_geometry", "cylindrical.nbTrees_theta", 8);
    nbTrees_z = cfg.getInteger("p4est_geometry", "cylindrical.nbTrees_z", 1);

    if (strcmp(name, "cylindrical_compute") == 0)
      geom = p8est_geometry_new_cylindrical_compute(conn, R0, R1, nbTrees_theta, nbTrees_z);
    else
      geom = p8est_geometry_new_cylindrical_io(conn, R0, R1, nbTrees_theta, nbTrees_z);
  }

  else
  {

    P4EST_GLOBAL_LERRORF("###### Unrecognized geometry \"%s\".\n", name);
    P4EST_GLOBAL_LERROR("###### Defaulting to nullptr.\n");
    geom = nullptr;
  }

  return geom;

} // geometry_3d_new_byname
