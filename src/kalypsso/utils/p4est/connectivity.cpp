// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * @file connectivity.cpp
 * @date 17 octobre 2020
 * @author pkestene
 * @note
 *
 * This file is part of the Kalypsso software project.
 *
 */
#include "connectivity.h"

#include <kalypsso/utils/config/ConfigMap.h>

p4est_connectivity_t *
p4est_connectivity_new_two(connectivity_periodic_t is_periodic_x,
                           connectivity_periodic_t is_periodic_y)
{
  /*
   * p4est_connectivity_new_brick constructs a set of m x n x p trees that
   * are sitting side by side and can be periodic in either x, y or z.
   */
  return p4est_connectivity_new_brick(2, 1, is_periodic_x, is_periodic_y);
}

p8est_connectivity_t *
p8est_connectivity_new_two(connectivity_periodic_t is_periodic_x,
                           connectivity_periodic_t is_periodic_y,
                           connectivity_periodic_t is_periodic_z)
{
  /*
   * p4est_connectivity_new_brick constructs a set of m x n x p trees that
   * are sitting side by side and can be periodic in either x, y or z.
   */
  return p8est_connectivity_new_brick(2, 1, 1, is_periodic_x, is_periodic_y, is_periodic_z);
}

p4est_connectivity_t *
p4est_connectivity_new_two_simple(void)
{
  return p4est_connectivity_new_brick(2, 1, 0, 0);
}

p8est_connectivity_t *
p8est_connectivity_new_two_simple(void)
{
  return p8est_connectivity_new_brick(2, 1, 1, 0, 0, 0);
}

p4est_connectivity_t *
p4est_connectivity_new_shock_tube(void)
{
  return p4est_connectivity_new_brick(10, 1, 0, 1);
}

p8est_connectivity_t *
p8est_connectivity_new_shock_tube(void)
{
  return p8est_connectivity_new_brick(10, 1, 1, 0, 1, 0);
}

p4est_connectivity_t *
p4est_connectivity_new_tetris(void)
{

  // clang-format off
/* *INDENT-OFF* */
  const p4est_topidx_t num_vertices = 12;
  const p4est_topidx_t num_trees    = 5;
  const p4est_topidx_t num_corners  = 2;
  const double         vertices[12 * 3] = {
    0,  0,  0, /* vertex  0 */
    1,  0,  0, /* vertex  1 */
    2,  0,  0, /* vertex  2 */
    3,  0,  0, /* vertex  3 */
    4,  0,  0, /* vertex  4 */
    0, -1,  0, /* vertex  5 */
    1, -1,  0, /* vertex  6 */
    2, -1,  0, /* vertex  7 */
    3, -1,  0, /* vertex  8 */
    4, -1,  0, /* vertex  9 */
    1, -2,  0, /* vertex 10 */
    2, -2,  0, /* vertex 11 */
  };
  const p4est_topidx_t tree_to_vertex[5 * 4] = {
    5,  6,  0,  1, /* tree 0 */
    6,  7,  1,  2, /* tree 1 */
    7,  8,  2,  3, /* tree 2 */
    8,  9,  3,  4, /* tree 3 */
    10,11,  6,  7, /* tree 4 */
  };
  const p4est_topidx_t tree_to_tree[5 * 4] = {
    0,  1,  0,  0, /* tree 0 */
    0,  2,  4,  1, /* tree 1 */
    1,  3,  2,  2, /* tree 2 */
    2,  3,  3,  3, /* tree 3 */
    4,  4,  4,  1, /* tree 4 */
  };
  const int8_t tree_to_face[5 * 4] = {
    0,  0,  2,  3, /* tree 0 */
    1,  0,  3,  3, /* tree 1 */
    1,  0,  2,  3, /* tree 2 */
    1,  1,  2,  3, /* tree 3 */
    0,  1,  2,  2, /* tree 4 */
  };
  const p4est_topidx_t tree_to_corner[5 * 4] = {
    -1,  0, -1, -1, /* tree 0 */
    0 ,  1, -1, -1, /* tree 1 */
    1 , -1, -1, -1, /* tree 2 */
    -1, -1, -1, -1, /* tree 3 */
    -1, -1,  0,  1, /* tree 4 */
  };
  const p4est_topidx_t ctt_offset[2+1] = {
    0,3,6,
  };
  const p4est_topidx_t corner_to_tree[6] = {
    0,1,4,  /* corner 0 (i.e vertex 6) */
    1,2,4,  /* corner 1 (i.e vertex 7) */
  };

  /* a given corner belong to multiple trees;
   for each tree, we report the index identifying the vertex location
   in the tree_to_vertex.
  e.g. here :
  - corner 0 is vertex 6
  - corner 0 is vertex 7
  For each entry in corner_to_tree, we report the location of the vertex in
  tree_to_vertex
  */
  const int8_t corner_to_corner[6] = {
    1, 0, 2, /* corner 0 (i.e vertex 6) */
    1, 0, 3, /* corner 1 (i.e vertex 7) */
  };

/* *INDENT-ON* */
  // clang-format on

  p4est_connectivity_t * conn = p4est_connectivity_new_copy(num_vertices,
                                                            num_trees,
                                                            num_corners,
                                                            vertices,
                                                            tree_to_vertex,
                                                            tree_to_tree,
                                                            tree_to_face,
                                                            tree_to_corner,
                                                            ctt_offset,
                                                            corner_to_tree,
                                                            corner_to_corner);

  P4EST_ASSERT(p4est_connectivity_is_valid(conn));

  return conn;

} /* p4est_connectivity_new_tetris */

p4est_connectivity_t *
p4est_connectivity_new_ring(int    num_trees_radial,
                            int    num_trees_orthoradial,
                            double rMin,
                            double rMax)
{
  p4est_topidx_t num_vertices = (num_trees_radial + 1) * (num_trees_orthoradial);
  p4est_topidx_t num_trees = num_trees_radial * num_trees_orthoradial;
  p4est_topidx_t num_ctt = 0;
  double *       vertices = new double[num_vertices * 3];

  p4est_topidx_t * tree_to_vertex = new p4est_topidx_t[num_trees * 4];
  p4est_topidx_t * tree_to_tree = new p4est_topidx_t[num_trees * 4];
  int8_t *         tree_to_face = new int8_t[num_trees * 4];

  int iVertex = 0;
  int iRadial;
  int iOrthoradial;
  int iTree;

  // fill vertices array
  for (iRadial = 0; iRadial < num_trees_radial + 1; iRadial++)
  {

    double radius = rMin + iRadial * (rMax - rMin) / num_trees_radial;

    for (iOrthoradial = 0; iOrthoradial < num_trees_orthoradial; iOrthoradial++)
    {
      vertices[3 * iVertex + 0] = radius * cos(2 * M_PI / num_trees_orthoradial * iOrthoradial);
      vertices[3 * iVertex + 1] = radius * sin(2 * M_PI / num_trees_orthoradial * iOrthoradial);
      vertices[3 * iVertex + 2] = 0;

      iVertex++;
    }
  }

  // fill tree to vertex
  iTree = 0;
  for (iRadial = 0; iRadial < num_trees_radial; iRadial++)
  {
    for (iOrthoradial = 0; iOrthoradial < num_trees_orthoradial; iOrthoradial++)
    {

      p4est_topidx_t left_corner_vertex = iOrthoradial + num_trees_orthoradial * iRadial;

      tree_to_vertex[4 * iTree + 0] = left_corner_vertex;
      tree_to_vertex[4 * iTree + 1] = left_corner_vertex + num_trees_orthoradial;
      tree_to_vertex[4 * iTree + 2] = left_corner_vertex + 1;
      tree_to_vertex[4 * iTree + 3] = left_corner_vertex + 1 + num_trees_orthoradial;

      // need to modify this when we cross iTree % num_trees_orthoradial == 0
      if (iTree % num_trees_orthoradial == num_trees_orthoradial - 1)
      {
        tree_to_vertex[4 * iTree + 2] -= num_trees_orthoradial;
        tree_to_vertex[4 * iTree + 3] -= num_trees_orthoradial;
      }

      iTree++;
    }
  }

  // fill tree to tree
  iTree = 0;
  for (iRadial = 0; iRadial < num_trees_radial; iRadial++)
  {
    for (iOrthoradial = 0; iOrthoradial < num_trees_orthoradial; iOrthoradial++)
    {

      tree_to_tree[4 * iTree + 0] =
        iTree - num_trees_orthoradial < 0 ? iTree : iTree - num_trees_orthoradial;
      tree_to_tree[4 * iTree + 1] =
        iTree + num_trees_orthoradial >= num_trees ? iTree : iTree + num_trees_orthoradial;
      tree_to_tree[4 * iTree + 2] = iTree - 1;
      tree_to_tree[4 * iTree + 3] = iTree + 1;

      // need to modify this when we cross iTree % num_trees_orthoradial == 0
      if (iTree % num_trees_orthoradial == 0)
      {
        tree_to_tree[4 * iTree + 2] += num_trees_orthoradial;
      }
      if (iTree % num_trees_orthoradial == num_trees_orthoradial - 1)
      {
        tree_to_tree[4 * iTree + 3] -= num_trees_orthoradial;
      }

      iTree++;
    }
  }

  // tree_to_face
  iTree = 0;
  for (iRadial = 0; iRadial < num_trees_radial; iRadial++)
  {
    for (iOrthoradial = 0; iOrthoradial < num_trees_orthoradial; iOrthoradial++)
    {
      tree_to_face[4 * iTree + 0] = 1;
      tree_to_face[4 * iTree + 1] = 0;
      tree_to_face[4 * iTree + 2] = 3;
      tree_to_face[4 * iTree + 3] = 2;

      // need to modify this when tree is on edge (inner or outer)
      if (iTree < num_trees_orthoradial)
      { // inner border
        tree_to_face[4 * iTree + 0] = 0;
      }
      if (iTree + num_trees_orthoradial >= num_trees)
      { // outer border
        tree_to_face[4 * iTree + 1] = 1;
      }

      iTree++;
    }
  }

  p4est_connectivity_t * conn = p4est_connectivity_new_copy(num_vertices,
                                                            num_trees,
                                                            0,
                                                            vertices,
                                                            tree_to_vertex,
                                                            tree_to_tree,
                                                            tree_to_face,
                                                            NULL,
                                                            &num_ctt,
                                                            NULL,
                                                            NULL);

  P4EST_ASSERT(p4est_connectivity_is_valid(conn));

  delete[] vertices;
  delete[] tree_to_vertex;
  delete[] tree_to_tree;
  delete[] tree_to_face;

  return conn;

} /* p4est_connectivity_new_ring */

p4est_connectivity_t *
p4est_connectivity_new_forward_facing_step_small(void)
{

  const p4est_topidx_t num_vertices = 21; /* 6*3 + 3 */
  const p4est_topidx_t num_trees = 12;    /* 5*2 + 2 */
  const p4est_topidx_t num_corners = 6;   /* 4 + 2 */
  double               vertices[21 * 3];
  p4est_topidx_t       tree_to_vertex[12 * 4];
  p4est_topidx_t       tree_to_tree[12 * 4];
  int8_t               tree_to_face[12 * 4];
  p4est_topidx_t       tree_to_corner[12 * 4];
  p4est_topidx_t       ctt_offset[6 + 1];
  p4est_topidx_t       corner_to_tree[23];   /* 5 corners*4trees + 1 corner*3trees */
  int8_t               corner_to_corner[23]; /* 5 corners*4trees + 1 corner*3trees */

  int i, j, iVertex = 0;

  double dx = 0.2;
  double dy = 0.2;

  /* fill vertices the 18 first vertices */
  for (j = 0; j <= 2; j++)
  {
    for (i = 0; i <= 5; i++)
    {
      vertices[3 * iVertex + 0] = i * dx;       /* */
      vertices[3 * iVertex + 1] = 0.6 - j * dy; /* */
      vertices[3 * iVertex + 2] = 0.0;          /* */
      iVertex++;
    }
  }

  /* and the last 3 */
  vertices[3 * 18 + 0] = 0.0;
  vertices[3 * 18 + 1] = 0.0;
  vertices[3 * 18 + 2] = 0.0;

  vertices[3 * 19 + 0] = 0.0 + dx;
  vertices[3 * 19 + 1] = 0.0;
  vertices[3 * 19 + 2] = 0.0;

  vertices[3 * 20 + 0] = 0.0 + 2 * dx;
  vertices[3 * 20 + 1] = 0.0;
  vertices[3 * 20 + 2] = 0.0;

  /*
   * tree to vertex
   */
  {
    int iTree = 0;
    int v0 = 6;
    int v1 = 7;
    int v2 = 0;
    int v3 = 1;

    for (j = 0; j < 2; j++)
    {
      for (i = 0; i < 5; i++)
      {
        iTree = j * 5 + i;

        tree_to_vertex[4 * iTree + 0] = v0 + i + j * 6;
        tree_to_vertex[4 * iTree + 1] = v1 + i + j * 6;
        tree_to_vertex[4 * iTree + 2] = v2 + i + j * 6;
        tree_to_vertex[4 * iTree + 3] = v3 + i + j * 6;

        /* printf("ttv %d | %d %d %d %d\n",iTree, */
        /*        tree_to_vertex[4*iTree + 0], */
        /*        tree_to_vertex[4*iTree + 1], */
        /*        tree_to_vertex[4*iTree + 2], */
        /*        tree_to_vertex[4*iTree + 3]); */
      }
    }

    tree_to_vertex[4 * 10 + 0] = 18;
    tree_to_vertex[4 * 10 + 1] = 19;
    tree_to_vertex[4 * 10 + 2] = 12;
    tree_to_vertex[4 * 10 + 3] = 13;

    tree_to_vertex[4 * 11 + 0] = 19;
    tree_to_vertex[4 * 11 + 1] = 20;
    tree_to_vertex[4 * 11 + 2] = 13;
    tree_to_vertex[4 * 11 + 3] = 14;
  }

  /*
   * tree to tree
   */
  {
    int iTree = 0;

    for (j = 0; j < 2; j++)
    {
      for (i = 0; i < 5; i++)
      {
        iTree = j * 5 + i;

        tree_to_tree[4 * iTree + 0] = iTree - 1; /* left */
        tree_to_tree[4 * iTree + 1] = iTree + 1; /* right */
        tree_to_tree[4 * iTree + 2] = iTree + 5; /* below */
        tree_to_tree[4 * iTree + 3] = iTree - 5; /* above */

        if (i == 0)
          tree_to_tree[4 * iTree + 0] = iTree;
        if (i == 4)
          tree_to_tree[4 * iTree + 1] = iTree;
        if (j == 0)
          tree_to_tree[4 * iTree + 3] = iTree;
        if (j == 1 && i >= 2)
          tree_to_tree[4 * iTree + 2] = iTree;

        /* printf("ttt %d | %d %d %d %d\n",iTree, */
        /*        tree_to_tree[4*iTree + 0], */
        /*        tree_to_tree[4*iTree + 1], */
        /*        tree_to_tree[4*iTree + 2], */
        /*        tree_to_tree[4*iTree + 3]); */
      }
    }

    tree_to_tree[4 * 10 + 0] = 10;
    tree_to_tree[4 * 10 + 1] = 11;
    tree_to_tree[4 * 10 + 2] = 10;
    tree_to_tree[4 * 10 + 3] = 10 - 5;

    tree_to_tree[4 * 11 + 0] = 10;
    tree_to_tree[4 * 11 + 1] = 11;
    tree_to_tree[4 * 11 + 2] = 11;
    tree_to_tree[4 * 11 + 3] = 11 - 5;
  }

  /*
   * tree to face
   */
  {
    int iTree = 0;

    for (j = 0; j < 2; j++)
    {
      for (i = 0; i < 5; i++)
      {
        iTree = j * 5 + i;

        tree_to_face[4 * iTree + 0] = 1; /* left */
        tree_to_face[4 * iTree + 1] = 0; /* right */
        tree_to_face[4 * iTree + 2] = 3; /* bottow */
        tree_to_face[4 * iTree + 3] = 2; /* up */

        if (i == 0)
          tree_to_face[4 * iTree + 0] = 0;
        if (i == 4)
          tree_to_face[4 * iTree + 1] = 1;
        if (j == 0)
          tree_to_face[4 * iTree + 3] = 3;
        if (j == 1 && i >= 2)
          tree_to_face[4 * iTree + 2] = 2;

        /* printf("ttf %d | %d %d %d %d\n",iTree, */
        /*        tree_to_face[4*iTree + 0], */
        /*        tree_to_face[4*iTree + 1], */
        /*        tree_to_face[4*iTree + 2], */
        /*        tree_to_face[4*iTree + 3]); */
      }
    }

    tree_to_face[4 * 10 + 0] = 0; /* left */
    tree_to_face[4 * 10 + 1] = 0; /* right */
    tree_to_face[4 * 10 + 2] = 2; /* bottow */
    tree_to_face[4 * 10 + 3] = 2; /* up */

    tree_to_face[4 * 11 + 0] = 1; /* left */
    tree_to_face[4 * 11 + 1] = 1; /* right */
    tree_to_face[4 * 11 + 2] = 2; /* bottow */
    tree_to_face[4 * 11 + 3] = 2; /* up */
  }

  /*
   * tree to corner
   */
  {
    int iTree = 0;
    for (j = 0; j < 2; j++)
    {
      for (i = 0; i < 5; i++)
      {
        iTree = j * 5 + i;
        tree_to_corner[4 * iTree + 0] = i - 1 + 4 * j;
        tree_to_corner[4 * iTree + 1] = i + 4 * j;
        tree_to_corner[4 * iTree + 2] = i - 1 + 4 * (j - 1);
        tree_to_corner[4 * iTree + 3] = i + 4 * (j - 1);

        if (i == 0)
        {
          tree_to_corner[4 * iTree + 0] = -1;
          tree_to_corner[4 * iTree + 2] = -1;
        }
        if (i == 4)
        {
          tree_to_corner[4 * iTree + 1] = -1;
          tree_to_corner[4 * iTree + 3] = -1;
        }
        if (j == 0)
        {
          tree_to_corner[4 * iTree + 2] = -1;
          tree_to_corner[4 * iTree + 3] = -1;
        }
        if (j == 1 && i == 2)
        {
          tree_to_corner[4 * iTree + 0] = 5;
          tree_to_corner[4 * iTree + 1] = -1;
        }
        if (j == 1 && i > 2)
        {
          tree_to_corner[4 * iTree + 0] = -1;
          tree_to_corner[4 * iTree + 1] = -1;
        }

        /* printf("ttc %d | %d %d %d %d\n",iTree, */
        /*        tree_to_corner[4*iTree + 0], */
        /*        tree_to_corner[4*iTree + 1], */
        /*        tree_to_corner[4*iTree + 2], */
        /*        tree_to_corner[4*iTree + 3]); */
      }
    }

    tree_to_corner[4 * 10 + 0] = -1;
    tree_to_corner[4 * 10 + 1] = -1;
    tree_to_corner[4 * 10 + 2] = -1;
    tree_to_corner[4 * 10 + 3] = 4;

    tree_to_corner[4 * 11 + 0] = -1;
    tree_to_corner[4 * 11 + 1] = -1;
    tree_to_corner[4 * 11 + 2] = 4;
    tree_to_corner[4 * 11 + 3] = 5;
  }

  /*
   * corner to tree
   */
  {
    int iCorner, iTree;

    for (iCorner = 0; iCorner < 6; iCorner++)
      ctt_offset[iCorner] = iCorner * 4;
    ctt_offset[6] = 5 * 4 + 3;

    for (j = 0; j < 1; j++)
    {
      for (i = 0; i < 4; i++)
      {
        iCorner = j * 4 + i;
        iTree = j * 5 + i;

        corner_to_tree[4 * iCorner + 0] = iTree;
        corner_to_tree[4 * iCorner + 1] = iTree + 1;
        corner_to_tree[4 * iCorner + 2] = iTree + 5;
        corner_to_tree[4 * iCorner + 3] = iTree + 6;

        /* printf("ctt %d | %d %d %d %d\n",iCorner, */
        /*        corner_to_tree[4*iCorner + 0], */
        /*        corner_to_tree[4*iCorner + 1], */
        /*        corner_to_tree[4*iCorner + 2], */
        /*        corner_to_tree[4*iCorner + 3]); */
      }
    }

    corner_to_tree[4 * 4 + 0] = 5;
    corner_to_tree[4 * 4 + 1] = 6;
    corner_to_tree[4 * 4 + 2] = 10;
    corner_to_tree[4 * 4 + 3] = 11;

    corner_to_tree[4 * 5 + 0] = 6;
    corner_to_tree[4 * 5 + 1] = 7;
    corner_to_tree[4 * 5 + 2] = 11;
  }

  /*
   * corner to corner
   */
  {
    int iCorner;

    for (j = 0; j < 1; j++)
    {
      for (i = 0; i < 4; i++)
      {
        iCorner = j * 4 + i;

        corner_to_corner[4 * iCorner + 0] = 1;
        corner_to_corner[4 * iCorner + 1] = 0;
        corner_to_corner[4 * iCorner + 2] = 3;
        corner_to_corner[4 * iCorner + 3] = 2;
      }
    }

    corner_to_corner[4 * 4 + 0] = 1;
    corner_to_corner[4 * 4 + 1] = 0;
    corner_to_corner[4 * 4 + 2] = 3;
    corner_to_corner[4 * 4 + 3] = 2;

    corner_to_corner[4 * 5 + 0] = 1;
    corner_to_corner[4 * 5 + 1] = 0;
    corner_to_corner[4 * 5 + 2] = 3;
  }

  p4est_connectivity_t * conn = p4est_connectivity_new_copy(num_vertices,
                                                            num_trees,
                                                            num_corners,
                                                            vertices,
                                                            tree_to_vertex,
                                                            tree_to_tree,
                                                            tree_to_face,
                                                            tree_to_corner,
                                                            ctt_offset,
                                                            corner_to_tree,
                                                            corner_to_corner);


  connectivity_print(conn);

  P4EST_GLOBAL_INFOF("Is connectivity ok : %d\n", p4est_connectivity_is_valid(conn));

  P4EST_ASSERT(p4est_connectivity_is_valid(conn));

  return conn;

} /* p4est_connectivity_new_forward_facing_step_small */

p4est_connectivity_t *
p4est_connectivity_new_forward_facing_step(void)
{

  const p4est_topidx_t num_vertices = 84; /* 16*5 + 4 */
  const p4est_topidx_t num_trees = 63;    /* 15*4 + 3 */
  const p4est_topidx_t num_corners = 45;  /* 14*3 + 3 */
  double               vertices[84 * 3];
  p4est_topidx_t       tree_to_vertex[63 * 4];
  p4est_topidx_t       tree_to_tree[63 * 4];
  int8_t               tree_to_face[63 * 4];
  p4est_topidx_t       tree_to_corner[63 * 4];
  p4est_topidx_t       ctt_offset[45 + 1];
  p4est_topidx_t       corner_to_tree[179];   /* 44 corners*4trees + 1 corner*3tree */
  int8_t               corner_to_corner[179]; /* 44 corners*4trees + 1 corner*3tree */

  int i, j, iVertex = 0;

  double dx = 0.2;
  double dy = 0.2;

  /* fill vertices the 80 first vertices */
  for (j = 0; j <= 4; j++)
  {
    for (i = 0; i <= 15; i++)
    {
      vertices[3 * iVertex + 0] = i * dx;       /* */
      vertices[3 * iVertex + 1] = 1.0 - j * dy; /* */
      vertices[3 * iVertex + 2] = 0.0;          /* */
      iVertex++;
    }
  }

  /* and the last 4 */
  vertices[3 * 80 + 0] = 0.0;
  vertices[3 * 80 + 1] = 0.0;
  vertices[3 * 80 + 2] = 0.0;

  vertices[3 * 81 + 0] = 0.2;
  vertices[3 * 81 + 1] = 0.0;
  vertices[3 * 81 + 2] = 0.0;

  vertices[3 * 82 + 0] = 0.4;
  vertices[3 * 82 + 1] = 0.0;
  vertices[3 * 82 + 2] = 0.0;

  vertices[3 * 83 + 0] = 0.6;
  vertices[3 * 83 + 1] = 0.0;
  vertices[3 * 83 + 2] = 0.0;

  /*
   * tree to vertex
   */
  {
    int iTree = 0;
    int v0 = 16;
    int v1 = 17;
    int v2 = 0;
    int v3 = 1;

    for (j = 0; j < 4; j++)
    {
      for (i = 0; i < 15; i++)
      {
        iTree = j * 15 + i;

        tree_to_vertex[4 * iTree + 0] = v0 + i + j * 16;
        tree_to_vertex[4 * iTree + 1] = v1 + i + j * 16;
        tree_to_vertex[4 * iTree + 2] = v2 + i + j * 16;
        tree_to_vertex[4 * iTree + 3] = v3 + i + j * 16;

        /* printf("ttv %d | %d %d %d %d\n",iTree, */
        /*        tree_to_vertex[4*iTree + 0], */
        /*        tree_to_vertex[4*iTree + 1], */
        /*        tree_to_vertex[4*iTree + 2], */
        /*        tree_to_vertex[4*iTree + 3]); */
      }
    }

    tree_to_vertex[4 * 60 + 0] = 80;
    tree_to_vertex[4 * 60 + 1] = 81;
    tree_to_vertex[4 * 60 + 2] = 64;
    tree_to_vertex[4 * 60 + 3] = 65;

    tree_to_vertex[4 * 61 + 0] = 81;
    tree_to_vertex[4 * 61 + 1] = 82;
    tree_to_vertex[4 * 61 + 2] = 65;
    tree_to_vertex[4 * 61 + 3] = 66;

    tree_to_vertex[4 * 62 + 0] = 82;
    tree_to_vertex[4 * 62 + 1] = 83;
    tree_to_vertex[4 * 62 + 2] = 66;
    tree_to_vertex[4 * 62 + 3] = 67;
  }

  /*
   * tree to tree
   */
  {
    int iTree = 0;

    for (j = 0; j < 4; j++)
    {
      for (i = 0; i < 15; i++)
      {
        iTree = j * 15 + i;

        tree_to_tree[4 * iTree + 0] = iTree - 1;  /* left */
        tree_to_tree[4 * iTree + 1] = iTree + 1;  /* right */
        tree_to_tree[4 * iTree + 2] = iTree + 15; /* below */
        tree_to_tree[4 * iTree + 3] = iTree - 15; /* above */

        if (i == 0)
          tree_to_tree[4 * iTree + 0] = iTree;
        if (i == 14)
          tree_to_tree[4 * iTree + 1] = iTree;
        if (j == 0)
          tree_to_tree[4 * iTree + 3] = iTree;
        if (j == 3 && i >= 3)
          tree_to_tree[4 * iTree + 2] = iTree;

        /* printf("ttt %d | %d %d %d %d\n",iTree, */
        /*        tree_to_tree[4*iTree + 0], */
        /*        tree_to_tree[4*iTree + 1], */
        /*        tree_to_tree[4*iTree + 2], */
        /*        tree_to_tree[4*iTree + 3]);	 */
      }
    }

    tree_to_tree[4 * 60 + 0] = 60;
    tree_to_tree[4 * 60 + 1] = 61;
    tree_to_tree[4 * 60 + 2] = 60;
    tree_to_tree[4 * 60 + 3] = 60 - 15;

    tree_to_tree[4 * 61 + 0] = 60;
    tree_to_tree[4 * 61 + 1] = 62;
    tree_to_tree[4 * 61 + 2] = 61;
    tree_to_tree[4 * 61 + 3] = 61 - 15;

    tree_to_tree[4 * 62 + 0] = 61;
    tree_to_tree[4 * 62 + 1] = 62;
    tree_to_tree[4 * 62 + 2] = 62;
    tree_to_tree[4 * 62 + 3] = 62 - 15;
  }

  /*
   * tree to face
   */
  {
    int iTree = 0;

    for (j = 0; j < 4; j++)
    {
      for (i = 0; i < 15; i++)
      {
        iTree = j * 15 + i;

        tree_to_face[4 * iTree + 0] = 1; /* left */
        tree_to_face[4 * iTree + 1] = 0; /* right */
        tree_to_face[4 * iTree + 2] = 3; /* bottow */
        tree_to_face[4 * iTree + 3] = 2; /* up */

        if (i == 0)
          tree_to_face[4 * iTree + 0] = 0;
        if (i == 14)
          tree_to_face[4 * iTree + 1] = 1;
        if (j == 0)
          tree_to_face[4 * iTree + 3] = 3;
        if (j == 3 && i >= 3)
          tree_to_face[4 * iTree + 2] = 2;

        /* printf("ttf %d | %d %d %d %d\n",iTree, */
        /*        tree_to_face[4*iTree + 0], */
        /*        tree_to_face[4*iTree + 1], */
        /*        tree_to_face[4*iTree + 2], */
        /*        tree_to_face[4*iTree + 3]); */
      }
    }

    tree_to_face[4 * 60 + 0] = 0; /* left */
    tree_to_face[4 * 60 + 1] = 0; /* right */
    tree_to_face[4 * 60 + 2] = 2; /* bottow */
    tree_to_face[4 * 60 + 3] = 2; /* up */

    tree_to_face[4 * 61 + 0] = 1; /* left */
    tree_to_face[4 * 61 + 1] = 0; /* right */
    tree_to_face[4 * 61 + 2] = 2; /* bottow */
    tree_to_face[4 * 61 + 3] = 2; /* up */
    tree_to_face[4 * 62 + 0] = 1; /* left */
    tree_to_face[4 * 62 + 1] = 1; /* right */
    tree_to_face[4 * 62 + 2] = 2; /* bottow */
    tree_to_face[4 * 62 + 3] = 2; /* up */
  }

  /*
   * tree to corner
   */
  {
    int iTree = 0;

    for (j = 0; j < 4; j++)
    {
      for (i = 0; i < 15; i++)
      {
        iTree = j * 15 + i;

        tree_to_corner[4 * iTree + 0] = i - 1 + 14 * j;
        tree_to_corner[4 * iTree + 1] = i + 14 * j;
        tree_to_corner[4 * iTree + 2] = i - 1 + 14 * (j - 1);
        tree_to_corner[4 * iTree + 3] = i + 14 * (j - 1);

        if (i == 0)
        {
          tree_to_corner[4 * iTree + 0] = -1;
          tree_to_corner[4 * iTree + 2] = -1;
        }
        if (i == 14)
        {
          tree_to_corner[4 * iTree + 1] = -1;
          tree_to_corner[4 * iTree + 3] = -1;
        }
        if (j == 0)
        {
          tree_to_corner[4 * iTree + 2] = -1;
          tree_to_corner[4 * iTree + 3] = -1;
        }
        if (j == 3 && i == 3)
        {
          tree_to_corner[4 * iTree + 0] = 44;
          tree_to_corner[4 * iTree + 1] = -1;
        }
        if (j == 3 && i > 3)
        {
          tree_to_corner[4 * iTree + 0] = -1;
          tree_to_corner[4 * iTree + 1] = -1;
        }

        /* printf("ttc %d | %d %d %d %d\n",iTree, */
        /*        tree_to_corner[4*iTree + 0], */
        /*        tree_to_corner[4*iTree + 1], */
        /*        tree_to_corner[4*iTree + 2], */
        /*        tree_to_corner[4*iTree + 3]); */
      }
    }

    tree_to_corner[4 * 60 + 0] = -1;
    tree_to_corner[4 * 60 + 1] = -1;
    tree_to_corner[4 * 60 + 2] = -1;
    tree_to_corner[4 * 60 + 3] = 42;

    tree_to_corner[4 * 61 + 0] = -1;
    tree_to_corner[4 * 61 + 1] = -1;
    tree_to_corner[4 * 61 + 2] = 42;
    tree_to_corner[4 * 61 + 3] = 43;

    tree_to_corner[4 * 62 + 0] = -1;
    tree_to_corner[4 * 62 + 1] = -1;
    tree_to_corner[4 * 62 + 2] = 43;
    tree_to_corner[4 * 62 + 3] = 44;
  }

  /*
   * corner to tree
   */
  {
    int iCorner, iTree;

    for (iCorner = 0; iCorner < 45; iCorner++)
      ctt_offset[iCorner] = iCorner * 4;
    ctt_offset[45] = 44 * 4 + 3;

    for (j = 0; j < 3; j++)
    {
      for (i = 0; i < 14; i++)
      {
        iCorner = j * 14 + i;
        iTree = j * 15 + i;

        corner_to_tree[4 * iCorner + 0] = iTree;
        corner_to_tree[4 * iCorner + 1] = iTree + 1;
        corner_to_tree[4 * iCorner + 2] = iTree + 15;
        corner_to_tree[4 * iCorner + 3] = iTree + 16;

        /* printf("ctt %d | %d %d %d %d\n",iCorner, */
        /*        corner_to_tree[4*iCorner + 0], */
        /*        corner_to_tree[4*iCorner + 1], */
        /*        corner_to_tree[4*iCorner + 2], */
        /*        corner_to_tree[4*iCorner + 3]); */
      }
    }

    corner_to_tree[4 * 42 + 0] = 45;
    corner_to_tree[4 * 42 + 1] = 46;
    corner_to_tree[4 * 42 + 2] = 60;
    corner_to_tree[4 * 42 + 3] = 61;

    corner_to_tree[4 * 43 + 0] = 46;
    corner_to_tree[4 * 43 + 1] = 47;
    corner_to_tree[4 * 43 + 2] = 61;
    corner_to_tree[4 * 43 + 3] = 62;

    corner_to_tree[4 * 44 + 0] = 47;
    corner_to_tree[4 * 44 + 1] = 48;
    corner_to_tree[4 * 44 + 2] = 62;
  }

  /*
   * corner to corner
   */
  {
    int iCorner;

    for (j = 0; j < 3; j++)
    {
      for (i = 0; i < 14; i++)
      {
        iCorner = j * 14 + i;

        corner_to_corner[4 * iCorner + 0] = 1;
        corner_to_corner[4 * iCorner + 1] = 0;
        corner_to_corner[4 * iCorner + 2] = 3;
        corner_to_corner[4 * iCorner + 3] = 2;
      }
    }

    corner_to_corner[4 * 42 + 0] = 1;
    corner_to_corner[4 * 42 + 1] = 0;
    corner_to_corner[4 * 42 + 2] = 3;
    corner_to_corner[4 * 42 + 3] = 2;

    corner_to_corner[4 * 43 + 0] = 1;
    corner_to_corner[4 * 43 + 1] = 0;
    corner_to_corner[4 * 43 + 2] = 3;
    corner_to_corner[4 * 43 + 3] = 2;

    corner_to_corner[4 * 44 + 0] = 1;
    corner_to_corner[4 * 44 + 1] = 0;
    corner_to_corner[4 * 44 + 2] = 3;
  }

  p4est_connectivity_t * conn = p4est_connectivity_new_copy(num_vertices,
                                                            num_trees,
                                                            num_corners,
                                                            vertices,
                                                            tree_to_vertex,
                                                            tree_to_tree,
                                                            tree_to_face,
                                                            tree_to_corner,
                                                            ctt_offset,
                                                            corner_to_tree,
                                                            corner_to_corner);


  connectivity_print(conn);

  P4EST_GLOBAL_INFOF("Is connectivity ok : %d\n", p4est_connectivity_is_valid(conn));

  P4EST_ASSERT(p4est_connectivity_is_valid(conn));

  return conn;

} /* p4est_connectivity_new_forward_facing_step */

p4est_connectivity_t *
p4est_connectivity_new_backward_facing_step(void)
{

  const p4est_topidx_t num_vertices = 8;
  const p4est_topidx_t num_trees = 3;
  const p4est_topidx_t num_corners = 1;

  double dx = 1.3;
  double dy = 1.3;

  /* vertices */
  const double vertices[8 * 3] = {
    /* num_vertices*3 */
    0 * dx, 2 * dy, 0, /* vertex 0 */
    1 * dx, 2 * dy, 0, /* vertex 1 */
    2 * dx, 2 * dy, 0, /* vertex 2 */
    0 * dx, 1 * dy, 0, /* vertex 3 */
    1 * dx, 1 * dy, 0, /* vertex 4 */
    2 * dx, 1 * dy, 0, /* vertex 5 */
    1 * dx, 0 * dy, 0, /* vertex 6 */
    2 * dx, 0 * dy, 0, /* vertex 7 */
  };

  const p4est_topidx_t tree_to_vertex[3 * 4] = {
    3, 4, 0, 1, /* tree 0 */
    4, 5, 1, 2, /* tree 1 */
    6, 7, 4, 5, /* tree 2 */
  };

  const p4est_topidx_t tree_to_tree[3 * 4] = {
    0, 1, 0, 0, /* tree 0 */
    0, 1, 2, 1, /* tree 1 */
    2, 2, 2, 1, /* tree 2 */
  };

  const int8_t tree_to_face[3 * 4] = {
    0, 0, 2, 3, /* tree 0 */
    1, 1, 3, 3, /* tree 1 */
    0, 1, 2, 2, /* tree 2 */
  };

  const p4est_topidx_t tree_to_corner[3 * 4] = {
    -1, 0,  -1, -1, /* tree 0 */
    0,  -1, -1, -1, /* tree 1 */
    -1, -1, 0,  -1, /* tree 2 */
  };

  const p4est_topidx_t ctt_offset[1 + 1] = {
    /* num_corners=1 */
    0,
    3,
  };
  const p4est_topidx_t corner_to_tree[1 * 3] = {
    /* num_corners=1 */
    0,
    1,
    2, /* corner 0 (i.e. vertex 4) */
  };

  const int8_t corner_to_corner[3] = {
    1, 0, 2 /* corner 0 (i.e. vertex 4) */
  };

  p4est_connectivity_t * conn = p4est_connectivity_new_copy(num_vertices,
                                                            num_trees,
                                                            num_corners,
                                                            vertices,
                                                            tree_to_vertex,
                                                            tree_to_tree,
                                                            tree_to_face,
                                                            tree_to_corner,
                                                            ctt_offset,
                                                            corner_to_tree,
                                                            corner_to_corner);

  connectivity_print(conn);

  P4EST_GLOBAL_INFOF("Is connectivity ok : %d\n", p4est_connectivity_is_valid(conn));

  P4EST_ASSERT(p4est_connectivity_is_valid(conn));

  return conn;

} /* p4est_connectivity_new_backward_facing_step */


/* TODO: add tree_to_edge or however its called */
void
connectivity_print(p4est_connectivity_t * c)
{
  SC_GLOBAL_INFOF("num_vertices = %d\n", c->num_vertices);
  SC_GLOBAL_INFOF("num_trees = %d\n", c->num_trees);
  SC_GLOBAL_INFOF("num_corners = %d\n", c->num_corners);

  SC_GLOBAL_INFO("vertices:\n");
  for (int i = 0; i < c->num_vertices && c->vertices; ++i)
  {
    SC_GLOBAL_INFOF("v=%d | %g %g %g\n",
                    i,
                    c->vertices[3 * i + 0],
                    c->vertices[3 * i + 1],
                    c->vertices[3 * i + 2]);
  }

  SC_GLOBAL_INFO("tree_to_vertex:\n");
  for (int i = 0; i < c->num_trees && c->tree_to_vertex; ++i)
  {
    SC_GLOBAL_INFOF("tree=%d | %d %d %d %d\n",
                    i,
                    c->tree_to_vertex[4 * i + 0],
                    c->tree_to_vertex[4 * i + 1],
                    c->tree_to_vertex[4 * i + 2],
                    c->tree_to_vertex[4 * i + 3]);
  }

  SC_GLOBAL_INFO("tree_to_tree:\n");
  for (int i = 0; i < c->num_trees && c->tree_to_tree; ++i)
  {
    SC_GLOBAL_INFOF("tree=%d | %d %d %d %d\n",
                    i,
                    c->tree_to_tree[4 * i + 0],
                    c->tree_to_tree[4 * i + 1],
                    c->tree_to_tree[4 * i + 2],
                    c->tree_to_tree[4 * i + 3]);
  }

  SC_GLOBAL_INFO("tree_to_face:\n");
  for (int i = 0; i < c->num_trees && c->tree_to_face; ++i)
  {
    SC_GLOBAL_INFOF("tree=%d | %d %d %d %d\n",
                    i,
                    c->tree_to_face[4 * i + 0],
                    c->tree_to_face[4 * i + 1],
                    c->tree_to_face[4 * i + 2],
                    c->tree_to_face[4 * i + 3]);
  }

  SC_GLOBAL_INFO("tree_to_corner:\n");
  for (int i = 0; i < c->num_trees && c->tree_to_corner; ++i)
  {
    SC_GLOBAL_INFOF("tree=%d | %d %d %d %d\n",
                    i,
                    c->tree_to_corner[4 * i + 0],
                    c->tree_to_corner[4 * i + 1],
                    c->tree_to_corner[4 * i + 2],
                    c->tree_to_corner[4 * i + 3]);
  }

  SC_GLOBAL_INFO("ctt_offset:\n[libsc] ");
  for (int i = 0; (i < (c->num_corners + 1)) && c->ctt_offset; ++i)
  {
    printf("%d ", c->ctt_offset[i]);
  }
  printf("\n");

  SC_GLOBAL_INFO("corner_to_tree:\n");
  for (int i = 0; i < c->num_corners && c->corner_to_tree; ++i)
  {
    printf("[libsc] ");

    printf("corner=%d | ", i);
    for (int j = c->ctt_offset[i]; j < c->ctt_offset[i + 1]; ++j)
    {
      printf("%d ", c->corner_to_tree[j]);
    }
    printf("\n");
  }

  SC_GLOBAL_INFO("corner_to_corner:\n");
  for (int i = 0; i < c->num_corners && c->corner_to_corner; ++i)
  {
    printf("[libsc] ");

    printf("corner=%d | ", i);
    for (int j = c->ctt_offset[i]; j < c->ctt_offset[i + 1]; ++j)
    {
      printf("%d ", c->corner_to_corner[j]);
    }
    printf("\n");
  }
}

void
connectivity_print(p8est_connectivity_t * c)
{
  SC_GLOBAL_INFOF("num_vertices = %d\n", c->num_vertices);
  SC_GLOBAL_INFOF("num_trees = %d\n", c->num_trees);
  SC_GLOBAL_INFOF("num_corners = %d\n", c->num_corners);

  /* TODO */
}

/*
 * DONE:
 * modify interface to pass cfg (so that connectivity like ring can
 * initialize the number of trees along radial and orthoradial direction
 * by using values read from the input configuration  file.
 */
p4est_connectivity_t *
connectivity_2d_new_byname(const char * name, const kalypsso::ConfigMap & cfg)
{

  // first look at hard coded connectivity in p4est
  p4est_connectivity_t * c = p4est_connectivity_new_byname(name);

  if (c != nullptr)
  {
    return c;
  }

  // else, look into for our own connectivities
  if (strcmp(name, "shock_tube") == 0)
  {

    return p4est_connectivity_new_shock_tube();
  }
  else if (strcmp(name, "two") == 0)
  {

    int per_x = cfg.getInteger("p4est_connectivity", "periodic_x", CONNECTIVITY_PERIODIC_FALSE);
    connectivity_periodic_t periodic_x = (per_x == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    int per_y = cfg.getInteger("p4est_connectivity", "periodic_y", CONNECTIVITY_PERIODIC_TRUE);
    connectivity_periodic_t periodic_y = (per_y == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    // int per_z = cfg.getInteger("p4est_connectivity", "periodic_z", CONNECTIVITY_PERIODIC_FALSE);
    // [[maybe_unused]] connectivity_periodic_t periodic_z = (per_z == CONNECTIVITY_PERIODIC_TRUE)
    //                                                         ? CONNECTIVITY_PERIODIC_TRUE
    //                                                         : CONNECTIVITY_PERIODIC_FALSE;

    return p4est_connectivity_new_two(periodic_x, periodic_y);
  }
  else if (strcmp(name, "two_simple") == 0)
  {

    return p4est_connectivity_new_two_simple();
  }
  else if (strcmp(name, "brick") == 0)
  {

    int nbrick_x = cfg.getInteger("p4est_connectivity", "nbrick_x", 2);
    int nbrick_y = cfg.getInteger("p4est_connectivity", "nbrick_y", 2);
    //[[maybe_unused]] int nbrick_z = cfg.getInteger("p4est_connectivity", "nbrick_z", 2);

    int per_x = cfg.getInteger("p4est_connectivity", "periodic_x", CONNECTIVITY_PERIODIC_FALSE);
    connectivity_periodic_t periodic_x = (per_x == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    int per_y = cfg.getInteger("p4est_connectivity", "periodic_y", CONNECTIVITY_PERIODIC_TRUE);
    connectivity_periodic_t periodic_y = (per_y == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    // int per_z = cfg.getInteger("p4est_connectivity", "periodic_z", CONNECTIVITY_PERIODIC_FALSE);
    // [[maybe_unused]] connectivity_periodic_t periodic_z = (per_z == CONNECTIVITY_PERIODIC_TRUE)
    //                                                         ? CONNECTIVITY_PERIODIC_TRUE
    //                                                         : CONNECTIVITY_PERIODIC_FALSE;

    return p4est_connectivity_new_brick(nbrick_x, nbrick_y, periodic_x, periodic_y);
  }
  else if (strcmp(name, "tetris") == 0)
  {
    return p4est_connectivity_new_tetris();
  }
  else if (strcmp(name, "forward_facing_step_small") == 0)
  {
    return p4est_connectivity_new_forward_facing_step();
  }
  else if (strcmp(name, "forward_facing_step") == 0)
  {
    return p4est_connectivity_new_forward_facing_step();
  }
  else if (strcmp(name, "backward_facing_step") == 0)
  {
    return p4est_connectivity_new_backward_facing_step();
  }

  /* ring and shell2d only available in 2D ! */
  else if (strcmp(name, "ring") == 0)
  {

    int    num_trees_radial, num_trees_orthoradial;
    double rMin, rMax;
    num_trees_radial = cfg.getInteger("p4est_connectivity", "ring.num_trees_radial", 3);
    num_trees_orthoradial = cfg.getInteger("p4est_connectivity", "ring.num_trees_orthoradial", 10);
    rMin = cfg.getDouble("p4est_connectivity", "ring.rMin", 0.5);
    rMax = cfg.getDouble("p4est_connectivity", "ring.rMax", 4.0);
    return p4est_connectivity_new_ring(num_trees_radial, num_trees_orthoradial, rMin, rMax);
  }
  else if (strcmp(name, "disk") == 0)
  {
#if defined(KALYPSSO_CORE_USE_OLD_P4EST_API)
    return p4est_connectivity_new_disk();
#else
    int per_x = cfg.getInteger("p4est_connectivity", "periodic_x", CONNECTIVITY_PERIODIC_FALSE);
    connectivity_periodic_t periodic_x = (per_x == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    int per_y = cfg.getInteger("p4est_connectivity", "periodic_y", CONNECTIVITY_PERIODIC_TRUE);
    connectivity_periodic_t periodic_y = (per_y == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;
    return p4est_connectivity_new_disk(periodic_x, periodic_y);
#endif
  }
  else if (strcmp(name, "disk2d") == 0)
  {
    return p4est_connectivity_new_disk2d();
  }
  else if (strcmp(name, "shell2d") == 0)
  {
    return p4est_connectivity_new_shell2d();
  }
  else if (strcmp(name, "icosahedron") == 0)
  {
    return p4est_connectivity_new_icosahedron();
  }

  /*
   * cylindrical connectivity has only been tested with ramses solver.
   */

  /* cylindrical ramses */
  else if (strcmp(name, "cylindrical") == 0)
  {

    int nbrick_r = 1;
    int nbrick_theta = cfg.getInteger("geometry.cylindrical", "nbTrees_theta", 8);
    //[[maybe_unused]] int nbrick_z = cfg.getInteger("geometry.cylindrical", "nbTrees_z", 1);
    int periodic_r = 0;
    int periodic_theta = 1; /* periodic in theta */
    //[[maybe_unused]] int periodic_z = cfg.getInteger("geometry.cylindrical", "zPeriodic", 0);

    return p4est_connectivity_new_brick(nbrick_r, nbrick_theta, periodic_r, periodic_theta);
  }

  // If still not, look for a mesh file name.inp and let p4est do the
  // connectivity based on this. p4est_connectivity_read_inp will return
  // NULL if the given name is not a filename either.
  return p4est_connectivity_read_inp(name);

} // connectivity_2d_new_byname

p8est_connectivity_t *
connectivity_3d_new_byname(const char * name, const kalypsso::ConfigMap & cfg)
{

  // first look at hard coded connectivity in p4est
  p8est_connectivity_t * c = p8est_connectivity_new_byname(name);

  if (c != nullptr)
  {
    return c;
  }

  // else, look into for our own connectivities
  if (strcmp(name, "shock_tube") == 0)
  {

    return p8est_connectivity_new_shock_tube();
  }
  else if (strcmp(name, "two") == 0)
  {

    int per_x = cfg.getInteger("p4est_connectivity", "periodic_x", CONNECTIVITY_PERIODIC_FALSE);
    connectivity_periodic_t periodic_x = (per_x == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    int per_y = cfg.getInteger("p4est_connectivity", "periodic_y", CONNECTIVITY_PERIODIC_TRUE);
    connectivity_periodic_t periodic_y = (per_y == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    int per_z = cfg.getInteger("p4est_connectivity", "periodic_z", CONNECTIVITY_PERIODIC_FALSE);
    connectivity_periodic_t periodic_z = (per_z == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    return p8est_connectivity_new_two(periodic_x, periodic_y, periodic_z);
  }
  else if (strcmp(name, "two_simple") == 0)
  {

    return p8est_connectivity_new_two_simple();
  }
  else if (strcmp(name, "brick") == 0)
  {

    int nbrick_x = cfg.getInteger("p4est_connectivity", "nbrick_x", 2);
    int nbrick_y = cfg.getInteger("p4est_connectivity", "nbrick_y", 2);
    int nbrick_z = cfg.getInteger("p4est_connectivity", "nbrick_z", 2);

    int per_x = cfg.getInteger("p4est_connectivity", "periodic_x", CONNECTIVITY_PERIODIC_FALSE);
    connectivity_periodic_t periodic_x = (per_x == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    int per_y = cfg.getInteger("p4est_connectivity", "periodic_y", CONNECTIVITY_PERIODIC_TRUE);
    connectivity_periodic_t periodic_y = (per_y == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    int per_z = cfg.getInteger("p4est_connectivity", "periodic_z", CONNECTIVITY_PERIODIC_FALSE);
    connectivity_periodic_t periodic_z = (per_z == CONNECTIVITY_PERIODIC_TRUE)
                                           ? CONNECTIVITY_PERIODIC_TRUE
                                           : CONNECTIVITY_PERIODIC_FALSE;

    return p8est_connectivity_new_brick(
      nbrick_x, nbrick_y, nbrick_z, periodic_x, periodic_y, periodic_z);
  }

  /*
   * cylindrical connectivity has only been tested with ramses solver.
   */

  /* cylindrical ramses */
  else if (strcmp(name, "cylindrical") == 0)
  {

    int nbrick_r = 1;
    int nbrick_theta = cfg.getInteger("geometry.cylindrical", "nbTrees_theta", 8);
    int nbrick_z = cfg.getInteger("geometry.cylindrical", "nbTrees_z", 1);
    int periodic_r = 0;
    int periodic_theta = 1; /* periodic in theta */
    int periodic_z = cfg.getInteger("geometry.cylindrical", "zPeriodic", 0);

    return p8est_connectivity_new_brick(
      nbrick_r, nbrick_theta, nbrick_z, periodic_r, periodic_theta, periodic_z);
  }

  // If still not, look for a mesh file name.inp and let p4est do the
  // connectivity based on this. p4est_connectivity_read_inp will return
  // NULL if the given name is not a filename either.
  return p8est_connectivity_read_inp(name);

} // connectivity_3d_new_byname
