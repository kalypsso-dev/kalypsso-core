// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HDF5_IO_common.cpp
 *
 */
#include <kalypsso/core/HDF5_Xdmf_Writer.h>

#include <stdio.h>

namespace kalypsso
{

#ifdef KALYPSSO_CORE_USE_MPI
// =======================================================
// =======================================================
void
check_collective_io(const HighFive::DataTransferProps & xfer_props)
{
  auto mnccp = HighFive::MpioNoCollectiveCause(xfer_props);
  if (mnccp.getLocalCause() || mnccp.getGlobalCause())
  {
    std::cout << "The operation was successful, but couldn't use collective MPI-IO. local cause: "
              << mnccp.getLocalCause() << " global cause:" << mnccp.getGlobalCause() << std::endl;
  }
}
#endif // KALYPSSO_CORE_USE_MPI

// =======================================================
// =======================================================
template <>
void
io_fill_coordinates<2>(forest_t<2> *      forest,
                       geometry_t<2> *    geom,
                       p4est::topidx_t    which_tree,
                       quadrant_t<2> *    q,
                       float *            data,
                       std::array<int, 3> bSize)
{

  auto                  ROOT_LEN = p4est::Wrapper<2>::ROOT_LEN;
  auto                  QUADRANT_LEN = p4est::Wrapper<2>::QUADRANT_LEN;
  [[maybe_unused]] auto NB_CHILDREN = p4est::Wrapper<2>::NB_CHILDREN;
  using qcoord_t = p4est::qcoord_t;

  double xyz[3] = { 0, 0, 0 };
  double XYZ[3] = { 0, 0, 0 };

  const double   hd = (static_cast<double>(QUADRANT_LEN(q->level))) / ROOT_LEN;
  const double   h2 = 0.5 * QUADRANT_LEN(q->level) / ROOT_LEN;
  const qcoord_t h = static_cast<qcoord_t>(QUADRANT_LEN(q->level));
  const double   intsize = 1.0 / ROOT_LEN;
  const double   scale = 1;

  int & bx = bSize[0];
  int & by = bSize[1];
  int & bz = bSize[2];

  bool use_block_amr = (bx < 0 or by < 0 or bz < 0) ? false : true;

  if (use_block_amr)
  {

    double hx = hd / bx;
    double hy = hd / by;
    // double hz = hd/bz;

    double eta_z = 0;

    int inode = 0;
    for (int32_t jy = 0; jy < by + 1; ++jy)
    {
      double eta_y = intsize * q->y + hy * jy;
      for (int32_t jx = 0; jx < bx + 1; ++jx)
      {
        double eta_x = intsize * q->x + hx * jx;

        if (geom != nullptr)
        {

          xyz[0] = eta_x;
          xyz[1] = eta_y;
          xyz[2] = eta_z;
          // from logical coordinates to physical cartesian coordinates
          geom->X(geom, which_tree, xyz, XYZ);
          data[3 * inode + 0] = static_cast<float>(XYZ[0]);
          data[3 * inode + 1] = static_cast<float>(XYZ[1]);
          data[3 * inode + 2] = static_cast<float>(XYZ[2]);
        }
        else
        {
          std::array<double, 3> eta{ eta_x, eta_y, 0 };

          auto vcoord = logical2vertex<2>(forest, which_tree, eta);

          data[3 * inode + 0] = static_cast<float>(vcoord[0]);
          data[3 * inode + 1] = static_cast<float>(vcoord[1]);
          data[3 * inode + 2] = static_cast<float>(vcoord[2]);
        }

        ++inode;

      } // end for jx
    } // end for jy
  }
  else // cell-based AMR
  {
    int    k = 0; // vertex index
    double eta_z = 0;

    for (int yi = 0; yi < 2; ++yi)
    {
      double eta_y = intsize * q->y + h2 * (1.0 + (yi * 2 - 1) * scale);

      for (int xi = 0; xi < 2; ++xi)
      {
        KALYPSSO_ASSERT(0 <= k and k < NB_CHILDREN);
        double eta_x = intsize * q->x + h2 * (1.0 + (xi * 2 - 1) * scale);

        if (geom != nullptr)
        {

          xyz[0] = eta_x;
          xyz[1] = eta_y;
          xyz[2] = eta_z;
          // from logical coordinates to physical cartesian coordinates
          geom->X(geom, which_tree, xyz, XYZ);
          data[3 * k + 0] = static_cast<float>(XYZ[0]);
          data[3 * k + 1] = static_cast<float>(XYZ[1]);
          data[3 * k + 2] = static_cast<float>(XYZ[2]);
        }
        else
        {

          qcoord_t qxyz[3] = { q->x + h * xi, q->y + h * yi, 0 };

          // cartesian geometry, use the regular qcoord_to_vertex function
          // to retrieve physical coordinates
          p4est::Wrapper<2>::qcoord_to_vertex(forest->connectivity, which_tree, qxyz, XYZ);

          data[3 * k + 0] = static_cast<float>(XYZ[0]);
          data[3 * k + 1] = static_cast<float>(XYZ[1]);
          data[3 * k + 2] = static_cast<float>(XYZ[2]);
        }
        ++k;
      } // for xi
    } // for yi
  }

} // io_fill_coordinates - 2D

// =======================================================
// =======================================================
template <>
void
io_fill_coordinates<3>(forest_t<3> *      forest,
                       geometry_t<3> *    geom,
                       p4est::topidx_t    which_tree,
                       quadrant_t<3> *    q,
                       float *            data,
                       std::array<int, 3> bSize)
{

  auto                  ROOT_LEN = p4est::Wrapper<3>::ROOT_LEN;
  auto                  QUADRANT_LEN = p4est::Wrapper<3>::QUADRANT_LEN;
  [[maybe_unused]] auto NB_CHILDREN = p4est::Wrapper<3>::NB_CHILDREN;
  using qcoord_t = p4est::qcoord_t;

  double xyz[3] = { 0, 0, 0 };
  double XYZ[3] = { 0, 0, 0 };

  const double   hd = (static_cast<double>(QUADRANT_LEN(q->level))) / ROOT_LEN;
  const double   h2 = 0.5 * QUADRANT_LEN(q->level) / ROOT_LEN;
  const qcoord_t h = static_cast<qcoord_t>(QUADRANT_LEN(q->level));
  const double   intsize = 1.0 / ROOT_LEN;
  const double   scale = 1;

  int & bx = bSize[0];
  int & by = bSize[1];
  int & bz = bSize[2];

  bool use_block_amr = (bx < 0 or by < 0 or bz < 0) ? false : true;

  if (use_block_amr)
  {

    double hx = hd / bx;
    double hy = hd / by;
    double hz = hd / bz;

    int inode = 0;
    for (int32_t jz = 0; jz < bz + 1; ++jz)
    {
      double eta_z = intsize * q->z + hz * jz;
      for (int32_t jy = 0; jy < by + 1; ++jy)
      {
        double eta_y = intsize * q->y + hy * jy;
        for (int32_t jx = 0; jx < bx + 1; ++jx)
        {
          double eta_x = intsize * q->x + hx * jx;

          if (geom != nullptr)
          {

            xyz[0] = eta_x;
            xyz[1] = eta_y;
            xyz[2] = eta_z;
            // from logical coordinates to physical cartesian coordinates
            geom->X(geom, which_tree, xyz, XYZ);
            data[3 * inode + 0] = static_cast<float>(XYZ[0]);
            data[3 * inode + 1] = static_cast<float>(XYZ[1]);
            data[3 * inode + 2] = static_cast<float>(XYZ[2]);
          }
          else
          {
            std::array<double, 3> eta{ eta_x, eta_y, eta_z };

            auto vcoord = logical2vertex<3>(forest, which_tree, eta);

            data[3 * inode + 0] = static_cast<float>(vcoord[0]);
            data[3 * inode + 1] = static_cast<float>(vcoord[1]);
            data[3 * inode + 2] = static_cast<float>(vcoord[2]);
          }

          ++inode;

        } // end for jx
      } // end for jy
    } // end for jz
  }
  else // cell-based AMR
  {
    int k = 0; // vertex index

    for (int zi = 0; zi < 2; ++zi)
    {
      double eta_z = intsize * q->z + h2 * (1.0 + (zi * 2 - 1) * scale);

      for (int yi = 0; yi < 2; ++yi)
      {
        double eta_y = intsize * q->y + h2 * (1.0 + (yi * 2 - 1) * scale);

        for (int xi = 0; xi < 2; ++xi)
        {
          KALYPSSO_ASSERT(0 <= k and k < NB_CHILDREN);
          double eta_x = intsize * q->x + h2 * (1.0 + (xi * 2 - 1) * scale);

          if (geom != nullptr)
          {

            xyz[0] = eta_x;
            xyz[1] = eta_y;
            xyz[2] = eta_z;
            // from logical coordinates to physical cartesian coordinates
            geom->X(geom, which_tree, xyz, XYZ);
            data[3 * k + 0] = static_cast<float>(XYZ[0]);
            data[3 * k + 1] = static_cast<float>(XYZ[1]);
            data[3 * k + 2] = static_cast<float>(XYZ[2]);
          }
          else
          {

            qcoord_t qxyz[3] = { q->x + h * xi, q->y + h * yi, q->z + h * zi };

            // cartesian geometry, use the regular qcoord_to_vertex function
            // to retrieve physical coordinates
            p4est::Wrapper<3>::qcoord_to_vertex(forest->connectivity, which_tree, qxyz, XYZ);

            data[3 * k + 0] = static_cast<float>(XYZ[0]);
            data[3 * k + 1] = static_cast<float>(XYZ[1]);
            data[3 * k + 2] = static_cast<float>(XYZ[2]);
          }
          ++k;
        } // for xi
      } // for yi
    } // for zi
  } // end cell-based amr

} // io_fill_coordinates - 3d

} // namespace kalypsso
