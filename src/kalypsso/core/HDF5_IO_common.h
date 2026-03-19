// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HDF5_IO_common.h
 *
 * Some utilities to save AMR data using hdf5+xdmf file format
 */
#ifndef KALYPSSO_CORE_HDF5_IO_COMMON_H_
#define KALYPSSO_CORE_HDF5_IO_COMMON_H_

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_DOUBLE, KALYPSSO_CORE_USE_MPI, ...
#include <type_traits>

#include <highfive/highfive.hpp>

#define IO_NODES_PER_CELL_2D 4
#define IO_TOPOLOGY_TYPE_2D "Quadrilateral"
#define IO_NODES_PER_CELL_3D 8
#define IO_TOPOLOGY_TYPE_3D "Hexahedron"

#define IO_XDMF_NUMBER_TYPE "NumberType=\"Float\" Precision=\"4\""

#ifndef IO_HDF5_COMPRESSION
#  define IO_HDF5_COMPRESSION 3
#endif

namespace kalypsso
{

#ifdef KALYPSSO_CORE_USE_MPI
// Currently, HighFive doesn't wrap retrieving information from property lists.
// Therefore, one needs to use HDF5 directly. For example, to see if collective
// MPI-IO operations were used, one may. Conveniently, this also provides identifiers
// of the cause for not using collective MPI calls.
void
check_collective_io(const HighFive::DataTransferProps & xfer_props);
#endif // KALYPSSO_CORE_USE_MPI

/**
 * \brief All the supported attribute types.
 *
 * \todo should probably be refactored or removed
 */
enum io_attribute_type_t
{
  IO_CELL_SCALAR, //!< A cell-centered scalar.
  IO_CELL_VECTOR, //!< A cell-centered vector.
  IO_NODE_SCALAR, //!< A node-centered scalar.
  IO_NODE_VECTOR, //!< A node-centered vector.
  IO_FACE_DATA,   //!< face-centered data (only for checkpointing FaceDataArrayBlock).
};

/**
 * \brief Compute the coordinates of the 4 (or 8) nodes of a quadrant.
 *
 * Usually, data will be a huge array containing all the local nodes of a
 * process. To write the nodes for a specific quadrant, we call this function
 * with (data + offset) where the offset is calculated as:
 *    offset = 3 * P4EST_CHILDREN * current_quadrant_id
 *
 * This routine is only used when a p4est_node_t is not available, i.e. the scale
 * parameter is smaller than 1.0.
 *
 *
 * If block size parameter (bSize) contains non-negative values, then write
 * all coordinates of all nodes of a sub cartesian grid of size (m_bx,m_by,m_bz).
 */
template <size_t dim>
void
io_fill_coordinates(forest_t<dim> *    forest,
                    geometry_t<dim> *  geom,
                    p4est::topidx_t    which_tree,
                    quadrant_t<dim> *  q,
                    float *            data,
                    std::array<int, 3> bSize);

// 2D specialization
template <>
void
io_fill_coordinates<2>(forest_t<2> *      forest,
                       geometry_t<2> *    geom,
                       p4est::topidx_t    which_tree,
                       quadrant_t<2> *    q,
                       float *            data,
                       std::array<int, 3> bSize);

// 3D specialization
template <>
void
io_fill_coordinates<3>(forest_t<3> *      forest,
                       geometry_t<3> *    geom,
                       p4est::topidx_t    which_tree,
                       quadrant_t<3> *    q,
                       float *            data,
                       std::array<int, 3> bSize);

// =======================================================
// =======================================================
/**
 * \brief Convert a H5T_NATIVE_* type to an XDMF NumberType
 */
template <typename data_t>
static inline const char *
hdf5_native_type_to_string()
{

  if constexpr (std::is_same<data_t, int>::value)
  {
    return "NumberType=\"Int\"";
  }

  else if constexpr (std::is_same<data_t, unsigned int>::value)
  {
    return "NumberType=\"UInt\"";
  }

  else if constexpr (std::is_same<data_t, char>::value)
  {
    return "NumberType=\"Char\"";
  }

  else if constexpr (std::is_same<data_t, unsigned char>::value)
  {
    return "NumberType=\"UChar\"";
  }

  else if constexpr (std::is_same<data_t, short>::value or std::is_same<data_t, int16_t>::value)
  {
    return "NumberType=\"Short\"";
  }

  else if constexpr (std::is_same<data_t, unsigned short>::value or
                     std::is_same<data_t, uint16_t>::value)
  {
    return "NumberType=\"UShort\"";
  }

  else if constexpr (std::is_same<data_t, uint32_t>::value)
  {
    return "NumberType=\"UInt\" Precision=\"4\"";
  }

  else if constexpr (std::is_same<data_t, uint64_t>::value)
  {
    return "NumberType=\"UInt\" Precision=\"8\"";
  }

  else if constexpr (std::is_same<data_t, float>::value)
  {
    return "NumberType=\"Float\" Precision=\"4\"";
  }

  else if constexpr (std::is_same<data_t, double>::value)
  {
    return "NumberType=\"Float\" Precision=\"8\"";
  }

  // KALYPSSO_GLOBAL_INFOF ("Unsupported number type %ld.\n", (long int)type);
  return IO_XDMF_NUMBER_TYPE;

} // hdf5_native_type_to_string

} // namespace kalypsso

#endif // KALYPSSO_CORE_HDF5_IO_COMMON_H_
