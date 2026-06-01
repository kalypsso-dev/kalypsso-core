// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file EdgeDataArrayBlock.h
 */
#ifndef KALYPSSO_CORE_EDGEDATAARRAYBLOCK_H_
#define KALYPSSO_CORE_EDGEDATAARRAYBLOCK_H_

#include <kalypsso/core/EdgeDataArrayBlock_utils.h>
#include <kalypsso/core/FaceDataArrayBlock.h>
#include <kalypsso/core/DataArrayBlock.h>

namespace kalypsso
{

// ============================================================================================
// ============================================================================================
// ============================================================================================
/**
 * A Data container helper to store d-dimensional vector field edge-staggered components.
 *
 * More precisely, on edges, we only stored longitudinal components.
 * The aim is to be able to compute a face-centered vector field as the curl of an edge-centered
 * vector field.
 *
 * Taking as example magnetic vector potential A (a vector field such that magnetic field B =
 * curl(A)):
 * - Ax is center on x-edges
 * - Ay is center on y-edges
 * - Az is center on z-edges
 *
 * \note Currently in kalypsso blocks can only be of equal size in each direction.
 * So, for example, in 3D, each component is stored using an array of N x (N+1) x (N+1) elements.
 * Later, if needed, we could add support for block of independent sizes along each direction.
 *
 * \note In 2d, only Az is actually used and stored.
 *
 */
template <size_t dim, typename T, typename device_t>
class EdgeDataArrayBlock
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, T, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, T, device_t>;
  using EdgeDataArrayBlock_t = EdgeDataArrayBlock<dim, T, device_t>;

  using EdgeFlatArray_t = Kokkos::View<T *, device_t>;
  using EdgeFlatArrayUnmanaged_t =
    Kokkos::View<T *, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using ExecutionSpace = typename device_t::execution_space;

private:
  //! block size : number of cells in each direction (without counting ghost cells).
  block_size_t<dim> m_bSize;

  //! ghost size : number of ghosts cells all around block of cells.
  //! m_gSize must be even because prologation operator would be too complex to implement
  //! defaulted value is zero.
  int32_t m_gSize;

  //! block size : number of cells in each direction (with ghost cells included).
  block_size_t<dim> m_bSize_total;

  //! number of x-edges in each direction - used for flat index computation
  block_size_t<dim> m_bSize_edge_x;

  //! number of y-edges in each direction - used for flat index computation
  block_size_t<dim> m_bSize_edge_y;

  //! number of z-edges in each direction - used for flat index computation
  block_size_t<dim> m_bSize_edge_z;

  //! offset used as starting address of each component of the field inside a block
  //! - m_offsets[1] number of x-edges elements per octant
  //! - m_offsets[3] number of x-edges and y-edges elements per octant
  //! - m_offsets[3] total num of edges elements (x,y and z) per octant
  edge_flat_index_offset_t m_offsets;

  //! total number of quadrants (owned, MPI ghost, outside, outside_ghosts)
  //! may changed when resizing
  int m_num_quadrants;

  //! 1D storage array, it should contain enough space to hold m_data_x, m_data_y and m_data_z
  //! may be reallocated when the number of octants changes
  EdgeFlatArray_t m_storage_data;

public:
  EdgeDataArrayBlock() = default;

  /**
   * Constructor without ghost cells.
   */
  EdgeDataArrayBlock(std::string name, block_size_t<dim> bSize, int num_quadrants)
    : m_bSize(bSize)
    , m_gSize(0)
    , m_bSize_total(bSize)
    , m_bSize_edge_x(get_edge_block_size<dim>(bSize, IX))
    , m_bSize_edge_y(get_edge_block_size<dim>(bSize, IY))
    , m_bSize_edge_z(get_edge_block_size<dim>(bSize, IZ))
    , m_offsets(compute_edge_flat_index_offsets<dim>(bSize))
    , m_num_quadrants(num_quadrants)
    , m_storage_data(Kokkos::view_alloc(Kokkos::WithoutInitializing, name),
                     m_offsets[3] * static_cast<size_t>(num_quadrants))
  {} // EdgeDataArrayBlock

  /**
   * Constructor with ghost cells.
   */
  EdgeDataArrayBlock(std::string       name,
                     block_size_t<dim> bSize,
                     int32_t           ghostwidth,
                     int               num_quadrants)
    : m_bSize(bSize)
    , m_gSize(ghostwidth)
    , m_bSize_total(bSize + 2 * ghostwidth)
    , m_bSize_edge_x(get_edge_block_size<dim>(bSize + 2 * ghostwidth, IX))
    , m_bSize_edge_y(get_edge_block_size<dim>(bSize + 2 * ghostwidth, IY))
    , m_bSize_edge_z(get_edge_block_size<dim>(bSize + 2 * ghostwidth, IZ))
    , m_offsets(compute_edge_flat_index_offsets<dim>(bSize + 2 * ghostwidth))
    , m_num_quadrants(num_quadrants)
    , m_storage_data(Kokkos::view_alloc(Kokkos::WithoutInitializing, name),
                     m_offsets[3] * static_cast<size_t>(num_quadrants))
  {} // EdgeDataArrayBlock

  /**
   * accumulate over all directions the number of edge elements in that direction per octant.
   */
  KOKKOS_FORCEINLINE_FUNCTION auto
  offsets() const
  {
    return m_offsets;
  } // offsets

  /**
   * convert a logical multi-index (i, j, iOct) into a flat index to storage.
   *
   * \tparam dir is used to determine which type of edge (along which direction: IX, IY or IZ)
   */
  template <int dir, typename I0, typename I1, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION int64_t
  flat_index(I0 i0, I1 i1, IOct iOct) const
  {
    if constexpr (dir == IX)
    {
      KOKKOS_ASSERT(static_cast<int32_t>(i0) < m_bSize_edge_x[IX] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT(static_cast<int32_t>(i1) < m_bSize_edge_x[IY] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i1");

      return m_offsets[0] + i0 + m_bSize_edge_x[IX] * i1 + iOct * m_offsets[3];
    }
    else if constexpr (dir == IY)
    {
      KOKKOS_ASSERT(static_cast<int32_t>(i0) < m_bSize_edge_y[IX] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT(static_cast<int32_t>(i1) < m_bSize_edge_y[IY] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i1");

      return m_offsets[1] + i0 + m_bSize_edge_y[IX] * i1 + iOct * m_offsets[3];
    }
    else if constexpr (dir == IZ)
    {
      KOKKOS_ASSERT(static_cast<int32_t>(i0) < m_bSize_edge_z[IX] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT(static_cast<int32_t>(i1) < m_bSize_edge_z[IY] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i1");

      return m_offsets[2] + i0 + m_bSize_edge_z[IX] * i1 + iOct * m_offsets[3];
    }
    return 0;
  } // flat_index

  /**
   * convert a logical multi-index (i, j, k, iOct) into a flat index to storage.
   *
   * \tparam dir is used to determine which type of edge (along which direction: IX, IY or IZ)
   */
  template <int dir, typename I0, typename I1, typename I2, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION int64_t
  flat_index(I0 i0, I1 i1, I2 i2, IOct iOct) const
  {
    if constexpr (dir == IX)
    {
      KOKKOS_ASSERT(static_cast<int32_t>(i0) < m_bSize_edge_x[IX] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT(static_cast<int32_t>(i1) < m_bSize_edge_x[IY] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i1");
      KOKKOS_ASSERT(static_cast<int32_t>(i2) < m_bSize_edge_x[IZ] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i2");

      return m_offsets[0] + i0 + m_bSize_edge_x[IX] * (i1 + m_bSize_edge_x[IY] * i2) +
             iOct * m_offsets[3];
    }
    else if constexpr (dir == IY)
    {
      KOKKOS_ASSERT(static_cast<int32_t>(i0) < m_bSize_edge_y[IX] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT(static_cast<int32_t>(i1) < m_bSize_edge_y[IY] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i1");
      KOKKOS_ASSERT(static_cast<int32_t>(i2) < m_bSize_edge_y[IZ] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i2");

      return m_offsets[1] + i0 + m_bSize_edge_y[IX] * (i1 + m_bSize_edge_y[IY] * i2) +
             iOct * m_offsets[3];
    }
    else if constexpr (dir == IZ)
    {
      KOKKOS_ASSERT(static_cast<int32_t>(i0) < m_bSize_edge_z[IX] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i0");
      KOKKOS_ASSERT(static_cast<int32_t>(i1) < m_bSize_edge_z[IY] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i1");
      KOKKOS_ASSERT(static_cast<int32_t>(i2) < m_bSize_edge_z[IZ] and
                    "[EdgeDataArrayBlock::flat_index] wrong index i2");

      return m_offsets[2] + i0 + m_bSize_edge_z[IX] * (i1 + m_bSize_edge_z[IY] * i2) +
             iOct * m_offsets[3];
    }
    return 0;
  } // flat_index

  //! memory access operator() as if dealing with a regular Kokkos::View - 2d case
  template <typename I0, typename I1, typename IDir, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, IDir dir, IOct iOct) const
  {
    KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) and "EdgeDataArrayBlock: wrong direction");
    KOKKOS_ASSERT((static_cast<size_t>(iOct) < m_storage_data.extent(0)) and
                  "EdgeDataArrayBlock: index iOct too large");

    if (dir == IX)
      return m_storage_data(flat_index<IX>(i0, i1, iOct));
    else if (dir == IY)
      return m_storage_data(flat_index<IY>(i0, i1, iOct));
    else if (dir == IZ)
      return m_storage_data(flat_index<IZ>(i0, i1, iOct));
    else // shouldn't be here
      return m_storage_data(flat_index<IX>(i0, i1, iOct));
  }

  //! memory access operator() as if dealing with a regular Kokkos::View - 3d case
  template <typename I0, typename I1, typename I2, typename IDir, typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(I0 i0, I1 i1, I2 i2, IDir dir, IOct iOct) const
  {
    KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) and "EdgeDataArrayBlock: wrong direction");
    KOKKOS_ASSERT((static_cast<size_t>(iOct) < m_storage_data.extent(0)) and
                  "EdgeDataArrayBlock: index iOct too large");

    if (dir == IX)
      return m_storage_data(flat_index<IX>(i0, i1, i2, iOct));
    else if (dir == IY)
      return m_storage_data(flat_index<IY>(i0, i1, i2, iOct));
    else if (dir == IZ)
      return m_storage_data(flat_index<IZ>(i0, i1, i2, iOct));
    else // shouldn't be here
      return m_storage_data(flat_index<IX>(i0, i1, i2, iOct));
  }

  //! memory access operator() as if dealing with a regular Kokkos::View - 2d/3d case
  template <typename IOct>
  KOKKOS_FORCEINLINE_FUNCTION T &
  operator()(edge_multiindex_t<dim> indexes, IOct iOct) const
  {
    if constexpr (dim == 2)
      return this->operator()(indexes[IX], indexes[IY], indexes[dim], iOct);
    else if constexpr (dim == 3)
      return this->operator()(indexes[IX], indexes[IY], indexes[IZ], indexes[dim], iOct);
  }

  auto
  storage() -> decltype(m_storage_data)
  {
    return m_storage_data;
  }

  auto
  storage_ref() -> decltype(m_storage_data) &
  {
    return m_storage_data;
  }

  //! Resize m_storage data using new number of octant.
  //!
  //! This function resizes data without copying old data, without even initializing the new array.
  //! This is enough especially when doing load balancing; data will be initialized by MPI
  //! communications.
  void
  resize(int32_t num_quads)
  {
    // update number of quadrants
    m_num_quadrants = num_quads;

    const size_t new_size = m_offsets[3] * static_cast<size_t>(num_quads);

    Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing), m_storage_data, new_size);

  } // resize

  //! Resize m_storage data using new number of octant.
  //! This function resizes data; it doesn't copy old data, but default initialize to zero.
  void
  resize_and_reset(int32_t num_quads)
  {
    resize(num_quads);
    Kokkos::deep_copy(m_storage_data, 0);

  } // resize_and_reset

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  num_quadrants() const
  {
    return m_num_quadrants;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int32_t
  num_elements_per_octant() const
  {
    return m_offsets[3];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  cell_block_size() const
  {
    return m_bSize_total;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  cell_block_size_inner() const
  {
    return m_bSize;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  edge_block_size(int dir) const
  {
    if (dir == IX)
      return m_bSize_edge_x;
    else if (dir == IY)
      return m_bSize_edge_y;
    else if (dir == IZ)
      return m_bSize_edge_z;

    return m_bSize_edge_x;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  ghostwidth() const
  {
    return m_gSize;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto
  is_ghosted() const
  {
    return m_gSize > 0;
  }

  auto
  label() const
  {
    return m_storage_data.label();
  }

  auto
  allocated_size_in_bytes() const
  {
    auto size = m_storage_data.extent(0) * sizeof(T);
    return size;
  }

  /**
   * \return maximum number of edges inside a block (ghost cells included).
   */
  KOKKOS_FORCEINLINE_FUNCTION
  auto
  max_num_edges_per_leaf() const
  {
    return maximum_number_edges_per_leaf<dim>(m_bSize_total);
  }

public:
  /**
   * Compute curl (rotationnel) of an EdgeDataArrayBlock.
   *
   * This was design to ease computing B = rot(A) where
   * - B is magnetic field (a FaceDataArrayBlock)
   * - A is vector potential (a EdgeDataArrayBlock)
   *
   * \param[in] edata is the edge data array
   * \param[in] orchard_keys
   *
   * \return a FaceDataArrayBlock that is the curl of input edge array
   */
  static FaceDataArrayBlock_t
  compute_curl(EdgeDataArrayBlock_t                          edata,
               typename orchard_key_base_t<device_t>::view_t orchard_keys)
  {
    const std::string curl("curl_of_");
    const auto        label = curl + edata.label();

    const auto    block_size = edata.cell_block_size();
    auto          res = FaceDataArrayBlock_t(label, block_size, edata.num_quadrants());
    const int64_t total_num_faces = res.num_elements_per_octant() * edata.num_quadrants();
    const auto    nbFacesPerLeaf = res.num_elements_per_octant();

    Kokkos::parallel_for(
      "compute curl of an EdgeDataArrayBlock",
      Kokkos::RangePolicy<ExecutionSpace>(0, total_num_faces),
      KOKKOS_LAMBDA(const int64_t & global_index) {
        // convert global index into
        // - octant id
        // - face_index inside block (from 0 to num_elements_per_octant-1)
        const auto iOct = global_index / nbFacesPerLeaf;
        const auto face_index = static_cast<int32_t>(global_index - iOct * nbFacesPerLeaf);

        // get block level
        const auto level = orchard_key_t<dim>::level(orchard_keys(iOct));

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support capturing variables inside the inner lambda inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (edata.num_quadrants() == 0 or res.num_quadrants() == 0 or block_size[IX] == 0)
          dummy++;
#endif

        if constexpr (dim == 2)
        {
          auto const face_indexes =
            face_flat_index_unravel<2>(face_index, block_size, res.offsets(), res.shift());
          auto const & i = face_indexes[IX];
          auto const & j = face_indexes[IY];
          auto const & ivar = face_indexes[dim];

          const auto   dx = compute_cell_length<dim>(level, block_size[IX]);
          const auto & dy = dx; // assuming square cells

          if (ivar == IX)
          {
            res(i, j, ivar, iOct) = (edata(i, j + 1, IZ, iOct) - edata(i, j, IZ, iOct)) / dy;
          }
          else if (ivar == IY)
          {
            res(i, j, ivar, iOct) = -(edata(i + 1, j, IZ, iOct) - edata(i, j, IZ, iOct)) / dx;
          }
          else if (ivar == IZ)
          {
            res(i, j, ivar, iOct) = (edata(i + 1, j, IY, iOct) - edata(i, j, IY, iOct)) / dx -
                                    (edata(i, j + 1, IX, iOct) - edata(i, j, IX, iOct)) / dy;
          }
        }
        else if constexpr (dim == 3)
        {
          auto const face_indexes =
            face_flat_index_unravel<3>(face_index, block_size, res.offsets(), res.shift());
          auto const & i = face_indexes[IX];
          auto const & j = face_indexes[IY];
          auto const & k = face_indexes[IZ];
          auto const & ivar = face_indexes[dim];

          const auto   dx = compute_cell_length<dim>(level, block_size[IX]);
          const auto & dy = dx; // assuming square cells
          const auto & dz = dx; // assuming square cells

          if (ivar == IX)
          {
            res(i, j, k, ivar, iOct) =
              (edata(i, j + 1, k, IZ, iOct) - edata(i, j, k, IZ, iOct)) / dy -
              (edata(i, j, k + 1, IY, iOct) - edata(i, j, k, IY, iOct)) / dz;
          }
          if (ivar == IY)
          {
            res(i, j, k, ivar, iOct) =
              (edata(i, j, k + 1, IX, iOct) - edata(i, j, k, IX, iOct)) / dz -
              (edata(i + 1, j, k, IZ, iOct) - edata(i, j, k, IZ, iOct)) / dx;
          }
          else if (ivar == IZ)
          {
            res(i, j, k, ivar, iOct) =
              (edata(i + 1, j, k, IY, iOct) - edata(i, j, k, IY, iOct)) / dx -
              (edata(i, j + 1, k, IX, iOct) - edata(i, j, k, IX, iOct)) / dy;
          }
        }
      });

    return res;

  } // EdgeDataArrayBlock::compute_curl

}; // EdgeDataArrayBlock

} // namespace kalypsso

#endif // KALYPSSO_CORE_EDGEDATAARRAYBLOCK_H_
