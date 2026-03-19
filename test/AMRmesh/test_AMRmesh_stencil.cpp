// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_AMRmesh_stencil.cpp
 *
 * Purpose:
 * test and validate code used to perform stencil computation (on the leaves of octree).
 *
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/io_utils.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>
#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/real_type.h>
#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/core/MeshGhostsExchanger.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <kalypsso/core/StencilHelper.h>

#include <test_common/InitialAMRSetup.h>
#include <test_common/DataWriter.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <cstdlib>

namespace kalypsso
{

// =============================================================================
// =============================================================================
/**
 * Example stencil computation.
 *
 * Evaluate laplacian:
 * \f$ \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} \f$
 * using a finite difference approximation.
 *
 * Here is the general 3-points finite difference approximation of
 * \f$ \frac{\partial^2 f}{\partial x^2} \f$ :
 *
 *         alpha*dx         beta*dx
 *    x1 <-----------> x0 <-----------> x2
 *
 * with \f$ x_1 = x_0 + \alpha \dif x \f$ with \f$ \alpha<0 \f$ and \f$ x_2 = x_0 + \beta \dif x \f$
 * with \f$ \beta<0 \f$.
 *
 * \f$ f''(x_0) \simeq \frac{2}{(\beta-\alpha)\dif x} \left[ \frac{f(x_0+\beta \dif x)}{\beta
 * \dif x} - \frac{f(x_0+\alpha \dif x)}{\alpha \dif x}
 * -(\frac{1}{\beta}-\frac{1}{\alpha})\frac{f(x_0)}{\dif x}\right] \f$
 *
 * On AMR mesh, when current cell "touches" block border, the neighbor cell can be either at coaser,
 * same or fine level, so we have to adapt the finite difference in that case.
 *
 * E.g. left neighbor is in coarser block, while current cell and right neighbor belong to the same
 * block. In that case we have \f$ \alpha =-\frac{3}{2}\f$ and \f$\beta = 1\f$
 *   ______
 *  |      |
 *  |      |_______
 *  |      |   |   |
 *  |______|___|___|
 *

 * \f$ f''(x_0) \simeq \frac{4}{5 (\dif x)^2} \left[ f(x_2) + \frac{2}{3} f(x_1) -
 *  \frac{5}{3}f(x_0)\right] \f$
 *
 * \tparam dim is dimension (2 or 3)
 * \tparam device_t is kokkos execution space
 * \tparam Function is a functor used to initial data outside domain (border condition)
 */
template <size_t dim, typename device_t, typename Function>
class StencilComputation
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using CellLocation_t = CellLocation<dim>;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  // ====================================================================
  // ====================================================================
  //! constructor.
  StencilComputation(ConfigMap const &        config_map,
                     amr_hashmap_t            amr_hashmap,
                     orchard_key_view_t       orchard_keys,
                     int32_t                  local_num_octants,
                     block_size_t<dim>        block_sizes,
                     brick_size_t<dim>        brick_sizes,
                     Kokkos::Array<bool, dim> is_brick_periodic,
                     int32_t                  iOct_begin,
                     int32_t                  num_octants,
                     DataArrayBlock_t         userdata_in,
                     DataArrayBlock_t         userdata_out,
                     const Function &         f)
    : m_helper(amr_hashmap, orchard_keys, block_sizes, brick_sizes, is_brick_periodic)
    , m_amr_hashmap_device(amr_hashmap)
    , m_orchard_keys_device(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_iOct_begin(iOct_begin)
    , m_num_octants(num_octants)
    , m_nbCellsPerLeaf(Kokkos::dim_prod(block_sizes))
    , m_block_sizes(block_sizes)
    , m_brick_sizes(brick_sizes)
    , m_is_brick_periodic(is_brick_periodic)
    , m_userdata_in(userdata_in)
    , m_userdata_out(userdata_out)
    , m_f(f)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map))
  {}

  // ====================================================================
  // ====================================================================
  //! destructor.
  virtual ~StencilComputation() = default;

  // ====================================================================
  // ====================================================================
  static void
  apply(ConfigMap const &        config_map,
        amr_hashmap_t            amr_hashmap,
        orchard_key_view_t       orchard_keys,
        int32_t                  local_num_octants,
        block_size_t<dim>        block_sizes,
        brick_size_t<dim>        brick_sizes,
        Kokkos::Array<bool, dim> is_brick_periodic,
        int32_t                  iOct_begin,
        int32_t                  num_octants,
        DataArrayBlock_t         userdata_in,
        DataArrayBlock_t         userdata_out)
  {
    Function                                    f;
    StencilComputation<dim, device_t, Function> functor(config_map,
                                                        amr_hashmap,
                                                        orchard_keys,
                                                        local_num_octants,
                                                        block_sizes,
                                                        brick_sizes,
                                                        is_brick_periodic,
                                                        iOct_begin,
                                                        num_octants,
                                                        userdata_in,
                                                        userdata_out,
                                                        f);

    const auto nbCellsPerLeaf = Kokkos::dim_prod(block_sizes);

    const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

    Kokkos::parallel_for(
      "StencilComputation", Kokkos::RangePolicy<exec_space>(0, nbCellsTotal), functor);
  }; // apply

  // ==============================================================
  // ==============================================================
  //! compute face-averaged value in a 2^dim cell mesh.
  //!
  //! Example situation where dir is IX, face is on the right:
  //!    _______________     ______________
  //!   |       |       |   |              |
  //!   |       |   x   |   |              |
  //!   |       |       |   |   neighbor   |
  //!   |_______|_______|   |   cell at    |
  //!   |       |       |   |   coarser    |
  //!   |       |   x   |   |   level      |
  //!   |       |       |   |              |
  //!   |_______|_______|   |______________|
  //!
  //!
  //! Current cell is one of the cells flagged with "x"; what we do here is that we average the
  //! value of all the siblings cells marked with "x".
  //! What we do is find a generic way to identify face-siblings in a 2^dim neighborhood.
  //!
  //! Typical application : compute face average value at a non-conform interface; the face average
  //! value can be used to compute a ghost cell at coarser level.
  //!
  //! \param[in] coord is a dim-vector of integer block-coordinates of current cell
  //! \param[in] ivar identify which variable we want to average
  //! \param[in] iOct_global is the block-octant id (which current cell belong to)
  //!
  //! \tparam dir can be IX, IY or IZ, it identifies a direction orthogonal to a face we want to
  //!         average.
  //!
  //! \return average value
  template <int dir>
  KOKKOS_INLINE_FUNCTION real_t
  compute_face_average(coord_t<dim> const & coord, int ivar, iOct_t iOct_global) const
  {
    auto const & b = m_block_sizes;

    // round down coordinates to a multiple of two
    coord_t<dim> coord0;
    coord0[IX] = (coord[IX] / 2) * 2;
    coord0[IY] = (coord[IY] / 2) * 2;
    if constexpr (dim == 3)
      coord0[IZ] = (coord[IZ] / 2) * 2;

    auto cell_index0 = coord_to_cellindex<dim>(coord0, b);

    real_t        data = 0;
    const int16_t nbCells = (1 << (dim - 1));
    for (int16_t i = 0; i < nbCells; ++i)
    {
      // index ii will span all siblings index matching a face orthogonal to direction "dir"
      // in other words, in the following we swap bit "dir" the last dimension (IY in 2D, IZ in 3D)
      auto ii = swapBits(i, dim - 1, dir);

      // if coord[dir] is odd  we are in a right face
      // if coord[dir] is even we are in a left  face
      if (coord[dir] % 2 == 1)
        ii += (1 << dir);

      if constexpr (dim == 2)
      {
        data +=
          m_userdata_in(cell_index0 + (ii & 0x1) + ((ii & 0x2) >> 1) * b[IX], ivar, iOct_global);
      }
      else if constexpr (dim == 3)
      {
        data += m_userdata_in(cell_index0 + (ii & 0x1) + ((ii & 0x2) >> 1) * b[IX] +
                                ((ii & 0x4) >> 2) * b[IX] * b[IY],
                              ivar,
                              iOct_global);
      }
    }

    return data / nbCells;

  } // compute_face_average

  // ==============================================================
  // ==============================================================
  /**
   * Compute second derivative along direction dir
   *
   * \tparam dir direction along which second partial derivative is computed
   */
  template <int dir>
  KOKKOS_INLINE_FUNCTION real_t
  compute_second_derivative(int32_t const & cell_index, int ivar, int32_t const & iOct_local) const
  {

    const auto & b = m_block_sizes;

    const auto     iOct_global = m_iOct_begin + iOct_local;
    const auto     coord = cellindex_to_coord<dim>(cell_index, m_block_sizes);
    const auto     key_cur = m_orchard_keys_device(iOct_global);
    CellLocation_t cell_loc{ coord, key_cur, iOct_global, false };

    constexpr shift_t<dim> shift_left = []() {
      if constexpr (dim == 2)
      {
        if constexpr (dir == IX)
          return shift_t<dim>{ -1, 0 };
        else if constexpr (dir == IY)
          return shift_t<dim>{ 0, -1 };
      }
      else
      {
        if constexpr (dir == IX)
        {
          return shift_t<dim>{ -1, 0, 0 };
        }
        else if constexpr (dir == IY)
        {
          return shift_t<dim>{ 0, -1, 0 };
        }
        else if constexpr (dir == IZ)
        {
          return shift_t<dim>{ 0, 0, -1 };
        }
      }
    }();

    constexpr shift_t<dim> shift_right = []() {
      if constexpr (dim == 2)
      {
        if constexpr (dir == IX)
          return shift_t<dim>{ 1, 0 };
        else if constexpr (dir == IY)
          return shift_t<dim>{ 0, 1 };
      }
      else
      {
        if constexpr (dir == IX)
        {
          return shift_t<dim>{ 1, 0, 0 };
        }
        else if constexpr (dir == IY)
        {
          return shift_t<dim>{ 0, 1, 0 };
        }
        else if constexpr (dir == IZ)
        {
          return shift_t<dim>{ 0, 0, 1 };
        }
      }
    }();

    const auto cell_loc_left = m_helper.getNeighLoc(cell_loc, shift_left);
    const auto cell_loc_right = m_helper.getNeighLoc(cell_loc, shift_right);
    const auto cell_index_left = coord_to_cellindex<dim>(cell_loc_left.ijk, m_block_sizes);
    const auto cell_index_right = coord_to_cellindex<dim>(cell_loc_right.ijk, m_block_sizes);

    real_t data = 0;

    real_t dx = static_cast<real_t>(orchard_key_t<dim>::octantLength(key_cur)) /
                static_cast<real_t>(orchard_key_t<dim>::ROOT_LENGTH) /
                static_cast<real_t>(m_block_sizes[dir]);

    // compute stencil coef
    const auto coef = [&cell_loc, &cell_loc_left, &cell_loc_right, &dx]() {
      if (cell_loc_left.level() == cell_loc.level() and cell_loc_right.level() == cell_loc.level())
      {
        Kokkos::Array<real_t, 3> stencil_coef{ KALYPSSO_NUM(1.0),
                                               KALYPSSO_NUM(-2.0),
                                               KALYPSSO_NUM(1.0) };
        return stencil_coef / (dx * dx);
      }
      else if (cell_loc_left.level() == cell_loc.level() - 1)
      {
        Kokkos::Array<real_t, 3> stencil_coef{ KALYPSSO_NUM(8.0) / 15,
                                               KALYPSSO_NUM(-20.0) / 15,
                                               KALYPSSO_NUM(12.0) / 15 };
        return stencil_coef / (dx * dx);
      }
      else if (cell_loc_right.level() == cell_loc.level() - 1)
      {
        Kokkos::Array<real_t, 3> stencil_coef{ KALYPSSO_NUM(12.0) / 15,
                                               KALYPSSO_NUM(-20.0) / 15,
                                               KALYPSSO_NUM(8.0) / 15 };
        return stencil_coef / (dx * dx);
      }
      else if (cell_loc_left.level() == cell_loc.level() + 1)
      {
        Kokkos::Array<real_t, 3> stencil_coef{ KALYPSSO_NUM(32.0) / 21,
                                               KALYPSSO_NUM(-56.0) / 21,
                                               KALYPSSO_NUM(24.0) / 21 };
        return stencil_coef / (dx * dx);
      }
      else if (cell_loc_right.level() == cell_loc.level() + 1)
      {
        Kokkos::Array<real_t, 3> stencil_coef{ KALYPSSO_NUM(24.0) / 21,
                                               KALYPSSO_NUM(-56.0) / 21,
                                               KALYPSSO_NUM(32.0) / 21 };
        return stencil_coef / (dx * dx);
      }
      else
      {
        return Kokkos::Array<real_t, 3>{ KALYPSSO_NUM(0.0), KALYPSSO_NUM(0.0), KALYPSSO_NUM(0.0) };
      }
    }();

    // functor to get data in neighbor cell (whether it is a regular cell, or cell outside
    // domain)
    auto getNeighDataSameLevel = [&](CellLocation<dim> cell_loc_neigh,
                                     int32_t           cell_index_neigh,
                                     shift_t<dim>      shift) {
      if (cell_loc_neigh.is_outside_domain)
      {
        // use analytical value to init data on the left (direct neighbor, same level)

        // get real space coordinates of lower left corner of the block
        constexpr bool use_center = false;
        const auto     xyz_corner_vertex = orchard_key_to_vertex_coord<dim>(key_cur, use_center);

        const auto xyz_cell_vertex =
          compute_cell_coordinates<dim>(cell_loc.level(), xyz_corner_vertex, coord, m_block_sizes);
        const auto xyz_cell =
          vertex_coord_to_real_space<dim>(xyz_cell_vertex, m_scaling_factor, m_xyz_min);

        // clang-format off
        if constexpr (dim == 2)
        {
          return static_cast<real_t>(m_f(xyz_cell[IX] + static_cast<real_t>(shift[IX]) * dx,
                                         xyz_cell[IY] + static_cast<real_t>(shift[IY]) * dx,
                                         ivar));
        }
        else if constexpr (dim == 3)
        {
          return static_cast<real_t>(m_f(xyz_cell[IX] + static_cast<real_t>(shift[IX]) * dx,
                                         xyz_cell[IY] + static_cast<real_t>(shift[IY]) * dx,
                                         xyz_cell[IZ] + static_cast<real_t>(shift[IZ]) * dx,
                                         ivar));
        }
        // clang-format on
      }
      else
      {
        return m_userdata_in(cell_index_neigh, ivar, cell_loc_neigh.iOct);
      }
    };

    // Note: among left and right cell neighbors, only one of them can be at different level,
    // this can only append when cell is at block border

    // left and right neighbor are at same level, one of them can be outside domain
    if (cell_loc_left.level() == cell_loc.level() and cell_loc_right.level() == cell_loc.level())
    {

      const auto data_L = getNeighDataSameLevel(cell_loc_left, cell_index_left, shift_left);
      const auto data_R = getNeighDataSameLevel(cell_loc_right, cell_index_right, shift_right);
      const auto data_C = m_userdata_in(cell_index, ivar, iOct_global);
      data = (coef[0] * data_L + coef[1] * data_C + coef[2] * data_R);
    }

    // left neighbor is coarser and right neighbor is same level (actually same block)
    else if (cell_loc_left.level() == cell_loc.level() - 1)
    {
      auto coord_R = coord;
      coord_R[dir] += 1;

      const auto data_L = m_userdata_in(cell_index_left, ivar, cell_loc_left.iOct);
      const auto data_C = compute_face_average<dir>(coord, ivar, iOct_global);
      const auto data_R = compute_face_average<dir>(coord_R, ivar, iOct_global);
      data = (coef[0] * data_L + coef[1] * data_C + coef[2] * data_R);
    }

    // right neighbor is coarser and left neighbor is same level (actually same block)
    else if (cell_loc_right.level() == cell_loc.level() - 1)
    {
      auto coord_L = coord;
      coord_L[dir] -= 1;

      const auto data_L = compute_face_average<dir>(coord_L, ivar, iOct_global);
      const auto data_C = compute_face_average<dir>(coord, ivar, iOct_global);
      const auto data_R = m_userdata_in(cell_index_right, ivar, cell_loc_right.iOct);
      data = (coef[0] * data_L + coef[1] * data_C + coef[2] * data_R);
    }

    // left neighbor is finer
    // average the direct finer neighbors
    else if (cell_loc_left.level() == cell_loc.level() + 1)
    {
      auto coord_L = cellindex_to_coord<dim>(cell_index_left, b);
      // add extra shift because CellLocation contains coordinate of the lower left corner fine cell
      coord_L[dir] += 1;

      auto data_L = compute_face_average<dir>(coord_L, ivar, cell_loc_left.iOct);
      auto data_C = m_userdata_in(cell_index, ivar, iOct_global);
      auto data_R =
        m_userdata_in(shifted_cellindex<dim>(cell_index, shift_right, b), ivar, iOct_global);

      data = (coef[0] * data_L + coef[1] * data_C + coef[2] * data_R);
    }

    // right neighbor is finer
    // average the direct finer neighbors
    else if (cell_loc_right.level() == cell_loc.level() + 1)
    {
      auto coord_R = cellindex_to_coord<dim>(cell_index_right, b);

      auto data_L =
        m_userdata_in(shifted_cellindex<dim>(cell_index, shift_left, b), ivar, iOct_global);
      auto data_C = m_userdata_in(cell_index, ivar, iOct_global);
      auto data_R = compute_face_average<dir>(coord_R, ivar, cell_loc_right.iOct);

      data = (coef[0] * data_L + coef[1] * data_C + coef[2] * data_R);
    }

    return data;

  } // compute_second_derivative

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(const index_t & global_index) const
  {
    const auto iOct_local = global_index / m_nbCellsPerLeaf;
    const auto cell_index = global_index - iOct_local * m_nbCellsPerLeaf;

    const auto iOct_global = m_iOct_begin + iOct_local;

    real_t laplacian = 0.0;
    laplacian += compute_second_derivative<IX>(cell_index, 0, iOct_local);
    laplacian += compute_second_derivative<IY>(cell_index, 0, iOct_local);
    if constexpr (dim == 3)
      laplacian += compute_second_derivative<IZ>(cell_index, 0, iOct_local);

    m_userdata_out(cell_index, 0, iOct_global) = laplacian;

    // just for debug and for facilitating comparison, save additional fields
    for (int32_t var = 1; var < m_userdata_out.num_vars(); ++var)
    {
      if (var == m_userdata_out.num_vars() - 1)
      {
        // difference between computed value and exact value
        m_userdata_out(cell_index, var, iOct_global) =
          laplacian - m_userdata_in(cell_index, var, iOct_global);
      }
      else
      {
        m_userdata_out(cell_index, var, iOct_global) = m_userdata_in(cell_index, var, iOct_global);
      }
    }
  } // operator()

private:
  //! help to compute cell location
  StencilHelper<dim, device_t> m_helper;

  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! total number of octants in the current MPI process (ghost block excluded)
  const int32_t m_local_num_octants;

  //! starting octant id
  const int32_t m_iOct_begin;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_octants;

  //! number of cells per leaf block
  const int32_t m_nbCellsPerLeaf;

  //! block sizes
  const block_size_t<dim> m_block_sizes;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! is p4est connectivity periodic ?
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! a block data array (no ghosts, sizes= bx,by,bz)
  DataArrayBlock_t m_userdata_in;

  //! a ghosted data array (which block ghost cells need to be filled)
  DataArrayBlock_t m_userdata_out;

  //! pointwise init functor
  const Function & m_f;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

}; // class StencilComputation

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t>
void
run_test(const ParallelEnv & par_env, const ConfigMap & config_map)
{
  // using exec_space = typename device_t::execution_space;

  // using DataArrayLeaf_t = DataArrayLeaf<real_t, device_t>;
  // using DataArrayHost_t = DataArrayHost<real_t, device_t>;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  // using DataArrayBlockHost_t = DataArrayBlockHost<real_t, device_t>;

  // using exec_space = typename device_t::execution_space;

  using Hydro_t = core::models::Hydro;

  InitialAMRSetup<dim, device_t, InitFuncParabola> initial_amr_setup(
    par_env, config_map, InitFuncParabola{});
  // InitialAMRSetup<dim, device_t, InitFuncSineWave> initial_amr_setup(
  //   par_env, config_map, InitFuncSineWave{});

  const auto level_min = config_map.getInteger("amr", "level_min", 2);
  const auto level_max = config_map.getInteger("amr", "level_max", 2);
  const auto no_refine = (level_min == level_max);

  // setup initial p4est mesh with some test pseudo random refined/coarsen cells pattern
  initial_amr_setup.setup_initial_mesh(no_refine);

  auto & amr_mesh = initial_amr_setup.mesh();
  auto   mesh_map = initial_amr_setup.mesh_map();
  auto   block_sizes = initial_amr_setup.block_sizes();
  auto   brick_sizes = initial_amr_setup.brick_sizes();
  auto   is_brick_periodic = initial_amr_setup.is_brick_periodic();
  auto   amr_mesh_info = initial_amr_setup.amr_mesh_info();

  // retrieve amr keys
  mesh_map->update_orchard_keys(amr_mesh.forest(), amr_mesh.ghost());
  auto orchard_keys_host = mesh_map->orchard_keys_host_clone();
  auto orchard_keys_device = mesh_map->orchard_keys_clone();

#ifdef KALYPSSO_CORE_USE_MPI
  // mirror keys array must be up to date for MeshGhostExchanger to be functional
  mesh_map->update_mirror_orchard_keys(amr_mesh.ghost());
#endif // KALYPSSO_CORE_USE_MPI

  //  rebuild the hashmap (must be done after AMR cycle)
  constexpr bool on_device = true;
  mesh_map->update_hashmap(on_device);
  auto amr_hashmap_device = mesh_map->hashmap();


  // auto [userdata_leaf, userdata_block] =
  // initial_amr_setup.setup_initial_data(orchard_keys_device);
  auto userdata_leaf = initial_amr_setup.setup_initial_data_leaf(orchard_keys_device);
  auto userdata_block_in = initial_amr_setup.setup_initial_data_block_new(orchard_keys_device);
  userdata_block_in.resize(amr_mesh_info.local_num_quadrants_total());

  auto userdata_block_in_host = DataArrayBlock_t::create_host_mirror_view(userdata_block_in);

  auto userdata_block_out = DataArrayBlock_t("userdata_block_out",
                                             userdata_block_in.block_size(),
                                             userdata_block_in.num_vars(),
                                             userdata_block_in.num_quadrants());
  Kokkos::deep_copy(userdata_block_out.logical_view(), -1.0);

  //
  // save data before stencil computation
  //
  // {
  //   std::string filename = dim == 2 ? "test_AMR_stencil_2d_before" :
  //   "test_AMR_stencil_3d_before";

  //   DataWriter<dim, device_t, Hydro_t>::save(
  //     filename, userdata_block_in, config_map, amr_mesh);
  // }

#ifdef KALYPSSO_CORE_USE_MPI
  //
  // make sure MPI ghosts are OK
  //

  // create the main object doing MPI comm to exchange ghost data
  MeshGhostsExchanger<dim, real_t, device_t> mesh_ghosts_exchanger(
    config_map, par_env, amr_mesh, *mesh_map);
  MPI_Barrier(par_env.mpi_comm());

  mesh_ghosts_exchanger.exchange(userdata_block_in);
#endif // KALYPSSO_CORE_USE_MPI

  //
  // filling ghost blocks (piecewise)
  //
  {
    const auto nbOcts = amr_mesh_info.local_num_quadrants();


    StencilComputation<dim, device_t, InitFuncParabola>::apply(config_map,
                                                               amr_hashmap_device,
                                                               orchard_keys_device,
                                                               nbOcts,
                                                               block_sizes,
                                                               brick_sizes,
                                                               is_brick_periodic,
                                                               0,
                                                               nbOcts,
                                                               userdata_block_in,
                                                               userdata_block_out);
    // StencilComputation<dim, device_t, InitFuncSineWave>::apply(config_map,
    //                                                            amr_hashmap_device,
    //                                                            orchard_keys_device,
    //                                                            nbOcts,
    //                                                            block_sizes,
    //                                                            brick_sizes,
    //                                                            is_brick_periodic,
    //                                                            0,
    //                                                            nbOcts,
    //                                                            userdata_block_in,
    //                                                            userdata_block_out);


    {
      std::string filename = dim == 2 ? "test_AMR_stencil_2d_after" : "test_AMR_stencil_3d_after";

      // provide mapping between variables Id and variables names
      const Hydro_t model(dim);

      DataWriter<dim, device_t, Hydro_t>::save(
        filename, userdata_block_out, config_map, amr_mesh, model);
    }
  }
  // create data array with ghost block filled with analytical values
  auto userdata_ghosted_block_true =
    initial_amr_setup.setup_initial_data_ghosted_block(orchard_keys_device, true);

  // perform comparison
  // auto diff = initial_amr_setup.compute_diff_ghosted_block(
  //   orchard_keys_device, userdata_ghosted_block, userdata_ghosted_block_true);

  // {
  //   std::string filename =
  //     dim == 2 ? "test_AMR_fill_block_ghosts_2d_diff" : "test_AMR_fill_block_ghosts_3d_diff";

  //   DataWriter<dim, device_t, Hydro_t>::save(filename, diff, config_map, amr_mesh, true);
  // }

  //
  // perform a reduce to check everything is ok (this is a unit test)
  //
  // double              errors_square = 0;
  // Kokkos::Sum<double> reducer(errors_square);

  // auto diff_v = diff.view();

  // Kokkos::parallel_reduce(
  //   "Compute_sum_of_differences",
  //   Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
  //     { 0, 0, 0 },
  //     { (int32_t)diff_v.num_cells(), (int32_t)diff_v.num_vars(), (int32_t)diff_v.num_quadrants()
  //     }),
  //   KOKKOS_LAMBDA(const int32_t i, const int32_t j, const int32_t k, double & local_sum) {
  //     local_sum += diff_v(i, j, k) * diff_v(i, j, k);
  //   },
  //   reducer);

  // printf("Square of errors is : %f\n", errors_square);

} // run_test

} // namespace kalypsso

bool
hasEnding(std::string const & fullString, std::string const & ending)
{
  if (fullString.length() >= ending.length())
  {
    return (0 ==
            fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  }
  else
  {
    return false;
  }
} // hasEnding

// =============================================================================
// =============================================================================
// =============================================================================
int
main(int argc, char ** argv)
{

  {
    // initialize mpi, kokkos and p4est, use default MPI communicator : MPI_COMM_WORLD
    kalypsso::ParallelEnv par_env(argc, argv);

#ifdef KALYPSSO_CORE_USE_SPDLOG
    kalypsso::kalypsso_spdlog_config(argc, argv, par_env.rank(), par_env.size());
#endif

    if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
    {
      if (par_env.rank() == 0)
      {
        // clang-format off
        std::cout << "Example cmdline: \"mpirun -np 4 ./test_AMRmesh_stencil --ini test_AMRmesh_stencil_2d.ini --refine_level 6\"\n";
        // clang-format on
      }
      return 0;
    }

    // parse command line : 2d or 3d ?
    bool use_3d = kalypsso::cmdline_arg_exists(argv, argv + argc, "--3d");

    // check if user passed a custom ini filename
    std::string config_filename = kalypsso::cmdline_get_string(argv, argv + argc, "--ini");

    // provide a default config filename (that exists)
    if (config_filename.size() == 0)
      config_filename = use_3d ? "./test_AMRmesh_stencil_3d.ini" : "./test_AMRmesh_stencil_2d.ini";

    kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters(config_filename);

    // check if input file is valid, i.e. parameter run/dimension is 2 or 3
    const auto dim = kalypsso::get_dim(config_map);
    assertm(dim == 2 or dim == 3, "[test_AMRmesh_stencil] Wrong dimension");

    // check command line for optional parameter (refine_level)
    kalypsso::InitialAMRSetupBase::refine_level =
      kalypsso::cmdline_arg_exists(argv, argv + argc, "--refine_level")
        ? kalypsso::cmdline_get_integer(argv, argv + argc, "--refine_level")
        : 4;

    if (par_env.rank() == 0)
    {
      printf("================================================\n");
      printf("START RUN_TEST\n");
      printf("================================================\n");
    }

    if (dim == 2)
    {
      kalypsso::run_test<2, kalypsso::DefaultDevice>(par_env, config_map);
    }
    else if (dim == 3)
    {
      kalypsso::run_test<3, kalypsso::DefaultDevice>(par_env, config_map);
    }
    else
    {
      if (par_env.rank() == 0)
      {
        std::cout << "Input file is not valid ! check parameter run/dimension.\n";
      }
    }

    if (par_env.rank() == 0)
    {
      printf("================================================\n");
      printf("END RUN_TEST\n");
      printf("================================================\n");
    }
  }

  return 0;

} // main
