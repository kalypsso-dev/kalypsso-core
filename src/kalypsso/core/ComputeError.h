// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeError.h
 *
 * Just compute L1 or L2 norm of the difference between two data array.
 */
#ifndef KALYPSSO_CORE_COMPUTEERROR_H_
#define KALYPSSO_CORE_COMPUTEERROR_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <../better-enums/enum.h>

namespace kalypsso
{

// clang-format off
BETTER_ENUM(NormType, int, L1, L2)
BETTER_ENUM(NormId, int, ERR, REF)
// clang-format on

//! a simple data structure to hold two values
//! - an error norm
//! - a reference norm
//!
//! this struct is used in a custom sum reduction
struct Norms
{
  real_t values[2];

  KOKKOS_INLINE_FUNCTION
  Norms()
  {
    values[0] = 0;
    values[1] = 0;
  }

  KOKKOS_INLINE_FUNCTION
  Norms(const Norms & rhs)
  {
    values[0] = rhs.values[0];
    values[1] = rhs.values[1];
  }

  KOKKOS_INLINE_FUNCTION
  Norms &
  operator+=(const Norms & src)
  {
    values[0] += src.values[0];
    values[1] += src.values[1];
    return *this;
  }
}; // struct Norms
} // namespace kalypsso

namespace Kokkos
{
template <>
struct reduction_identity<kalypsso::Norms>
{
  KOKKOS_FORCEINLINE_FUNCTION static kalypsso::Norms
  sum()
  {
    return kalypsso::Norms();
  }
};
} // namespace Kokkos

namespace kalypsso
{
/**
 * \class ComputeError
 */
template <size_t dim, typename device_t>
class ComputeError
{
public:
  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! type alias for a data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  using index_t = int64_t;

private:
  //! heavy data
  DataArrayBlock_t m_data1, m_data2;

  //! identify which variable is used to compute error
  int m_varId;

  //! identify which norm is used to compute error
  NormType m_norm_type;

  //! flag to indicate if we want to divide (normalize) values
  bool m_divideByVar0;

public:
  // ====================================================================
  // ====================================================================
  //! constructor.
  //!
  //! \param[in] varId identify which variable to reduce (ID, IE, IU, ...)
  ComputeError(DataArrayBlock_t data1,
               DataArrayBlock_t data2,
               int              varId,
               NormType         norm_type,
               bool             divideByVar0)
    : m_data1(data1)
    , m_data2(data2)
    , m_varId(varId)
    , m_norm_type(norm_type)
    , m_divideByVar0(divideByVar0)
  {}

  // ====================================================================
  // ====================================================================
  //! destructor.
  ~ComputeError() = default;

  // ====================================================================
  // ====================================================================
  //! perform reduction.
  //!
  //! \return return relative error: |U-Uref| / |Uref|
  //! norm used can be either L1 or L2
  static auto
  apply([[maybe_unused]] const ParallelEnv & par_env,
        DataArrayBlock_t                     data1,
        DataArrayBlock_t                     data2,
        int                                  varId,
        NormType                             norm_type,
        bool                                 divideByVar0)
  {
    ComputeError<dim, device_t> functor(data1, data2, varId, norm_type, divideByVar0);

    // check that data1 and data2 have same sizes
    // clang-format off
    assertm(data1.num_cells() == data2.num_cells() and
            data1.num_vars() == data2.num_vars() and
            data1.num_quadrants() == data2.num_quadrants(),
            "[ComputeError] data and data2 array sizes don't match !");
    // clang-format on

    const auto nbCellsPerLeaf = data1.num_cells();
    const auto local_num_octants = data1.num_quadrants();
    const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

    Norms              norms;
    Kokkos::Sum<Norms> reducer(norms);
    Kokkos::parallel_reduce(
      "ComputeError", Kokkos::RangePolicy<exec_space>(0, nbCellsTotal), functor, reducer);

    real_t total_error = 0;
    real_t total_ref = 0;

#ifdef KALYPSSO_CORE_USE_MPI
    par_env.comm().MPI_Reduce<MpiComm::SUM>(&norms.values[NormId::ERR], &total_error, 1, 0);
    par_env.comm().MPI_Reduce<MpiComm::SUM>(&norms.values[NormId::REF], &total_ref, 1, 0);
#else
    total_error = norms.values[NormId::ERR];
    total_ref = norms.values[NormId::REF];
#endif

    if (norm_type == +NormType::L2)
    {
      total_error = sqrt(total_error);
      total_ref = sqrt(total_ref);
    }

    // return relative error
    return total_error / total_ref;

  } // apply

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index, Norms & norms) const
  {
    // convert global index into
    // - octant id
    // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
    const auto iOct = global_index / m_data1.num_cells();
    const auto cell_index = global_index - iOct * m_data1.num_cells();

    const auto value1 = m_divideByVar0
                          ? m_data1(cell_index, m_varId, iOct) / m_data1(cell_index, 0, iOct)
                          : m_data1(cell_index, m_varId, iOct);
    const auto value2 = m_divideByVar0
                          ? m_data2(cell_index, m_varId, iOct) / m_data2(cell_index, 0, iOct)
                          : m_data2(cell_index, m_varId, iOct);

    if (m_norm_type == +NormType::L1)
    {
      norms.values[NormId::ERR] += fabs(value1 - value2);
      norms.values[NormId::REF] += fabs(value2);
    }
    else if (m_norm_type == +NormType::L2)
    {
      norms.values[NormId::ERR] += (value1 - value2) * (value1 - value2);
      norms.values[NormId::REF] += (value2 * value2);
    }

  } // operator ()

}; // class ComputeError

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTEERROR_H_
