// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeSmoothInterfaceFunction.h
 */
#ifndef KALYPSSO_CORE_COMPUTE_SMOOTH_INTERFACE_FUNCTION_H_
#define KALYPSSO_CORE_COMPUTE_SMOOTH_INTERFACE_FUNCTION_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>

namespace kalypsso
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute smooth interface function defined as \f$ \psi =
 * \frac{\phi_0^\alpha}{\phi_0^\alpha+\phi_1^\alpha}\f$
 *
 * \f$ \psi \f$ must have same block size as the conservative/primitive variables array but can have
 * a different ghost width
 */
template <size_t dim, typename device_t>
class ComputeSmoothInterfaceFunction
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int64_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

private:
  //! a block array (typically of conservative variables)
  DataArrayBlock_t m_U;

  //! smooth interface function
  DataArrayGhostedBlock_t m_sif;

  //! variable index to address m_q identifying a scalar used for computing the
  //! smooth interface function; e.g. for the godunov_five_eq, it should fm[Hydro::IPHI]
  int32_t m_ivar;

  //! offset to first octant
  const int32_t m_iOct_first;

  //! number of quadrants to process
  const int32_t m_num_quads;

  //! alpha
  const real_t m_alpha;

public:
  // ==============================================================
  // ==============================================================
  ComputeSmoothInterfaceFunction(DataArrayBlock_t        userdata,
                                 DataArrayGhostedBlock_t smooth_interface_function,
                                 int32_t                 ivar,
                                 const int32_t           iOct_first,
                                 const int32_t           num_quads,
                                 const real_t            alpha);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  static void
  apply(ConfigMap const &       config_map,
        DataArrayBlock_t        userdata,
        DataArrayGhostedBlock_t smooth_interface_function,
        int32_t                 ivar,
        const int32_t           iOct_first,
        const int32_t           num_quads);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // ComputeSmoothInterfaceFunction

// explicit template instantiation
extern template class ComputeSmoothInterfaceFunction<2, kalypsso::DefaultDevice>;
extern template class ComputeSmoothInterfaceFunction<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTE_SMOOTH_INTERFACE_FUNCTION_H_
