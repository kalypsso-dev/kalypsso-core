// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file StiffenedGasMixture.h
 *
 * This is only useful when using at least two material to define pressure and
 * internal  energy in mixed cells.
 */
#ifndef KALYPSSO_CORE_EOS_STIFFENED_GAS_MIXTURE_H_
#define KALYPSSO_CORE_EOS_STIFFENED_GAS_MIXTURE_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/eos/eos_utils.h>

#include <kalypsso/core/eos/StiffenedGasEos.h>

namespace kalypsso
{

namespace core
{

namespace eos
{

//! type alias for an array of equation of state
template <typename device_t>
using sg_eos_array_t = Kokkos::View<StiffenedGasEos *, device_t>;

// ==========================================================================
// ==========================================================================
// ==========================================================================
/**
 * Stiffened gas mixture helper class.
 *
 * In a multifluid simulation, when each fluid obeys a stiffened gas EOS, by weighting each eos by
 * the the volume fraction \f$z_i\f$, summing all equations and using isobaric closure, one obtains
 * the mixture pressure:
 *
 * \f$ p = \frac{\rho e}{\epsilon} - \gamma p_{\infty} \f$,
 *
 * where \f$ \epsilon = \sum_i \frac{z_i}{\gamma_i-1}\f$ and
 * \f$ \gamma p_{\infty} = \frac{1}{\epsilon} \sum_i \frac{z_i p_{\infty,i} \gamma_i}{\gamma_i-1}
 * \f$
 *
 * The mixture quantities are:
 *
 * - volumic fraction conservation : \f$ \sum_i \phi_i = 1 \f$
 * - density : \f$ \rho = \sum_i \phi_i \rho_i \f$
 * - internal energy : \f$ e_{int} = \sum_i $\phi_i eint_i \f$
 * - pressure : \f$ \forall i, p_i = p \f$
 *
 */
template <typename device_t>
class StiffenedGasMixture
{

public:
  StiffenedGasMixture() = default;

  StiffenedGasMixture(const ConfigMap & config_map)
    : m_num_material(static_cast<size_t>(config_map.getInteger("run", "nmat", 2)))
    , m_sg_eos("eos stiffened gas eos array", m_num_material)
    , m_small_p(config_map.getReal("stiffened_gas", "mixture_smallp", KALYPSSO_NUM(1e-9)))
  {
    const auto eos_type = get_eos_type(config_map);

    // only initialize eos array if stiffened gas is requested
    if (eos_type == +EOS_TYPE::STIFFENED_GAS)
    {
      // create stiffened eos array
      auto sg_eos_host = Kokkos::create_mirror_view(m_sg_eos);
      for (size_t i_mat = 0; i_mat < m_num_material; i_mat++)
        sg_eos_host(i_mat) = StiffenedGasEos(i_mat, config_map);
      Kokkos::deep_copy(m_sg_eos, sg_eos_host);
    }
  }

  //! number of materials
  size_t m_num_material;

  //! array of equation of states, one per material
  sg_eos_array_t<device_t> m_sg_eos;

  //! small pressure guard
  const real_t m_small_p;

  /**
   * Compute mixture pressure.
   *
   * To be used only when there are two materials
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  mixture_pressure(real_t rho, real_t eint, real_t phi0) const
  {
    const real_t epsilon = compute_epsilon(phi0);
    const real_t gamma_pinf = compute_gamma_pinf(phi0);
    const auto   pressure = rho * eint / epsilon - gamma_pinf;
    return pressure < m_small_p ? m_small_p : pressure;
  }

  /**
   * Compute mixture specific internal energy.
   *
   * To be used only when there are two materials
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  mixture_specific_eint(real_t pressure, real_t rho, real_t phi0) const
  {
    const real_t epsilon = compute_epsilon(phi0);
    const real_t gamma_pinf = compute_gamma_pinf(phi0);
    return (pressure + gamma_pinf) * epsilon / rho;
  }

  /**
   * Compute mixture speed of sound.
   *
   * To be used only when there are two materials
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  mixture_sound_speed(real_t pressure, real_t rho, real_t phi0) const
  {
    const real_t epsilon = compute_epsilon(phi0);
    const real_t gamma_pinf = compute_gamma_pinf(phi0);
    return sqrt(1 / rho * (gamma_pinf + pressure * (1 + 1 / epsilon)));
  }

private:
  /**
   * Compute epsilon (mixture gas) from volumic fraction (material 0, phi = phi_0).
   *
   * To be used only when there are two materials.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  compute_epsilon(real_t phi0) const
  {
    return phi0 * m_sg_eos(0).one_over_gammam1() + (1 - phi0) * m_sg_eos(1).one_over_gammam1();
  }

  /**
   * Compute gamma times pinf for mixture gas.
   *
   * To be used only when there are two materials.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  compute_gamma_pinf(real_t phi0) const
  {
    const real_t epsilon = compute_epsilon(phi0);
    const real_t phi1 = 1 - phi0;
    auto const & gamma0 = m_sg_eos(0).gamma();
    auto const & gamma1 = m_sg_eos(1).gamma();
    auto const & one_over_gamma0m1 = m_sg_eos(0).one_over_gammam1();
    auto const & one_over_gamma1m1 = m_sg_eos(1).one_over_gammam1();
    auto const & pinf0 = m_sg_eos(0).pinf();
    auto const & pinf1 = m_sg_eos(1).pinf();
    // clang-format off
    return ((phi0 * pinf0 * gamma0) * one_over_gamma0m1 +
            (phi1 * pinf1 * gamma1) * one_over_gamma1m1) / epsilon;
    // clang-format on
  }

}; // struct StiffenedGasMixture

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_STIFFENED_GAS_MIXTURE_H_
