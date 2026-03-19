// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ShockBubbleParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_SHOCK_BUBBLE_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_SHOCK_BUBBLE_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Shock-bubble interaction test parameters.
 *
 * Here we can handle multiple bubbles.
 *
 * references :
 * - single bubble: http://amroc.sourceforge.net/examples/euler/2d/html/shbubble_n.htm
 * - multiple bubbles: Triangular metric-based mesh adaptation for compressible multi-material flows
 * in semi-Lagrangian coordinates, Stéphane Del Pino and Isabelle Marmajou, Journal of Computational
 * Physics Volume 478, 1 April 2023, 111975. https://doi.org/10.1016/j.jcp.2023.111975
 */
template <typename device_t>
struct ShockBubbleParams
{

  // shock-bubble problem parameters

  //! initial front location
  real_t x_front;

  //! post-shock fluid state
  real_t post_rho;
  real_t post_pressure;
  real_t post_u;
  real_t post_v;
  real_t post_w;

  //! pre-shock fluid state
  real_t pre_rho;
  real_t pre_pressure;
  real_t pre_u;
  real_t pre_v;
  real_t pre_w;

  //! number of bubbles
  int num_bubbles;

  //! bubble fluid states
  Kokkos::View<real_t *, device_t> bubble_rho;
  Kokkos::View<real_t *, device_t> bubble_pressure;
  Kokkos::View<real_t *, device_t> bubble_u;
  Kokkos::View<real_t *, device_t> bubble_v;
  Kokkos::View<real_t *, device_t> bubble_w;

  //! bubble center locations
  Kokkos::View<real_t *, device_t> bubble_x;
  Kokkos::View<real_t *, device_t> bubble_y;
  Kokkos::View<real_t *, device_t> bubble_z;

  //! bubble radius
  Kokkos::View<real_t *, device_t> bubble_radius;

  //! inlet border conditions enable
  bool use_inlet_bc;

  ShockBubbleParams(ConfigMap const & config_map)
  {
    x_front = config_map.getReal("shock_bubble", "x_front", KALYPSSO_NUM(1.0));

    post_rho = config_map.getReal("shock_bubble", "post_rho", KALYPSSO_NUM(3.81));
    post_pressure = config_map.getReal("shock_bubble", "post_p", KALYPSSO_NUM(10.0));
    post_u = config_map.getReal("shock_bubble", "post_u", KALYPSSO_NUM(2.85));
    post_v = config_map.getReal("shock_bubble", "post_v", KALYPSSO_NUM(0.0));
    post_w = config_map.getReal("shock_bubble", "post_w", KALYPSSO_NUM(0.0));

    pre_rho = config_map.getReal("shock_bubble", "pre_rho", KALYPSSO_NUM(1.0));
    pre_pressure = config_map.getReal("shock_bubble", "pre_p", KALYPSSO_NUM(1.0));
    pre_u = config_map.getReal("shock_bubble", "pre_u", KALYPSSO_NUM(0.0));
    pre_v = config_map.getReal("shock_bubble", "pre_v", KALYPSSO_NUM(0.0));
    pre_w = config_map.getReal("shock_bubble", "pre_w", KALYPSSO_NUM(0.0));

    num_bubbles = config_map.getInteger("shock_bubble", "num_bubbles", 1);

    auto bubble_rho_h = config_map.getRealVector(
      "shock_bubble", "bubble_rho", std::vector<real_t>{ KALYPSSO_NUM(0.1) });
    assertm(bubble_rho_h.size() == static_cast<size_t>(num_bubbles),
            "[ShockBubbleParams] wrong size.");
    bubble_rho = to_view<real_t, typename device_t::memory_space>(bubble_rho_h);

    auto bubble_pressure_h = config_map.getRealVector(
      "shock_bubble", "bubble_p", std::vector<real_t>{ KALYPSSO_NUM(1.0) });
    assertm(bubble_pressure_h.size() == static_cast<size_t>(num_bubbles),
            "[ShockBubbleParams] wrong size.");
    bubble_pressure = to_view<real_t, typename device_t::memory_space>(bubble_pressure_h);

    auto bubble_u_h = config_map.getRealVector(
      "shock_bubble", "bubble_u", std::vector<real_t>{ KALYPSSO_NUM(0.0) });
    assertm(bubble_u_h.size() == static_cast<size_t>(num_bubbles),
            "[ShockBubbleParams] wrong size.");
    bubble_u = to_view<real_t, typename device_t::memory_space>(bubble_u_h);

    auto bubble_v_h = config_map.getRealVector(
      "shock_bubble", "bubble_v", std::vector<real_t>{ KALYPSSO_NUM(0.0) });
    assertm(bubble_v_h.size() == static_cast<size_t>(num_bubbles),
            "[ShockBubbleParams] wrong size.");
    bubble_v = to_view<real_t, typename device_t::memory_space>(bubble_v_h);

    auto bubble_w_h = config_map.getRealVector(
      "shock_bubble", "bubble_w", std::vector<real_t>{ KALYPSSO_NUM(0.0) });
    assertm(bubble_w_h.size() == static_cast<size_t>(num_bubbles),
            "[ShockBubbleParams] wrong size.");
    bubble_w = to_view<real_t, typename device_t::memory_space>(bubble_w_h);

    auto bubble_x_h = config_map.getRealVector(
      "shock_bubble", "bubble_x", std::vector<real_t>{ KALYPSSO_NUM(0.4) });
    assertm(bubble_x_h.size() == static_cast<size_t>(num_bubbles),
            "[ShockBubbleParams] wrong size.");
    bubble_x = to_view<real_t, typename device_t::memory_space>(bubble_x_h);

    auto bubble_y_h = config_map.getRealVector(
      "shock_bubble", "bubble_y", std::vector<real_t>{ KALYPSSO_NUM(0.0) });
    assertm(bubble_y_h.size() == static_cast<size_t>(num_bubbles),
            "[ShockBubbleParams] wrong size.");
    bubble_y = to_view<real_t, typename device_t::memory_space>(bubble_y_h);

    auto bubble_z_h = config_map.getRealVector(
      "shock_bubble", "bubble_z", std::vector<real_t>{ KALYPSSO_NUM(0.0) });
    assertm(bubble_z_h.size() == static_cast<size_t>(num_bubbles),
            "[ShockBubbleParams] wrong size.");
    bubble_z = to_view<real_t, typename device_t::memory_space>(bubble_z_h);

    auto bubble_radius_h = config_map.getRealVector(
      "shock_bubble", "bubble_radius", std::vector<real_t>{ KALYPSSO_NUM(0.1) });
    assertm(bubble_radius_h.size() == static_cast<size_t>(num_bubbles),
            "[ShockBubbleParams] wrong size.");
    bubble_radius = to_view<real_t, typename device_t::memory_space>(bubble_radius_h);

    use_inlet_bc = config_map.getBool("shock_bubble", "use_inlet_bc", false);
  }

}; // struct ShockBubbleParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_SHOCK_BUBBLE_PARAMS_H_
