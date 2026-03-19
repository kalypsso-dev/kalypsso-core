// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeRefineFlags_utils.h
 */
#ifndef KALYPSSO_CORE_COMPUTEREFINEFLAGS_UTILS_H_
#define KALYPSSO_CORE_COMPUTEREFINEFLAGS_UTILS_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)

#include <kalypsso/utils/config/ConfigMap.h>
#include <../better-enums/enum.h>

namespace kalypsso
{

// clang-format off
/**
 * An enum type to represent all possible refine indicator types.
 */
BETTER_ENUM(Indicator, uint8_t,
            LOHNER_SPLIT = 0,
            LOHNER_UNSPLIT = 1,
            SIMPLE_GRADIENT = 2,
            THRESHOLD_AFFINE = 3)
// clang-format on

/**
 * A parameter to use when THRESHOLD_AFFINE indicator is activated.
 */
struct ThresholdAffineParams
{
  //! multiplicative factor
  real_t a;

  //! offset
  real_t b;

  // threshold
  real_t threshold;

  ThresholdAffineParams()
    : a(KALYPSSO_NUM(1.0))
    , b(KALYPSSO_NUM(0.0))
    , threshold(KALYPSSO_NUM(0.0))
  {}

  ThresholdAffineParams(real_t _a, real_t _b, real_t _threshold)
    : a(_a)
    , b(_b)
    , threshold(_threshold)
  {}

}; // struct ThresholdAffineParams

/**
 * A companion data structure for class ComputeRefineFlags holding parameters used to
 * compute refine flags.
 */
struct RefineIndicatorData
{
  static Indicator
  get_indicator(ConfigMap const & config_map)
  {
    auto indicator_name = config_map.getString("amr", "refine_criterion", "LOHNER_UNSPLIT");
    auto maybe_value = Indicator::_from_string_nothrow(indicator_name.c_str());
    if (maybe_value)
      return *maybe_value;
    return Indicator::LOHNER_UNSPLIT;
  }

  //! minimal AMR level allowed
  int level_min;

  //! maximum AMR level allowed
  int level_max;

  //! type of indicator
  Indicator indicator;

  //! threshold to decide refinement
  real_t refine_th;

  //! threshold to decide coarsening
  real_t coarsen_th;

  //! variable id used to compute refine flags
  int ivar;

  //! parameter used in Lohner criterion
  real_t epsilon;

  //! parameters only use when refine_criterion is THRESHOLD_AFFINE
  ThresholdAffineParams th_aff_par{};

}; // struct RefineIndicatorData

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTEREFINEFLAGS_UTILS_H_
