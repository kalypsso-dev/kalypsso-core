// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KALYPSSO_CORE_POINT_H
#define KALYPSSO_CORE_POINT_H

#include <array>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/** typedef Point holding coordinates of a point. */
template <size_t dim>
using Point = std::array<real_t, dim>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_POINT_H
