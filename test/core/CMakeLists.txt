# SPDX-FileCopyrightText: 2025 kalypsso-core authors
#
# SPDX-License-Identifier: Unlicense

add_executable(test_binomial_coef test_binomial_coef.cpp)

target_link_libraries(test_binomial_coef PRIVATE Kokkos::kokkos kalypsso::core)
target_link_libraries(test_binomial_coef PRIVATE kalypsso_warnings)
