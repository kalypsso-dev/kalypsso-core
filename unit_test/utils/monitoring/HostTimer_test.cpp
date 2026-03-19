// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/utils/monitoring/HostTimer.h>

#include <cstdint>
#include <thread>

#include <gtest/gtest.h>

namespace kalypsso
{

TEST(kalypsso_shared_HostTimer_test, timer_test)
{
  using std::chrono::operator"" ms;

  HostTimer timer;

  timer.start();
  const auto start = std::chrono::high_resolution_clock::now();
  std::this_thread::sleep_for(1500ms);
  const auto end = std::chrono::high_resolution_clock::now();
  timer.stop();

  const double elapsed1 =
    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;
  const auto elapsed2 = timer.elapsed();

  EXPECT_NEAR(elapsed1, elapsed2, 0.01) << "timer failed";
}

} // namespace kalypsso
