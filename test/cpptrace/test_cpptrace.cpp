// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * A minimalist example to use library backward-cpp to pretty print
 * error / stack trace...
 *
 * cpptrace is available from https://github.com/jeremy-rifkin/cpptrace
 */

#include <iostream>

#include <kalypsso/core/cpptrace_utils.h>

void
badass_function()
{
  KALYPSSO_DISABLE_STRINGOP_OVERFLOW_WARNINGS_PUSH()
  char * ptr = (char *)42;
  *ptr = 42;
  KALYPSSO_DISABLE_STRINGOP_OVERFLOW_WARNINGS_POP()
}

int
main(int argc, char * argv[])
{
  kalypsso::cpptrace_initialize();

  std::cout << "Create a segfault on purpose...\n";
  badass_function();

  return 0;
}
