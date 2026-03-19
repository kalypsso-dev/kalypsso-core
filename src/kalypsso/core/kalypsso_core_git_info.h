// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file kalypsso_core_git_info.h
 */
#ifndef KALYPSSO_CORE_KALYPSSO_CORE_GIT_INFO_H_
#define KALYPSSO_CORE_KALYPSSO_CORE_GIT_INFO_H_

#include <string>

namespace kalypsso
{

struct GitRevisionInfo
{
  //! cmake project version
  static std::string
  version();

  static bool
  has_git_info();

  static std::string
  git_tag();

  static std::string
  git_head();

  static std::string
  git_hash();

  static std::string
  git_remote_url();

  static std::string
  git_branch();

  static bool
  git_is_clean();

  static void
  print();
};

} // namespace kalypsso

#endif // KALYPSSO_CORE_KALYPSSO_CORE_GIT_INFO_H_
