// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/kalypsso_core_git_info.h>

#include <kalypsso_core_version.h>

#include <iostream>

namespace kalypsso
{

std::string
GitRevisionInfo::version()
{
  return KALYPSSO_CORE_VERSION;
}

bool
GitRevisionInfo::has_git_info()
{
#ifdef KALYPSSO_HAS_GIT_INFO
  return true;
#else
  return false;
#endif
}

std::string
GitRevisionInfo::git_tag()
{
#ifdef KALYPSSO_HAS_GIT_INFO
  return KALYPSSO_GIT_TAG;
#else
  return "unknown tag";
#endif
}

std::string
GitRevisionInfo::git_head()
{
#ifdef KALYPSSO_HAS_GIT_INFO
  return KALYPSSO_GIT_HEAD;
#else
  return "unknown head";
#endif
}

std::string
GitRevisionInfo::git_hash()
{
#ifdef KALYPSSO_HAS_GIT_INFO
  return KALYPSSO_GIT_HASH;
#else
  return "unknown hash";
#endif
}

std::string
GitRevisionInfo::git_remote_url()
{
#ifdef KALYPSSO_HAS_GIT_INFO
  return KALYPSSO_GIT_REMOTE_URL;
#else
  return "unknown remote url";
#endif
}

std::string
GitRevisionInfo::git_branch()
{
#ifdef KALYPSSO_HAS_GIT_INFO
  return KALYPSSO_GIT_BRANCH;
#else
  return "unknown git branch";
#endif
}

bool
GitRevisionInfo::git_is_clean()
{
#ifdef KALYPSSO_HAS_GIT_INFO
  return KALYPSSO_GIT_IS_CLEAN;
#else
  return false;
#endif
}

void
GitRevisionInfo::print()
{
  if (has_git_info())
  {
    std::cout << "#############################################\n";
    std::cout << "kalypsso_core - git information" << "\n";
    std::cout << "git remote url : " << git_remote_url() << "\n";
    std::cout << "git branch     : " << git_branch() << "\n";
    std::cout << "git head       : " << git_head() << "\n";
    std::cout << "git hash       : " << git_hash() << " (";
    if (git_is_clean())
    {
      std::cout << "clean";
    }
    else
    {
      std::cout << "dirty";
    }
    std::cout << ")\n";
    std::cout << "#############################################\n";
  }
  else
  {
    std::cout << "#############################################\n";
    std::cout << "kalypsso - not built from a git repository      \n";
    std::cout << "version       : " << version() << "\n";
    std::cout << "#############################################\n";
  }

} // GitRevisionInfo::print

} // namespace kalypsso
