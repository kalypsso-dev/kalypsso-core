// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file cmdline_utils.h
 * \brief a simple stes of function to retrieve arguments from command line.
 */
#ifndef KALYPSSO_CORE_CMDLINE_UTILS_HPP
#define KALYPSSO_CORE_CMDLINE_UTILS_HPP

#include <string>
#include <algorithm> // for std::find
#include <cstdlib>   // for std::atoi

namespace kalypsso
{

/**
 * check if flag was given on command line.
 *
 * \param[in] begin iterator
 * \param[in] end iterator
 * \param[in] arg is a string to match
 *
 * \return boolean indicating if flag was found on command line.
 *
 */
bool
cmdline_arg_exists(char ** begin, char ** end, const std::string & arg)
{
  return std::find(begin, end, arg) != end;
} // cmdline_arg_exists

/**
 * Get a string value from command line.
 *
 * \param[in] begin iterator
 * \param[in] end iterator
 * \param[in] arg is a string to match
 *
 * \return string that is the next token after flag.
 */
std::string
cmdline_get_string(char ** begin, char ** end, const std::string & arg)
{

  char ** itr = std::find(begin, end, arg);

  // check that ++itr exists
  if (itr != end && ++itr != end)
  {
    return std::string(*itr);
  }
  const std::string empty = "";
  return empty;

} // cmdline_get_string

/**
 * Get a integer value from command line.
 *
 * \param[in] begin iterator
 * \param[in] end iterator
 * \param[in] arg is a string to match
 *
 * \return integer that is the next token after flag.
 */
int
cmdline_get_integer(char ** begin, char ** end, const std::string & arg)
{

  char ** itr = std::find(begin, end, arg);

  // check that ++itr exists
  if (itr != end && ++itr != end)
  {
    return std::atoi(*itr);
  }
  return -1;

} // cmdline_get_integer

} // namespace kalypsso

#endif // KALYPSSO_CORE_CMDLINE_UTILS_HPP
