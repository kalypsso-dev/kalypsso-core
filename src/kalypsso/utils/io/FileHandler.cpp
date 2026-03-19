// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/utils/io/FileHandler.h>

#include <sstream>

namespace kalypsso
{
namespace io
{

// =======================================================
// =======================================================
FileHandler::FileHandler()
  : directory("./")
  , name("data")
  , suffix("txt")
{} // FileHandler::FileHandler

// =======================================================
// =======================================================
FileHandler::FileHandler(std::string directory_, std::string name_, std::string suffix_)
  : directory(directory_)
  , name(name_)
  , suffix(suffix_)
{} // FileHandler::FileHandler

// =======================================================
// =======================================================
FileHandler::~FileHandler() {} // FileHandler::~FileHandler

// =======================================================
// =======================================================
std::string
FileHandler::getFullPath()
{

  std::stringstream filename;

  if (!directory.empty())
    filename << directory << "/";
  filename << name;

  filename << "." << suffix;

  return filename.str();

} // FileHandler::getFullPath

} // namespace io

} // namespace kalypsso
