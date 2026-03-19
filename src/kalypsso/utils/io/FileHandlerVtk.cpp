// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/utils/io/FileHandlerVtk.h>

#include <sstream>

namespace kalypsso
{
namespace io
{

// =======================================================
// =======================================================
FileHandlerVtk::FileHandlerVtk()
  : FileHandler()
  , timeStep(0)
  , isParallel(false)
  , mpiRank(0)
{

  suffix = "vtu";

} // FileHandlerVtk::FileHandlerVtk

// =======================================================
// =======================================================
FileHandlerVtk::FileHandlerVtk(std::string directory_, std::string name_, std::string suffix_)
  : FileHandler(directory_, name_, suffix_)
  , timeStep(0)
  , isParallel(false)
  , mpiRank(0)
{} // FileHandlerVtk::FileHandlerVtk

// =======================================================
// =======================================================
FileHandlerVtk::~FileHandlerVtk() {} // FileHandlerVtk::~FileHandlerVtk

// =======================================================
// =======================================================
std::string
FileHandlerVtk::getFullPath()
{

  std::stringstream filename;

  if (!directory.empty())
    filename << directory << "/";
  filename << name;

  // write timeStep in string timeFormat
  std::ostringstream timeFormat;
  timeFormat.width(7);
  timeFormat.fill('0');
  timeFormat << timeStep;
  filename << "_time";
  filename << timeFormat.str();

  if (isParallel)
  {
    // write MPI rank in string rankFormat
    std::ostringstream rankFormat;
    rankFormat.width(5);
    rankFormat.fill('0');
    rankFormat << mpiRank;
    filename << "_mpi";
    filename << rankFormat.str();
  }

  filename << "." << suffix;

  return filename.str();

} // FileHandlerVtk::getFullPath

} // namespace io

} // namespace kalypsso
