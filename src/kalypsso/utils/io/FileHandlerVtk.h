// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KALYPSSO_UTILS_IO_FILEHANDLERVTK_H
#define KALYPSSO_UTILS_IO_FILEHANDLERVTK_H

#include <kalypsso/utils/io/FileHandler.h>

namespace kalypsso
{
namespace io
{

// =======================================================
// =======================================================
/**
 * \class FileHandlerVtk
 */
class FileHandlerVtk : public FileHandler
{

protected:
  int  timeStep;   /**< identify a time step number */
  bool isParallel; /**< if true, the file name will contain the MPI rank */
  int  mpiRank;    /**< only used if isParallel is true */

public:
  FileHandlerVtk();
  FileHandlerVtk(std::string directory_, std::string name_, std::string suffix_);
  virtual ~FileHandlerVtk();

  void
  setTimeStep(int the_timeStep)
  {
    timeStep = the_timeStep;
  };

  void
  setIsParallel(bool the_isParallel)
  {
    isParallel = the_isParallel;
  };

  void
  setRank(int the_rank)
  {
    mpiRank = the_rank;
  };

  std::string
  getFullPath();

  bool
  IsParallel() const
  {
    return isParallel;
  };

  std::string
  getTimeStep() const
  {
    return directory;
  };

  int
  getRank() const
  {
    return mpiRank;
  };

}; // class FileHandlerVtk

} // namespace io

} // namespace kalypsso

#endif // KALYPSSO_UTILS_IO_FILEHANDLERVTK_H
