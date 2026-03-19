// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KALYPSSO_UTILS_IO_FILEHANDLER_H
#define KALYPSSO_UTILS_IO_FILEHANDLER_H

#include <string>

namespace kalypsso
{
namespace io
{

// ==================================================================
// ==================================================================
/**
 * \class FileHandler
 * \brief Create file name.
 *
 * Does not handle the file descriptor, only the client code can do that.
 * This class is just a simple structure, holding a few parameters
 * to ease building the full filename.
 *
 */
class FileHandler
{

protected:
  std::string directory; /**< name of directory where file resides. */
  std::string name;      /**< name of file. */
  std::string suffix;    /**< suffix. */

public:
  FileHandler();
  FileHandler(std::string directory_, std::string name_, std::string suffix_);
  virtual ~FileHandler();

  void
  setDirectory(std::string the_directory)
  {
    directory = the_directory;
  };
  void
  setName(std::string the_name)
  {
    name = the_name;
  };
  void
  setSuffix(std::string the_suffix)
  {
    suffix = the_suffix;
  };

  /**
   * This is where the full path is build.
   * It will be overloaded in derived class, taking into account
   * specific file numbering for a time series.
   */
  virtual std::string
  getFullPath();

  std::string
  getName() const
  {
    return name;
  };
  std::string
  getDirectory() const
  {
    return directory;
  };
  std::string
  getSuffix() const
  {
    return suffix;
  };

}; // class FileHandler

} // namespace io

} // namespace kalypsso

#endif // KALYPSSO_UTILS_IO_FILEHANDLER_H
