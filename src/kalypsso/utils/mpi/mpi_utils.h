// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KALYPSSO_UTILS_MPI_MPIUTILS_H_
#define KALYPSSO_UTILS_MPI_MPIUTILS_H_

#include <mpi.h>

#include <cstdio>

namespace kalypsso
{

//! inline function checking the result of a MPI API call
int inline mpi_check_error(int                err,
                           char const * const func,
                           const char * const file,
                           int const          line)
{

  if (err != MPI_SUCCESS)
  {
    int  errorStringLen;
    char errorString[MPI_MAX_ERROR_STRING];
    MPI_Error_string(err, errorString, &errorStringLen);
    std::fprintf(stderr, "Error at %s:%d: calling %s ==> %s\n", file, line, func, errorString);
  }

  return err;
} // check_mpi_error

//! preprocessor macro ensuring that the error message can print filen name and line number
#define CHECK_MPI_ERR(value) mpi_check_error((value), #value, __FILE__, __LINE__)

} // namespace kalypsso

#endif // KALYPSSO_UTILS_MPI_MPIUTILS_H_
