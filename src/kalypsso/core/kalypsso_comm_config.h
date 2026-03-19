// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file kalypsso_comm_config.h
 */
#ifndef KALYPSSO_CORE_KALYPSSOCOMMCONFIG_H_
#define KALYPSSO_CORE_KALYPSSOCOMMCONFIG_H_

#include <kalypsso/utils/p4est/p4est_wrapper.h>

namespace kalypsso
{

//! Define list of MPI tags used by kalypsso exclusively.
//! Start at number above the last tag value used by p4est itself.
enum kalypsso_comm_tag
{
  KALYPSSO_COMM_MESH_PARTITIONER_TAG = P4EST_COMM_TAG_LAST + 1,
  KALYPSSO_COMM_GHOST_EXCHANGE_TAG = P4EST_COMM_TAG_LAST + 2,
};

} // namespace kalypsso

#endif // KALYPSSO_CORE_KALYPSSOCOMMCONFIG_H_
