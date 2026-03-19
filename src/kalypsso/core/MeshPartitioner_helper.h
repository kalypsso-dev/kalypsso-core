// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MeshPartitioner_helper.h
 */

#include <kalypsso/core/kalypsso_core_base.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>

#include <kalypsso/core/kalypsso_comm_config.h>

#include <kalypsso/utils/log/kalypsso_log.h>

#include <vector> // for std::vector
#include <mpi.h>
#include <type_traits> // for std::is_same_v

namespace kalypsso
{

/**
 * Re-implement p4est_transfer_fixed for bringing device support (CUDA, HIP, ...)
 */
template <size_t dim, typename device_t>
struct MeshPartitioner_helper
{
  using p4est_t = typename p4est::Wrapper<dim>;

  using ExecSpace = typename device_t::execution_space;
  using MemorySpace = typename ExecSpace::memory_space;

  // ========================================================================================
  // ========================================================================================
  static void
  transfer_fixed(const p4est::gloidx_t * dest_gfq,
                 const p4est::gloidx_t * src_gfq,
                 MPI_Comm                mpicomm,
                 int                     tag,
                 void *                  dest_data,
                 const void *            src_data,
                 size_t                  data_size_in_bytes)
  {
    typename p4est_t::transfer_context_t * tc;

    tc = transfer_fixed_begin(
      dest_gfq, src_gfq, mpicomm, tag, dest_data, src_data, data_size_in_bytes);
    p4est_t::transfer_fixed_end(tc);
  } // kalypsso_transfer_fixed

  // ========================================================================================
  // ========================================================================================
  static void
  transfer_fixed_multi_var(const p4est::gloidx_t * dest_gfq,
                           const p4est::gloidx_t * src_gfq,
                           MPI_Comm                mpicomm,
                           int                     tag,
                           void *                  dest_data,
                           const void *            src_data,
                           size_t                  data_size_in_bytes,
                           const uint32_t *        src_offsets,
                           const uint32_t *        dst_offsets)
  {
    typename p4est_t::transfer_context_t * tc;

    tc = transfer_fixed_multi_var_begin(dest_gfq,
                                        src_gfq,
                                        mpicomm,
                                        tag,
                                        dest_data,
                                        src_data,
                                        data_size_in_bytes,
                                        src_offsets,
                                        dst_offsets);
    p4est_t::transfer_fixed_end(tc);
  } // transfer_fixed_multi_var

  // ========================================================================================
  // ========================================================================================
  static void
  transfer_assign_comm([[maybe_unused]] const p4est::gloidx_t * dest_gfq,
                       [[maybe_unused]] const p4est::gloidx_t * src_gfq,
                       sc_MPI_Comm                              mpicomm,
                       int *                                    mpisize,
                       int *                                    mpirank)
  {
    int mpiret;

    P4EST_ASSERT(dest_gfq != NULL && src_gfq != NULL);
    P4EST_ASSERT(dest_gfq[0] == 0 && src_gfq[0] == 0);
    P4EST_ASSERT(mpicomm != sc_MPI_COMM_NULL);
    P4EST_ASSERT(mpisize != NULL && mpirank != NULL);

    mpiret = sc_MPI_Comm_size(mpicomm, mpisize);
    SC_CHECK_MPI(mpiret);
    mpiret = sc_MPI_Comm_rank(mpicomm, mpirank);
    SC_CHECK_MPI(mpiret);

    P4EST_ASSERT(dest_gfq[*mpisize] == src_gfq[*mpisize]);
    P4EST_ASSERT(0 <= dest_gfq[*mpirank] && dest_gfq[*mpirank] <= dest_gfq[*mpirank + 1] &&
                 dest_gfq[*mpirank + 1] <= dest_gfq[*mpisize]);
    P4EST_ASSERT(0 <= src_gfq[*mpirank] && src_gfq[*mpirank] <= src_gfq[*mpirank + 1] &&
                 src_gfq[*mpirank + 1] <= src_gfq[*mpisize]);
  }

  // ========================================================================================
  // ========================================================================================
  static typename p4est_t::transfer_context_t *
  transfer_fixed_begin(const p4est::gloidx_t * dest_gfq,
                       const p4est::gloidx_t * src_gfq,
                       sc_MPI_Comm             mpicomm,
                       int                     tag,
                       void *                  dest_data,
                       const void *            src_data,
                       size_t                  data_size)
  {
    typename p4est_t::transfer_context_t * tc;
    int                                    mpiret;
    int                                    mpisize, mpirank;
    int                                    q;
    int                                    first_sender, last_sender;
    int                                    first_receiver, last_receiver;
    char *                                 rb;
    char *                                 dest_cp, *src_cp;
    size_t                                 byte_len, cp_len;
    p4est::gloidx_t                        dest_begin, dest_end;
    p4est::gloidx_t                        src_begin, src_end;
    p4est::gloidx_t                        gbegin, gend;
    sc_MPI_Request *                       rq;

    /* setup context structure */
    tc = P4EST_ALLOC_ZERO(typename p4est_t::transfer_context_t, 1);
    tc->variable = 0;

    /* there is nothing to do when there is no data */
    if (data_size == 0)
    {
      return tc;
    }

    /* grab local partition information */
    transfer_assign_comm(dest_gfq, src_gfq, mpicomm, &mpisize, &mpirank);
    dest_begin = dest_gfq[mpirank];
    dest_end = dest_gfq[mpirank + 1];
    src_begin = src_gfq[mpirank];
    src_end = src_gfq[mpirank + 1];

    /* prepare data copy for local overlap */
    dest_cp = src_cp = NULL;
    cp_len = 0;

    /* figure out subset of processes to receive from */
    if (dest_begin < dest_end)
    {
      P4EST_ASSERT(dest_data != NULL);

      /* our process as the receiver is not empty */
      first_sender = p4est_bsearch_partition(dest_begin, src_gfq, mpisize);
      P4EST_ASSERT(0 <= first_sender && first_sender < mpisize);
      last_sender =
        p4est_t::bsearch_partition(dest_end - 1, &src_gfq[first_sender], mpisize - first_sender) +
        first_sender;
      P4EST_ASSERT(first_sender <= last_sender && last_sender < mpisize);
      tc->num_senders = last_sender - first_sender + 1;
      P4EST_ASSERT(tc->num_senders > 0);

      /* go through sender processes and post receive calls */
      gend = dest_begin;
      rq = tc->recv_req = P4EST_ALLOC(sc_MPI_Request, static_cast<size_t>(tc->num_senders));
      rb = reinterpret_cast<char *>(dest_data);
      for (q = first_sender; q <= last_sender; ++q)
      {
        /* prepare positions for the sender process q */
        gbegin = gend;
        gend = src_gfq[q + 1];
        if (gend > dest_end)
        {
          P4EST_ASSERT(q == last_sender);
          gend = dest_end;
        }
        P4EST_ASSERT(q == first_sender || q == last_sender ? gbegin < gend : gbegin <= gend);

        /* choose how to treat the sender process */
        if (gbegin == gend)
        {
          /* the sender process is empty; we need no message */
          P4EST_ASSERT(first_sender < q && q < last_sender);
          *rq++ = sc_MPI_REQUEST_NULL;
        }
        else
        {
          /* nonzero message from this sender */
          byte_len = static_cast<size_t>(gend - gbegin) * data_size;
          if (q == mpirank)
          {
            /* on the same rank we remember pointers for memcpy */
            cp_len = byte_len;
            dest_cp = rb;
            *rq++ = sc_MPI_REQUEST_NULL;
          }
          else
          {
            /* we receive a proper message */
            mpiret =
              sc_MPI_Irecv(rb, static_cast<int>(byte_len), sc_MPI_BYTE, q, tag, mpicomm, rq++);
            SC_CHECK_MPI(mpiret);
          }
          rb += byte_len;
        }
      }
      P4EST_ASSERT(rb - (char *)dest_data == (ptrdiff_t)((dest_end - dest_begin) * data_size));
    }

    /* figure out subset of processes to send to */
    if (src_begin < src_end)
    {
      P4EST_ASSERT(src_data != NULL);

      /* our process as the sender is not empty */
      first_receiver = p4est_t::bsearch_partition(src_begin, dest_gfq, mpisize);
      P4EST_ASSERT(0 <= first_receiver && first_receiver < mpisize);
      last_receiver = p4est_t::bsearch_partition(
                        src_end - 1, &dest_gfq[first_receiver], mpisize - first_receiver) +
                      first_receiver;
      P4EST_ASSERT(first_receiver <= last_receiver && last_receiver < mpisize);
      tc->num_receivers = last_receiver - first_receiver + 1;
      P4EST_ASSERT(tc->num_receivers > 0);

      /* go through receiver processes and post send calls */
      gend = src_begin;
      rq = tc->send_req = P4EST_ALLOC(sc_MPI_Request, static_cast<size_t>(tc->num_receivers));
      rb = const_cast<char *>(reinterpret_cast<const char *>(src_data));
      for (q = first_receiver; q <= last_receiver; ++q)
      {
        /* prepare positions for the receiver process q */
        gbegin = gend;
        gend = dest_gfq[q + 1];
        if (gend > src_end)
        {
          P4EST_ASSERT(q == last_receiver);
          gend = src_end;
        }
        P4EST_ASSERT(q == first_receiver || q == last_receiver ? gbegin < gend : gbegin <= gend);

        /* choose how to treat the receiver process */
        if (gbegin == gend)
        {
          /* the receiver process is empty; we need no message */
          P4EST_ASSERT(first_receiver < q && q < last_receiver);
          *rq++ = sc_MPI_REQUEST_NULL;
        }
        else
        {
          /* nonzero message for this receiver */
          byte_len = static_cast<size_t>(gend - gbegin) * data_size;
          if (q == mpirank)
          {
            /* on the same rank we remember pointers for memcpy */
            P4EST_ASSERT(cp_len == byte_len);
            src_cp = rb;
            *rq++ = sc_MPI_REQUEST_NULL;
          }
          else
          {
            /* we send a proper message */
            mpiret =
              sc_MPI_Isend(rb, static_cast<int>(byte_len), sc_MPI_BYTE, q, tag, mpicomm, rq++);
            SC_CHECK_MPI(mpiret);
          }
          rb += byte_len;
        }
      }
      P4EST_ASSERT(rb - (char *)src_data == (ptrdiff_t)((src_end - src_begin) * data_size));
    }

    /* copy the data that remains local */
    P4EST_ASSERT((dest_cp == NULL) == (src_cp == NULL));
    if (cp_len > 0)
    {
      P4EST_ASSERT(dest_cp != NULL && src_cp != NULL);
      if constexpr (std::is_same<MemorySpace, Kokkos::HostSpace>::value)
      {
        memcpy(dest_cp, src_cp, cp_len);
      }
#if defined(KOKKOS_ENABLE_CUDA)
      else if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>)
      {
        cudaMemcpy(dest_cp, src_cp, cp_len, cudaMemcpyDeviceToDevice);
      }
#endif
#if defined(KOKKOS_ENABLE_HIP)
      else if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace>)
      {
        hipMemcpy(dest_cp, src_cp, cp_len, hipMemcpyDeviceToDevice);
      }
#endif
    }

    /* the rest goes into the p4est_transfer_fixed_end function */
    return tc;

  } // kalypsso_transfer_fixed_begin

  // ========================================================================================
  // ========================================================================================
  static typename p4est_t::transfer_context_t *
  transfer_fixed_multi_var_begin(const p4est::gloidx_t * dest_gfq,
                                 const p4est::gloidx_t * src_gfq,
                                 sc_MPI_Comm             mpicomm,
                                 int                     tag,
                                 void *                  dest_data,
                                 const void *            src_data,
                                 size_t                  data_size,
                                 const uint32_t *        src_offsets,
                                 const uint32_t *        dst_offsets)
  {
    typename p4est_t::transfer_context_t * tc;
    int                                    mpiret;
    int                                    mpisize, mpirank;
    int                                    q;
    int                                    first_sender, last_sender;
    int                                    first_receiver, last_receiver;
    char *                                 rb;
    char *                                 dest_cp, *src_cp;
    size_t                                 byte_len, cp_len;
    p4est::gloidx_t                        dest_begin, dest_end;
    p4est::gloidx_t                        src_begin, src_end;
    p4est::gloidx_t                        gbegin, gend;
    sc_MPI_Request *                       rq;

    /* setup context structure */
    tc = P4EST_ALLOC_ZERO(typename p4est_t::transfer_context_t, 1);
    tc->variable = 0;

    /* there is nothing to do when there is no data */
    if (data_size == 0)
    {
      return tc;
    }

    /* grab local partition information */
    transfer_assign_comm(dest_gfq, src_gfq, mpicomm, &mpisize, &mpirank);
    dest_begin = dest_gfq[mpirank]; /* this essentially correspond to the 0 'i_oct' of dst */
    dest_end = dest_gfq[mpirank + 1];
    src_begin = src_gfq[mpirank]; /* this essentially correspond to the 0 'i_oct' of src */
    src_end = src_gfq[mpirank + 1];

    /* prepare data copy for local overlap */
    dest_cp = src_cp = NULL;
    cp_len = 0;

    /* figure out subset of processes to receive from */
    if (dest_begin < dest_end)
    {
      P4EST_ASSERT(dest_data != NULL);

      /* our process as the receiver is not empty */
      first_sender = p4est_bsearch_partition(dest_begin, src_gfq, mpisize);
      P4EST_ASSERT(0 <= first_sender && first_sender < mpisize);
      last_sender =
        p4est_t::bsearch_partition(dest_end - 1, &src_gfq[first_sender], mpisize - first_sender) +
        first_sender;
      P4EST_ASSERT(first_sender <= last_sender && last_sender < mpisize);
      tc->num_senders = last_sender - first_sender + 1;
      P4EST_ASSERT(tc->num_senders > 0);

      /* go through sender processes and post receive calls */
      gend = dest_begin;
      rq = tc->recv_req = P4EST_ALLOC(sc_MPI_Request, static_cast<size_t>(tc->num_senders));
      rb = reinterpret_cast<char *>(dest_data);
      for (q = first_sender; q <= last_sender; ++q)
      {
        /* prepare positions for the sender process q */
        gbegin = gend;
        gend = src_gfq[q + 1];
        if (gend > dest_end)
        {
          P4EST_ASSERT(q == last_sender);
          gend = dest_end;
        }
        P4EST_ASSERT(q == first_sender || q == last_sender ? gbegin < gend : gbegin <= gend);

        /* choose how to treat the sender process */
        if (gbegin == gend)
        {
          /* the sender process is empty; we need no message */
          P4EST_ASSERT(first_sender < q && q < last_sender);
          *rq++ = sc_MPI_REQUEST_NULL;
        }
        else
        {
          /* nonzero message from this sender */
          byte_len =
            static_cast<size_t>(dst_offsets[gend - dest_begin] - dst_offsets[gbegin - dest_begin]) *
            data_size;
          if (q == mpirank)
          {
            /* on the same rank we remember pointers for memcpy */
            cp_len = byte_len;
            dest_cp = rb;
            *rq++ = sc_MPI_REQUEST_NULL;
          }
          else
          {
            /* we receive a proper message */
            mpiret =
              sc_MPI_Irecv(rb, static_cast<int>(byte_len), sc_MPI_BYTE, q, tag, mpicomm, rq++);
            SC_CHECK_MPI(mpiret);
          }
          rb += byte_len;
        }
      }
      P4EST_ASSERT(rb - (char *)dest_data ==
                   (ptrdiff_t)(dst_offsets[dest_end - dest_begin] * data_size));
    }

    /* figure out subset of processes to send to */
    if (src_begin < src_end)
    {
      P4EST_ASSERT(src_data != NULL);

      /* our process as the sender is not empty */
      first_receiver = p4est_t::bsearch_partition(src_begin, dest_gfq, mpisize);
      P4EST_ASSERT(0 <= first_receiver && first_receiver < mpisize);
      last_receiver = p4est_t::bsearch_partition(
                        src_end - 1, &dest_gfq[first_receiver], mpisize - first_receiver) +
                      first_receiver;
      P4EST_ASSERT(first_receiver <= last_receiver && last_receiver < mpisize);
      tc->num_receivers = last_receiver - first_receiver + 1;
      P4EST_ASSERT(tc->num_receivers > 0);

      /* go through receiver processes and post send calls */
      gend = src_begin;
      rq = tc->send_req = P4EST_ALLOC(sc_MPI_Request, static_cast<size_t>(tc->num_receivers));
      rb = const_cast<char *>(reinterpret_cast<const char *>(src_data));
      for (q = first_receiver; q <= last_receiver; ++q)
      {
        /* prepare positions for the receiver process q */
        gbegin = gend;
        gend = dest_gfq[q + 1];
        if (gend > src_end)
        {
          P4EST_ASSERT(q == last_receiver);
          gend = src_end;
        }
        P4EST_ASSERT(q == first_receiver || q == last_receiver ? gbegin < gend : gbegin <= gend);

        /* choose how to treat the receiver process */
        if (gbegin == gend)
        {
          /* the receiver process is empty; we need no message */
          P4EST_ASSERT(first_receiver < q && q < last_receiver);
          *rq++ = sc_MPI_REQUEST_NULL;
        }
        else
        {
          /* nonzero message for this receiver */
          byte_len =
            static_cast<size_t>(src_offsets[gend - src_begin] - src_offsets[gbegin - src_begin]) *
            data_size;
          if (q == mpirank)
          {
            /* on the same rank we remember pointers for memcpy */
            P4EST_ASSERT(cp_len == byte_len);
            src_cp = rb;
            *rq++ = sc_MPI_REQUEST_NULL;
          }
          else
          {
            /* we send a proper message */
            mpiret =
              sc_MPI_Isend(rb, static_cast<int>(byte_len), sc_MPI_BYTE, q, tag, mpicomm, rq++);
            SC_CHECK_MPI(mpiret);
          }
          rb += byte_len;
        }
      }
      P4EST_ASSERT(rb - (char *)src_data ==
                   (ptrdiff_t)(src_offsets[src_end - src_begin] * data_size));
    }

    /* copy the data that remains local */
    P4EST_ASSERT((dest_cp == NULL) == (src_cp == NULL));
    if (cp_len > 0)
    {
      P4EST_ASSERT(dest_cp != NULL && src_cp != NULL);
      if constexpr (std::is_same<MemorySpace, Kokkos::HostSpace>::value)
      {
        memcpy(dest_cp, src_cp, cp_len);
      }
#if defined(KOKKOS_ENABLE_CUDA)
      else if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>)
      {
        cudaMemcpy(dest_cp, src_cp, cp_len, cudaMemcpyDeviceToDevice);
      }
#endif
#if defined(KOKKOS_ENABLE_HIP)
      else if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace>)
      {
        hipMemcpy(dest_cp, src_cp, cp_len, hipMemcpyDeviceToDevice);
      }
#endif
    }

    /* the rest goes into the p4est_transfer_fixed_end function */
    return tc;

  } // transfer_fixed_multi_var_begin

}; // struct MeshPartitioner_helper

} // namespace kalypsso
