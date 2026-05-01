// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_kokkos_teamvector.cpp
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/kalypsso_core_config.h>
#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/utils/mpi/GlobalMpiSession.h>
#  include <mpi.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <iostream>

template <typename device_t>
struct TestFunctorBase
{
  using Data_t = Kokkos::View<int32_t *, device_t>;
  using DataHost_t = typename Data_t::HostMirror;
};

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Kokkos team vector - parallel for
 *
 * \note in this test, data array size must be a multiple of nbBlocks
 */
template <typename device_t>
class TestKokkosTeamVectorForFunctor : public TestFunctorBase<device_t>
{

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using exec_space_t = typename device_t::execution_space;
  using Data_t = typename TestFunctorBase<device_t>::Data_t;
  using team_policy_t = Kokkos::TeamPolicy<exec_space_t>;
  using thread_t = typename team_policy_t::member_type;

  void
  setNbTeams(uint32_t nbTeams_)
  {
    nbTeams = nbTeams_;
  }

  /**
   * test parallel for functor
   *
   */
  TestKokkosTeamVectorForFunctor(Data_t data, uint32_t bSize)
    : data(data)
    , bSize(bSize)
  {
    nbBlocks = (data.extent(0) + bSize - 1) / bSize;
  }

  // static method which does it all: create and execute functor
  static void
  apply(Data_t data, uint32_t bSize)
  {

    TestKokkosTeamVectorForFunctor functor(data, bSize);

    // kokkos execution policy
    uint32_t nbTeams_ = 16;
    functor.setNbTeams(nbTeams_);

    team_policy_t policy(nbTeams_, Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_for("TestKokkosTeamVectorForFunctor", policy, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(thread_t member) const
  {

    uint32_t iBlock = member.league_rank();

    while (iBlock < nbBlocks)
    {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, bSize), [&](const int32_t index) {
        // copy q state in q global
        data(index + iBlock * bSize) += 12;
      }); // end TeamVectorRange

      iBlock += nbTeams;

    } // end while iBlock < nbBlocks

  } // operator

  //! heavy data
  Data_t data;

  //! block size
  uint32_t bSize;

  //! number of blocks
  uint32_t nbBlocks;

}; // TestKokkosTeamVectorForFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Kokkos team vector - parallel reduce inside block
 *
 * \note in this test, data array size must be a multiple of nbBlocks
 */
template <typename device_t>
class TestKokkosTeamVectorReduceFunctor : public TestFunctorBase<device_t>
{

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using exec_space_t = typename device_t::execution_space;
  using Data_t = typename TestFunctorBase<device_t>::Data_t;
  using team_policy_t = Kokkos::TeamPolicy<exec_space_t>;
  using thread_t = typename team_policy_t::member_type;

  void
  setNbTeams(uint32_t nbTeams_)
  {
    nbTeams = nbTeams_;
  }

  /**
   * test parallel reduce functor
   *
   */
  TestKokkosTeamVectorReduceFunctor(Data_t data, uint32_t bSize)
    : data(data)
    , bSize(bSize)
  {
    nbBlocks = (data.extent(0) + bSize - 1) / bSize;
  }

  // static method which does it all: create and execute functor
  static void
  apply(Data_t data, uint32_t bSize)
  {

    TestKokkosTeamVectorReduceFunctor functor(data, bSize);

    // kokkos execution policy
    uint32_t nbTeams_ = 16;
    functor.setNbTeams(nbTeams_);

    team_policy_t policy(nbTeams_, Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_for("TestKokkosTeamVectorReduceFunctor", policy, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(thread_t member) const
  {

    uint32_t iBlock = member.league_rank();

    while (iBlock < nbBlocks)
    {

      int sum = 0;

      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, bSize),
        [&](const int32_t index, int & local_sum) {
          // copy q state in q global
          local_sum += data(index + iBlock * bSize);
        },
        sum); // end TeamVectorRange

      // check results :
      int32_t diff = sum - (bSize - 1) * bSize / 2 - iBlock * bSize * bSize;
      bool    valid = diff == 0 ? true : false;

      if (member.team_rank() == 0)
      {
        // std::cout << iBlock << ": res=" << sum << " results valid ? " << valid << "\n";
        printf("%d: res=%d results valid ? %d\n", iBlock, sum, valid);
      }

      iBlock += nbTeams;

    } // end while iBlock < nbBlocks

  } // operator

  //! heavy data
  Data_t data;

  //! block size
  uint32_t bSize;

  //! number of blocks
  uint32_t nbBlocks;

}; // TestKokkosTeamVectorReduceFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Kokkos team vector - parallel reduce.
 *
 * Do a hierarchical reduce, i.e. first reduce inside a team, then a global reduce.
 *
 * \note in this test, data array size must be a multiple of nbBlocks
 */
template <typename device_t>
class TestKokkosTeamVectorReduceFunctor2 : public TestFunctorBase<device_t>
{

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using exec_space_t = typename device_t::execution_space;
  using Data_t = typename TestFunctorBase<device_t>::Data_t;
  using team_policy_t = Kokkos::TeamPolicy<exec_space_t>;
  using thread_t = typename team_policy_t::member_type;

  void
  setNbTeams(uint32_t nbTeams_)
  {
    nbTeams = nbTeams_;
  }

  /**
   * test parallel reduce functor
   */
  TestKokkosTeamVectorReduceFunctor2(Data_t data, uint32_t bSize)
    : data(data)
    , bSize(bSize)
  {
    nbBlocks = (data.extent(0) + bSize - 1) / bSize;
  }

  // static method which does it all: create and execute functor
  static void
  apply(Data_t data, uint32_t bSize)
  {

    TestKokkosTeamVectorReduceFunctor2 functor(data, bSize);

    // kokkos execution policy
    uint32_t nbTeams_ = 16;
    functor.setNbTeams(nbTeams_);

    team_policy_t policy(nbTeams_, Kokkos::AUTO() /* team size chosen by kokkos */);

    // initialize reduce value to something really small
    Kokkos::MaxLoc<int32_t, int32_t>::value_type result;

    Kokkos::parallel_reduce("TestKokkosTeamVectorReduceFunctor2",
                            policy,
                            functor,
                            Kokkos::MaxLoc<int32_t, int32_t>(result));

    printf("maximum value is %d at location %d\n", result.val, result.loc);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(thread_t member, Kokkos::MaxLoc<int32_t, int32_t>::value_type & max_value) const
  {

    uint32_t iBlock = member.league_rank();
    uint32_t index = member.team_rank();

    auto result = max_value;

    while (iBlock < nbBlocks)
    {

      while (index < bSize)
      {

        // printf("[debug] index=%d iBlock=%d bSize=%d | data=%d\n",index,iBlock,bSize,data(index +
        // iBlock * bSize));

        // update result
        if (data(index + iBlock * bSize) > result.val)
        {
          result.val = data(index + iBlock * bSize);
          result.loc = index + iBlock * bSize;
        }

        index += member.team_size();
      }

      iBlock += nbTeams;

    } // end while iBlock < nbBlocks

    // update global reduced value
    if (max_value.val < result.val)
    {
      max_value.val = result.val;
      max_value.loc = result.loc;
    }

  } // operator

  //! heavy data
  Data_t data;

  //! block size
  uint32_t bSize;

  //! number of blocks
  uint32_t nbBlocks;

}; // TestKokkosTeamVectorReduceFunctor2

// =======================================================================
// =======================================================================
template <typename device_t>
void
run_test(uint32_t bSize, uint32_t nbBlocks)
{

  using exec_space = typename device_t::execution_space;
  using Data_t = typename TestFunctorBase<device_t>::Data_t;

  /*
   * TestKokkosTeamVectorForFunctor
   */
  {
    std::cout << "// ======================================\n";
    std::cout << "Testing TestKokkosTeamVectorForFunctor...\n";
    std::cout << "// ======================================\n";

    uint32_t dataSize = bSize * nbBlocks;

    // create and init test data
    Data_t data = Data_t("test_data", dataSize);
    Kokkos::parallel_for(
      "init_test_data", Kokkos::RangePolicy<exec_space>(0, dataSize), KOKKOS_LAMBDA(uint32_t i) {
        data(i) = i;
      });

    // Kokkos::fence();
    // Kokkos::parallel_for("print_results", Kokkos::RangePolicy<exec_space>(0, dataSize),
    //   KOKKOS_LAMBDA(uint32_t i) {
    //                        std::cout << i << " " << data(i) << "\n";
    //   });

    TestKokkosTeamVectorForFunctor<device_t>::apply(data, bSize);

    Kokkos::fence();

    Kokkos::parallel_for(
      "print_results", Kokkos::RangePolicy<exec_space>(0, dataSize), KOKKOS_LAMBDA(uint32_t i) {
        // std::cout << i << " " << data(i) << "\n";
        printf("%d %d\n", i, data(i));
      });
  }

  /*
   * TestKokkosTeamVectorReduceFunctor
   */
  {
    std::cout << "// ======================================\n";
    std::cout << "Testing TestKokkosTeamVectorReduceFunctor...\n";
    std::cout << "// ======================================\n";

    uint32_t dataSize = bSize * nbBlocks;

    // create and init test data
    Data_t data = Data_t("test_data", dataSize);
    Kokkos::parallel_for(
      "init_test_data", Kokkos::RangePolicy<exec_space>(0, dataSize), KOKKOS_LAMBDA(uint32_t i) {
        data(i) = i;
      });

    // Kokkos::fence();
    // Kokkos::parallel_for("print_results", Kokkos::RangePolicy<exec_space>(0, dataSize),
    //   KOKKOS_LAMBDA(uint32_t i) {
    //                        std::cout << i << " " << data(i) << "\n";
    //   });

    TestKokkosTeamVectorReduceFunctor<device_t>::apply(data, bSize);
  }

  /*
   * TestKokkosTeamVectorReduceFunctor2 - this i a "max" reduction
   */
  {
    std::cout << "// ======================================\n";
    std::cout << "Testing TestKokkosTeamVectorReduceFunctor2...\n";
    std::cout << "// ======================================\n";

    uint32_t dataSize = bSize * nbBlocks;

    // create and init test data
    Data_t data = Data_t("test_data", dataSize);
    Kokkos::parallel_for(
      "init_test_data", Kokkos::RangePolicy<exec_space>(0, dataSize), KOKKOS_LAMBDA(uint32_t i) {
        data(i) = 12 - (i - 13) * (i - 15);
      });

    TestKokkosTeamVectorReduceFunctor2<device_t>::apply(data, bSize);
  }

} // run_test

// =======================================================================
// =======================================================================
int
main(int argc, char * argv[])
{

  // Create MPI session if MPI enabled
#ifdef KALYPSSO_CORE_USE_MPI
  kalypsso::GlobalMpiSession mpiSession(argc, argv);
#endif // KALYPSSO_CORE_USE_MPI

  Kokkos::initialize(argc, argv);

  [[maybe_unused]] int rank = 0;
  [[maybe_unused]] int nRanks = 1;

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";

    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if (Kokkos::hwloc::available())
    {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
          << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
          << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";

#ifdef KALYPSSO_CORE_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#  ifdef KOKKOS_ENABLE_CUDA
    {

      // To enable kokkos accessing multiple GPUs don't forget to
      // add option "--ndevices=X" where X is the number of GPUs
      // you want to use per node.

      // on a large cluster, the scheduler should assign resources
      // in a way that each MPI task is mapped to a different GPU
      // let's cross-checked that:

      int cudaDeviceId;
      cudaGetDevice(&cudaDeviceId);
      std::cout << "I'm MPI task #" << rank << " (out of " << nRanks << ")" << " pinned to GPU #"
                << cudaDeviceId << "\n";
    }
#  endif // KOKKOS_ENABLE_CUDA
#endif   // KALYPSSO_CORE_USE_MPI
  } // end kokkos config

  uint32_t bSize = 4;
  uint32_t nbBlocks = 32;

  using DefaultDevice =
    Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

  run_test<DefaultDevice>(bSize, nbBlocks);

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
