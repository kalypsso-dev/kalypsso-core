// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file cpptrace_utils.cpp
 */
#include <kalypsso/core/cpptrace_utils.h>

namespace kalypsso
{

// =======================================================
// =======================================================
/**
 * This function is called by the signal handler and is responsible to transfer
 * stack frame data to the tracer program for printing.
 *
 * Communication between current process and tracer process is done via a unix pipe.
 */
void
do_signal_safe_trace(cpptrace::frame_ptr * buffer, std::size_t count)
{
  // Setup pipe and spawn child
  pipe_t                input_pipe;
  [[maybe_unused]] auto status = pipe(input_pipe.data);
  const pid_t           pid = fork();
  if (pid == -1)
  {
    const char *          fork_failure_message = "[cpptrace] fork() failed\n";
    [[maybe_unused]] auto write_status =
      write(STDERR_FILENO, fork_failure_message, strlen(fork_failure_message));
    return;
  }
  if (pid == 0)
  { // child
    dup2(input_pipe.io.read_end, STDIN_FILENO);
    close(input_pipe.io.read_end);
    close(input_pipe.io.write_end);

    // check if env variable CPPTRACE_STACK_PRINTER_EXE is set
    if ([[maybe_unused]] const char * env_value = std::getenv("CPPTRACE_STACK_PRINTER_EXE"))
    {
      execl(env_value, "kalypsso_stack_printer", nullptr);
    }
    else
    {
      execl(KALYPSSO_CORE_STACK_PRINTER_FULLPATH, "kalypsso_stack_printer", nullptr);
    }

    const char * exec_failure_message =
      "[cpptrace] exec(kalypsso_stack_printer) failed: Make sure the \"kalypsso_stack_printer\" "
      "executable is in "
      "the current working directory and the binary's permissions are correct.\n";
    [[maybe_unused]] auto write_status =
      write(STDERR_FILENO, exec_failure_message, strlen(exec_failure_message));
    _exit(1);
  }
  // Resolve to safe_object_frames and write those to the pipe
  for (std::size_t i = 0; i < count; i++)
  {
    cpptrace::safe_object_frame frame;
    cpptrace::get_safe_object_frame(buffer[i], &frame);
    [[maybe_unused]] auto write_status = write(input_pipe.io.write_end, &frame, sizeof(frame));
  }
  close(input_pipe.io.read_end);
  close(input_pipe.io.write_end);
  // Wait for child
  waitpid(pid, nullptr, 0);
}

// =======================================================
// =======================================================
/**
 * The signal handler will do two thing:
 *
 * - collect the stack trace data in a frame_ptr array named "buffer"
 * - pass buffer to do_signal_safe_trace
 */
void
segv_handler([[maybe_unused]] int         signo,
             [[maybe_unused]] siginfo_t * info,
             [[maybe_unused]] void *      context)
{
  // Print basic message
  const char *          message = "SIGSEGV occurred:\n";
  [[maybe_unused]] auto write_status = write(STDERR_FILENO, message, strlen(message));
  // Generate trace
  constexpr std::size_t N = 100;
  cpptrace::frame_ptr   buffer[N];
  std::size_t           count = cpptrace::safe_generate_raw_trace(buffer, N);
  do_signal_safe_trace(buffer, count);
  // Up to you if you want to exit or continue or whatever
  _exit(1);
}

// =======================================================
// =======================================================
void
install_segv_handler()
{
  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));

  action.sa_flags = 0;
  action.sa_sigaction = &kalypsso::segv_handler;
  if (sigaction(SIGSEGV, &action, NULL) == -1)
  {
    perror("sigaction");
    exit(EXIT_FAILURE);
  }
}

// =======================================================
// =======================================================
/**
 * The signal handler will do two thing:
 *
 * - collect the stack trace data in a frame_ptr array named "buffer"
 * - pass buffer to do_signal_safe_trace
 */
void
abort_handler([[maybe_unused]] int         signo,
              [[maybe_unused]] siginfo_t * info,
              [[maybe_unused]] void *      context)
{
  // Print basic message
  const char *          message = "SIGABRT occurred:\n";
  [[maybe_unused]] auto write_status = write(STDERR_FILENO, message, strlen(message));
  // Generate trace
  constexpr std::size_t N = 100;
  cpptrace::frame_ptr   buffer[N];
  std::size_t           count = cpptrace::safe_generate_raw_trace(buffer, N);
  do_signal_safe_trace(buffer, count);
  // Up to you if you want to exit or continue or whatever
  //_exit(1);
}

// =======================================================
// =======================================================
void
install_abort_handler()
{
  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));

  action.sa_flags = 0;
  action.sa_sigaction = &kalypsso::abort_handler;
  if (sigaction(SIGABRT, &action, NULL) == -1)
  {
    perror("sigaction");
    exit(EXIT_FAILURE);
  }
}

// =======================================================
// =======================================================
void
warmup_cpptrace()
{
  cpptrace::frame_ptr          buffer[10];
  [[maybe_unused]] std::size_t count = cpptrace::safe_generate_raw_trace(buffer, 10);
  cpptrace::safe_object_frame  frame;
  cpptrace::get_safe_object_frame(buffer[0], &frame);
}

// =======================================================
// =======================================================
void
cpptrace_initialize()
{
  cpptrace::absorb_trace_exceptions(false);
  cpptrace::register_terminate_handler();
  warmup_cpptrace();

  install_segv_handler();
  install_abort_handler();
}

} // namespace kalypsso
