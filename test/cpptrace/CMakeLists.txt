add_executable(test_cpptrace "")
target_sources(test_cpptrace PRIVATE test_cpptrace.cpp)
target_link_libraries(test_cpptrace PUBLIC cpptrace::cpptrace kalypsso::core)
