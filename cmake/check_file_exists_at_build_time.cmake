# check_file_exists_at_build_time
function(check_file_exists_at_build_time the_target file_to_check)

  add_custom_command(
    TARGET ${the_target}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -DFileToCheck=${file_to_check} -P
            ${CMAKE_SOURCE_DIR}/cmake/check_file_exists.cmake
    COMMENT "Checking if ${file_to_check} exists...")

endfunction()
