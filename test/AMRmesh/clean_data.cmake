file(GLOB files_to_remove "${CMAKE_CURRENT_BINARY_DIR}/*[.h5,.xmf,.npy]")
message("removing ${files_to_remove}")
file(REMOVE ${files_to_remove})
