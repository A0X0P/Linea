function(add_linea_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE Linea)

    # Force console subsystem on MinGW
    if (MINGW)
        target_link_options(${name} PRIVATE -mconsole)
    endif()
endfunction()

add_linea_example(matrix_basics)
#add_linea_example(matrix_math)
add_linea_example(vector_basics)
#add_linea_example(vector_math)
