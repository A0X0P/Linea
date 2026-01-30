cmake_minimum_required(VERSION 3.16)

project(Linea
    VERSION 0.1.0
    DESCRIPTION "Linea: Header-only Linear Algebra Library in Modern C++"
    LANGUAGES CXX
)

# ------------------------------------------------------------
# Toolchain safety (MinGW console executables)
# ------------------------------------------------------------
if (MINGW)
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -mconsole"
    )
endif()

# ------------------------------------------------------------
# Header-only library target
# ------------------------------------------------------------
add_library(Linea INTERFACE)

# Public include directory
target_include_directories(Linea
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Require C++20 (concepts)
target_compile_features(Linea
    INTERFACE
        cxx_std_20
)

# ------------------------------------------------------------
# Examples
# ------------------------------------------------------------
add_subdirectory(examples)

# ------------------------------------------------------------
# (Optional) Tests placeholder
# ------------------------------------------------------------
# add_subdirectory(tests)
