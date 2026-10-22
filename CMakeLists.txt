cmake_minimum_required(VERSION 3.21)

project(
    Tensorica
    VERSION 0.1.0
    DESCRIPTION "Modern C++ tensor mathematics library"
    LANGUAGES CXX
)

# ------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------

file(GLOB_RECURSE TENSORICA_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

if(TENSORICA_SOURCES)
    add_library(Tensorica ${TENSORICA_SOURCES})
    set(TENSORICA_API_SCOPE PUBLIC)
else()
    message(STATUS "No sources found in src/. Configuring Tensorica as a header-only INTERFACE target.")
    add_library(Tensorica INTERFACE)
    set(TENSORICA_API_SCOPE INTERFACE)
endif()

add_library(Tensorica::Tensorica ALIAS Tensorica)

target_compile_features(Tensorica ${TENSORICA_API_SCOPE} cxx_std_20)

target_include_directories(Tensorica
    ${TENSORICA_API_SCOPE}
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(TENSORICA_SOURCES)
    target_compile_options(Tensorica
        PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/W4>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
    )

    set_target_properties(Tensorica PROPERTIES
        CXX_EXTENSIONS OFF
    )
endif()

option(TENSORICA_ENABLE_SIMD "Enable SIMD optimizations" ON)

target_compile_definitions(Tensorica
    ${TENSORICA_API_SCOPE}
        $<$<BOOL:${TENSORICA_ENABLE_SIMD}>:TENSORICA_ENABLE_SIMD>
)

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------

include(CTest)

if(BUILD_TESTING)
    if(EXISTS "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    else()
        message(STATUS "BUILD_TESTING=ON, but tests/CMakeLists.txt was not found. Add it to register tests.")
    endif()
endif()