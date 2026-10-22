add_executable(tensorica_tests
    tensor_smoke.cpp
)

target_link_libraries(tensorica_tests
    PRIVATE
        Tensorica::Tensorica
)

add_test(NAME tensorica_tests COMMAND tensorica_tests)
