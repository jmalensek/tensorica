add_executable(tensorica_smoke_tests
    tensor_smoke.cpp
)

target_link_libraries(tensorica_smoke_tests
    PRIVATE
        Tensorica::Tensorica
)

add_test(NAME tensorica_smoke_tests COMMAND tensorica_smoke_tests)

add_executable(tensorica_arithmetic_tests
    arithmetic_test.cpp
)

target_link_libraries(tensorica_arithmetic_tests
    PRIVATE
        Tensorica::Tensorica
)

add_test(NAME tensorica_arithmetic_tests COMMAND tensorica_arithmetic_tests)
