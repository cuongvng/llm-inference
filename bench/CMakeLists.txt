add_executable(llm-devinfo src/devinfo_main.cc)
target_link_libraries(llm-devinfo PRIVATE llm_engine)
