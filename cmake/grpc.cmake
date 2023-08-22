option(DSPACES_USE_GRPC "Use gRPC for RPC communication" OFF)
if(DSPACES_USE_GRPC)
    add_compile_definitions(DSPACES_USE_GRPC)
    include(FetchContent)
    set(ABSL_ENABLE_INSTALL ON)
    set(ABSL_PROPAGATE_CXX_STD ON)
    set(gRPC_INSTALL ON)
    FetchContent_Declare(
        gRPC
        GIT_REPOSITORY https://github.com/grpc/grpc
        GIT_TAG        v1.56.0
    )
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_MakeAvailable(gRPC)
endif()

