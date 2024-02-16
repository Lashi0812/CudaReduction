load("//third_party/xla:workspace.bzl", xla_workspace = "repo")
xla_workspace()

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()

# # this can already available in local_cuda_config since it is part of cuda toolkit
# load("//third_party/cccl:workspace.bzl", cccl = "repo")
# cccl()

load("//third_party/cutlass:workspace.bzl", cutlass = "repo")
cutlass()

# # using nvtx for profile
# load("//tools/com_google_benchmark:workspace.bzl", com_google_benchmark = "repo")
# com_google_benchmark()

load("//tools/compile_command_extractor:workspace.bzl", hedron_workspace = "repo")
hedron_workspace()


load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()
load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")
hedron_compile_commands_setup_transitive()
load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")
hedron_compile_commands_setup_transitive_transitive()
load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")
hedron_compile_commands_setup_transitive_transitive_transitive()