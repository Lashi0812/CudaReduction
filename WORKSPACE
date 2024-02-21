load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
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

##############################################################################
##########################  bazel skylib          ############################
##############################################################################

load("//tools/bazel_skylib:workspace.bzl", bazel_skylib = "repo")

bazel_skylib()

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

##############################################################################
##########################  rules_cuda           ############################
##############################################################################

load("//third_party/rules_cuda:workspace.bzl", rules_cuda_workspace = "repo")

rules_cuda_workspace()

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")

rules_cuda_dependencies()

register_detected_cuda_toolchains()

##############################################################################
##########################  cutlass               ############################
##############################################################################

load("//third_party/cutlass:workspace.bzl", cutlass = "repo")

cutlass()

# # using nvtx for profile
# load("//tools/com_google_benchmark:workspace.bzl", com_google_benchmark = "repo")
# com_google_benchmark()

##############################################################################
##########################  nlohmann_json         ############################
##############################################################################

git_repository(
    name = "nlohmann_json",
    commit = "0457de21cffb298c22b629e538036bfeb96130b7",
    remote = "https://github.com/nlohmann/json",
)

##############################################################################
##########################  fmt         ############################
##############################################################################

git_repository(
    name = "fmt",
    commit = "0166f455f6681144a18553d2ea0cda8946bff019",
    patch_cmds = [
        "mv support/bazel/.bazelversion .bazelversion",
        "mv support/bazel/BUILD.bazel BUILD.bazel",
        "mv support/bazel/WORKSPACE.bazel WORKSPACE.bazel",
    ],
    remote = "https://github.com/fmtlib/fmt",
)

##############################################################################
##########################  nvbench               ############################
##############################################################################

load("//tools/nvbench:workspace.bzl", nvbench = "repo")

nvbench()

##############################################################################
##########################  compile_command_extractor         ################
##############################################################################

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
