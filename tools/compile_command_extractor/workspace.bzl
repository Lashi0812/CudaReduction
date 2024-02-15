"""
# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
"""

# load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party:repo.bzl", "tf_http_archive","tf_mirror_urls")

COMMIT = "0e990032f3c5a866e72615cf67e5ce22186dcb97"
SHA256 = "2b3ee8bba2df4542a508b0289727b031427162b4cd381850f89b406445c17578"

def repo():
    tf_http_archive(
        name = "hedron_compile_commands",
        urls = tf_mirror_urls("https://github.com/hedronvision/bazel-compile-commands-extractor/archive/{commit}.tar.gz".format(commit = COMMIT)),
        strip_prefix = "bazel-compile-commands-extractor-{commit}".format(commit = COMMIT),
        sha256 = SHA256,
        # patch_file = [
        #     "//tools/compile_command_extractor:gen_command.patch"
        # ]
    )
    
# def repo():
#     http_archive(
#         name = "hedron_compile_commands",
#         urls = ["https://github.com/hedronvision/bazel-compile-commands-extractor/archive/{commit}.tar.gz".format(commit = COMMIT)],
#         strip_prefix = "bazel-compile-commands-extractor-{commit}".format(commit = COMMIT),
#         sha256 = SHA256,
#     )
