# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Instantiate the TLT-pytorch docker container for developers."""

import argparse
import os
import subprocess
import sys

from utils import CI, DOCKER_ROOT, ROOT_DIR, RCFILE, TEST_MODULES
from utils import get_docker_command

STATIC_TESTS = [
    # test cases.
    "pylint --rcfile {}".format(os.path.join(DOCKER_ROOT, RCFILE)),
    "pydocstyle --ignore=D4,D107,D200,D203,D205,D210,D212,D213,D301,D400,D401",
    "flake8 --ignore=E24,W504,E501,C400,C403,C408,C409,C414,C413,C416,C417,C419"
]

def parse_command_line(args=sys.argv[1:]):
    """Parse command line args for running tests if needed."""
    parser = argparse.ArgumentParser(prog="run_tests_local", description="Simple script to run tests.")
    parser.add_argument("--tag", default=None, help="Tag to the local base docker image.", type=str)
    return vars(parser.parse_args(args))


def main(cl_args=None):
    """Simple function to run local tests."""
    try:
        args = parse_command_line(cl_args)

        manifest_file = os.path.join(ROOT_DIR, "docker/manifest.json")
        docker_command_prefix, docker_image = get_docker_command(manifest_file, args["tag"])

        test_command = []
        for test in STATIC_TESTS:
            for module in  TEST_MODULES:
                if "cv" in module:
                    submodules_to_test = [
                        os.path.join(module, item) 
                        for item in os.listdir(os.path.join(ROOT_DIR, module))
                        if item not in ["odise", "__pycache__"]
                    ]
                else:
                    submodules_to_test = [module]
                test_command.extend(["{} {}".format(test, submodule) for submodule in submodules_to_test])
        for command in test_command:
            launch_command = command
            if not CI:
                print(f"Running test in local environment. {command}")
                launch_command = '{} -v {}:{} {} {}'.format(
                    docker_command_prefix, ROOT_DIR, DOCKER_ROOT, docker_image, command
                )
            sys.stdout.flush()
            subprocess.check_call(launch_command, stdout=sys.stdout, stderr=sys.stdout, shell=True)
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        raise RuntimeError(f"Tests failed execution") from e
    except Exception as exc:
        raise RuntimeError(f"Tests failed with error {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
