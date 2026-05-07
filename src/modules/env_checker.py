import shutil
import sys
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from typing import Tuple, List, Literal


def check_requirements(quantizer: Literal["msmodelslim", "llmcompressor"]) -> bool:
    """Check all environment dependencies.

    Args:
        quantizer: The quantization tool to use.
    Returns:
        bool: True if all dependencies are met, False otherwise.
       """
    print(f"\n{'=' * 20} Environment Dependency Check Report \n{'=' * 20}")
    items_to_check = _items_to_check(quantizer)
    check_pass = True
    results = {}
    max_name_len = max(len(name) for _, name, _ in items_to_check)

    # check required packages
    print("\n[ Running checks ]\n")
    for item_type, name, description in items_to_check:
        print(f"Checking: {description} ({name})...")

        check_func = _check_shell_command if item_type == "shell" else _check_pip_version
        status_message, success = check_func(name)
        results[name] = status_message
        if not success:
            check_pass = False

    # print results
    print("\n[ Summary of Results ]")
    py_version = sys.version.split()[0]
    print(f"{'python'.ljust(max_name_len + 2)} : Installed (Version: {py_version})\n")
    for item_type, name, _ in items_to_check:
        status = results.get(name, " Check not performed")
        print(f"{name.ljust(max_name_len + 2)} : {status}")

    return check_pass

def _items_to_check(quantizer: Literal["msmodelslim", "llmcompressor"]) -> List[Tuple[str, str, str]]:
    """Get the items to check for the specified quantizer.

    Args:
        quantizer: The quantization tool to use.
    Returns:
        List[Tuple[str, str, str]]: A list of tuples with item type, name, and description.
       """
    base_packages = [
        ("pip", "vllm", "VLLM Framework"),
        ("pip", "vllm-ascend", "VLLM Ascend"),
        ("pip", "torch", "PyTorch Framework"),
        ("pip", "torch-npu", "PyTorch NPU Adapter"),
        ("pip", "ais-bench-benchmark", "AIS Bench Benchmark Tool"),
    ]
    if quantizer == "msmodelslim":
        additional_items = [
            ("shell", "msmodelslim", "ModelSlim Shell Tool"),
        ]
    elif quantizer == "llmcompressor":
        additional_items = [
            ("pip", "llmcompressor", "LLMCompressor Tool"),
        ]
    return base_packages + additional_items

def _check_pip_version(package_name: str) -> Tuple[str, bool]:
    """Check the version of a pip package using importlib.metadata.

    Args:
        package_name: The name of the pip package to check.
    Returns:
        Tuple[str, bool]: A tuple with a string with status and version, and a boolean indicating if the package is installed.
       """
    is_installed = False

    try:
        version = metadata.version(package_name)
        status_message = f"Installed (Version: {version})"
        is_installed = True
    except PackageNotFoundError:
        status_message = "Not Installed"
    except Exception as e:
        status_message = f"Error checking: {e}"

    return status_message, is_installed

def _check_shell_command(command_name: str) -> Tuple[str, bool]:
    """Check if a shell command exists in the PATH using shutil.which.

    Args:
        command_name: The name of the shell command to check.
    Returns:
        Tuple[str, bool]: A tuple with a string with status and path, and a boolean indicating if the command is found.
    """
    is_installed = False

    path = shutil.which(command_name)
    if path:
        status_message = f"Found (Path: {path})"
        is_installed = True
    else:
        status_message = "Command not found"

    return status_message, is_installed