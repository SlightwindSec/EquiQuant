import shutil
import sys
from importlib import metadata
from importlib.metadata import PackageNotFoundError


def get_pkg_version(package_name):
    """
    Checks the version of a pip package using importlib.metadata.
    Returns a string with status and version.
    """
    try:
        version = metadata.version(package_name)
        return f"Installed (Version: {version})"
    except PackageNotFoundError:
        return "Not Installed"
    except Exception as e:
        return f"Error checking: {e}"


def check_shell_command(command_name):
    """
    Checks if a shell command exists in the PATH using shutil.which.
    Returns a string with status and path.
    """
    path = shutil.which(command_name)
    if path:
        return f"Found (Path: {path})"
    else:
        return "Command not found"


def check_requirements():
    """
    Main function to execute all checks and print a formatted report.
    """
    print("--- Environment Dependency Check Report---")
    check_pass = True
    items_to_check = [
        ("shell", "msmodelslim", "ModelSlim Shell Tool"),
        ("pip", "vllm", "VLLM Framework"),
        ("pip", "vllm-ascend", "VLLM Ascend"),
        ("pip", "torch", "PyTorch Framework"),
        ("pip", "torch-npu", "PyTorch NPU Adapter"),
        ("pip", "ais-bench-benchmark", "AIS Bench Benchmark Tool"),
    ]
    results = {}
    max_name_len = max(len(name) for _, name, _ in items_to_check)
    print("\n[ Running checks... ]\n")

    py_version = sys.version.split()[0]
    print(f"{'python'.ljust(max_name_len + 2)} : Installed (Version: {py_version})\n")

    for item_type, name, description in items_to_check:
        print(f"Checking: {description} ({name})...")
        if item_type == "shell":
            results[name] = check_shell_command(name)
        elif item_type == "pip":
            results[name] = get_pkg_version(name)
        if "Command not found" in results[name]:
            check_pass = False

    print("\n--- Summary of Results ---")
    for item_type, name, _ in items_to_check:
        status = results.get(name, " Check not performed")
        print(f"{name.ljust(max_name_len + 2)} : {status}")
    print("--- Check complete ---")
    return check_pass
