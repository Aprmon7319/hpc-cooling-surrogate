import sys

def check_python_version():
    # Check for the required Python version
    required_major, required_minor = 3, 9

    if sys.version_info < (required_major, required_minor):
        sys.stderr.write(f"Error: RAPS requires Python {required_major}.{required_minor} or greater\n")
        sys.exit(1)
