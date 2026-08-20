#!/usr/bin/env bash
# Bootstrap the 2D hS-IGA release on a clean Ubuntu installation.
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_NAME=$(basename "$0")
PROJECT_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
INSTALL_SYSTEM_DEPS=0
SKIP_SYSTEM_DEPS=0
CHECK_ONLY=0
PYTHON_CMD=

info() {
    printf '[INFO] %s\n' "$*"
}

warn() {
    printf '[WARN] %s\n' "$*" >&2
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
Usage: ./setup.sh [options]

Install the 2D release on Ubuntu. By default, missing system dependencies,
including Python 3.10 and Pipenv, are installed automatically with sudo.

Options:
  --install-system-deps  Install or refresh all supported Ubuntu dependencies.
  --skip-system-deps     Do not install system packages; fail if prerequisites are absent.
  --check                Check an existing installation without modifying it.
  -h, --help             Show this help message.

Environment variables:
  PYTHON=/path/to/python3.10  Select the Python 3.10 interpreter.
USAGE
}

while (( $# > 0 )); do
    case "$1" in
        --install-system-deps)
            INSTALL_SYSTEM_DEPS=1
            ;;
        --skip-system-deps)
            SKIP_SYSTEM_DEPS=1
            ;;
        --check)
            CHECK_ONLY=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1 (run ./$SCRIPT_NAME --help)"
            ;;
    esac
    shift
done

(( INSTALL_SYSTEM_DEPS == 0 || SKIP_SYSTEM_DEPS == 0 )) || \
    die "--install-system-deps and --skip-system-deps cannot be used together."

cd "$PROJECT_ROOT"
[[ -f Pipfile && -f Pipfile.lock ]] || die "Pipfile and Pipfile.lock are required."
[[ -f main.py ]] || die "main.py is missing; run this script from the 2D repository root."

run_as_root() {
    if (( EUID == 0 )); then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        die "Administrator privileges are required to install missing Ubuntu packages, but sudo is unavailable."
    fi
}

require_ubuntu() {
    [[ -r /etc/os-release ]] || die "Automatic setup currently supports Ubuntu only."
    # shellcheck disable=SC1091
    . /etc/os-release
    [[ "${ID:-}" == "ubuntu" ]] || die "Automatic setup currently supports Ubuntu only (detected ${ID:-unknown})."
}

python310_available() {
    command -v python3.10 >/dev/null 2>&1 && python3.10 -m venv --help >/dev/null 2>&1
}

system_dependencies_missing() {
    ! python310_available || ! command -v pipenv >/dev/null 2>&1
}

install_system_dependencies() {
    command -v apt-get >/dev/null 2>&1 || die "apt-get is required for automatic setup."
    require_ubuntu

    info "Installing Ubuntu prerequisites (Python 3.10, Pipenv, and virtual-environment support)."
    run_as_root apt-get update
    run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates git python3-pip software-properties-common
    run_as_root add-apt-repository --yes universe
    run_as_root apt-get update

    if ! apt-cache show python3.10 >/dev/null 2>&1; then
        info "Python 3.10 is not provided by the configured Ubuntu repositories; enabling deadsnakes PPA."
        run_as_root add-apt-repository --yes ppa:deadsnakes/ppa
        run_as_root apt-get update
    fi

    run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.10 python3.10-venv pipenv
}

resolve_python() {
    local version

    if [[ -v PYTHON ]]; then
        PYTHON_CMD="$PYTHON"
    else
        PYTHON_CMD=$(command -v python3.10 || true)
    fi

    [[ -n "$PYTHON_CMD" ]] || die "Python 3.10 is unavailable. Run ./$SCRIPT_NAME without --skip-system-deps."
    [[ -x "$PYTHON_CMD" ]] || die "Configured Python interpreter is not executable: $PYTHON_CMD"
    version=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    [[ "$version" == "3.10" ]] || die "Python 3.10 is required by Pipfile.lock; selected interpreter is Python $version."
}

check_installation() {
    local failures=0

    command -v pipenv >/dev/null 2>&1 || { warn "pipenv is not on PATH."; failures=1; }
    [[ -x .venv/bin/python ]] || { warn "Missing project environment: $PROJECT_ROOT/.venv"; failures=1; }
    if [[ -x .venv/bin/python ]]; then
        [[ "$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.10" ]] || {
            warn "The project environment does not use Python 3.10."
            failures=1
        }
    fi

    (( failures == 0 )) || die "Installation check failed. Run ./$SCRIPT_NAME to install missing components."
    info "Installation check passed."
}

if (( CHECK_ONLY == 1 )); then
    check_installation
    exit 0
fi

if (( SKIP_SYSTEM_DEPS == 0 )) && { (( INSTALL_SYSTEM_DEPS == 1 )) || system_dependencies_missing; }; then
    install_system_dependencies
elif (( SKIP_SYSTEM_DEPS == 1 )); then
    info "Skipping system dependency installation."
else
    info "Required Python and Pipenv commands are already available."
fi

resolve_python
command -v pipenv >/dev/null 2>&1 || die "Pipenv is unavailable. Run ./$SCRIPT_NAME without --skip-system-deps."

if [[ -e .venv && ! -x .venv/bin/python ]]; then
    die ".venv exists but is not a usable virtual environment. Move it aside or remove it manually, then rerun setup."
fi
if [[ -x .venv/bin/python ]]; then
    venv_version=$(./.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    [[ "$venv_version" == "3.10" ]] || die ".venv uses Python $venv_version. Move it aside or remove it manually, then rerun setup."
fi

info "Creating or updating the project-local Pipenv environment."
PIPENV_VENV_IN_PROJECT=1 PIPENV_NOSPIN=1 PIPENV_PYTHON="$PYTHON_CMD" pipenv sync --dev

PIPENV_VENV_IN_PROJECT=1 pipenv run python -c "import numpy, scipy"

info "Setup completed."
printf 'Run a representative case with:\n  pipenv run python main.py\n'
