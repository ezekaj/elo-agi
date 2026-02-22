#!/usr/bin/env bash
set -euo pipefail

# ELO-AGI Installer
# Usage: curl -fsSL https://eloagi.com/install.sh | bash

INSTALL_DIR="$HOME/.elo-agi"
BIN_DIR="$HOME/.local/bin"
MIN_PYTHON="3.9"
PACKAGE="elo-agi"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}>${NC} $1"; }
success() { echo -e "${GREEN}>${NC} $1"; }
error() { echo -e "${RED}error:${NC} $1" >&2; exit 1; }

echo ""
echo -e "${PURPLE}${BOLD}  ELO-AGI Installer${NC}"
echo -e "  Neuroscience-inspired AGI framework"
echo ""

# --- Detect Python ---
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.9+ is required but not found.

  Install Python:
    macOS:   brew install python@3.12
    Ubuntu:  sudo apt install python3
    Fedora:  sudo dnf install python3

  Then run this installer again."
fi

PYTHON_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
info "Found Python ${PYTHON_VERSION}"

# --- Check for existing installation ---
if [ -d "$INSTALL_DIR" ]; then
    info "Upgrading existing installation..."
    "$INSTALL_DIR/venv/bin/pip" install --upgrade "$PACKAGE" --quiet 2>/dev/null || {
        info "Clean reinstall..."
        rm -rf "$INSTALL_DIR"
    }
fi

# --- Create virtual environment ---
if [ ! -d "$INSTALL_DIR" ]; then
    info "Creating virtual environment..."
    mkdir -p "$INSTALL_DIR"
    "$PYTHON" -m venv "$INSTALL_DIR/venv"

    info "Installing ${PACKAGE}..."
    "$INSTALL_DIR/venv/bin/pip" install --upgrade pip --quiet 2>/dev/null
    "$INSTALL_DIR/venv/bin/pip" install "$PACKAGE" --quiet
fi

# --- Create wrapper scripts ---
mkdir -p "$BIN_DIR"

# neuro command
cat > "$BIN_DIR/neuro" << 'WRAPPER'
#!/usr/bin/env bash
exec "$HOME/.elo-agi/venv/bin/neuro" "$@"
WRAPPER
chmod +x "$BIN_DIR/neuro"

# neuro-demo command
cat > "$BIN_DIR/neuro-demo" << 'WRAPPER'
#!/usr/bin/env bash
exec "$HOME/.elo-agi/venv/bin/neuro-demo" "$@"
WRAPPER
chmod +x "$BIN_DIR/neuro-demo"

# neuro-bench command
cat > "$BIN_DIR/neuro-bench" << 'WRAPPER'
#!/usr/bin/env bash
exec "$HOME/.elo-agi/venv/bin/neuro-bench" "$@"
WRAPPER
chmod +x "$BIN_DIR/neuro-bench"

# --- Ensure PATH includes ~/.local/bin ---
SHELL_NAME=$(basename "$SHELL" 2>/dev/null || echo "bash")
PROFILE=""
case "$SHELL_NAME" in
    zsh)  PROFILE="$HOME/.zshrc" ;;
    bash)
        if [ -f "$HOME/.bash_profile" ]; then
            PROFILE="$HOME/.bash_profile"
        else
            PROFILE="$HOME/.bashrc"
        fi
        ;;
    fish) PROFILE="$HOME/.config/fish/config.fish" ;;
    *)    PROFILE="$HOME/.profile" ;;
esac

PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
if [ "$SHELL_NAME" = "fish" ]; then
    PATH_LINE='fish_add_path $HOME/.local/bin'
fi

if ! echo "$PATH" | tr ':' '\n' | grep -q "$BIN_DIR"; then
    if [ -n "$PROFILE" ] && ! grep -q '.local/bin' "$PROFILE" 2>/dev/null; then
        echo "" >> "$PROFILE"
        echo "# ELO-AGI" >> "$PROFILE"
        echo "$PATH_LINE" >> "$PROFILE"
        info "Added ~/.local/bin to PATH in ${PROFILE##*/}"
    fi
    export PATH="$BIN_DIR:$PATH"
fi

# --- Verify ---
VERSION=$("$INSTALL_DIR/venv/bin/python" -c "import neuro; print(neuro.__version__)" 2>/dev/null || echo "unknown")

echo ""
success "${BOLD}ELO-AGI v${VERSION} installed successfully!${NC}"
echo ""
echo -e "  ${BOLD}Get started:${NC}"
echo -e "    ${CYAN}neuro${NC}              Interactive session"
echo -e "    ${CYAN}neuro${NC} --version     Check version"
echo -e "    ${CYAN}neuro-demo${NC} all      Run demos"
echo ""

if ! echo "$PATH" | tr ':' '\n' | grep -q "$BIN_DIR"; then
    echo -e "  ${BOLD}Note:${NC} Restart your terminal or run:"
    echo -e "    source ${PROFILE##*/}"
    echo ""
fi
