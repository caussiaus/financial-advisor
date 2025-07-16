#!/usr/bin/env zsh
###############################################################################
# IPS Config‑Space — macOS bootstrap (zsh)
# ----------------------------------------
# • Creates ~/ips_env virtual environment
# • Installs pandas & openpyxl
# • Activates the venv and prints next‑step hints
###############################################################################

set -e

VENV_DIR="$HOME/ips_env"

echo "🐍  Creating/refreshing Python venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"

echo "⬆️   Upgrading pip + wheel ..."
pip install --upgrade pip wheel >/dev/null

echo "📦  Installing required libraries (pandas, openpyxl) ..."
pip install 'pandas>=2.0' openpyxl >/dev/null

echo ""
echo "✅  Environment ready!"
echo "Next steps:"
echo "  1) cd into your project folder    (e.g.  cd ~/ips_case1)"
echo "  2) copy / create   ips_config.json   and   ips_model.py"
echo "  3) run:            python ips_model.py"
echo ""
echo "Tip ➜  add  source \"$VENV_DIR/bin/activate\"  to your shell profile if you"
echo "        want the environment automatically when you open a new terminal."
