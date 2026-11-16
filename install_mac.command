#!/bin/bash
# Mother machine pipeline - installation for macOS (and Linux). Double-click it once
# (the first time, macOS may ask: right-click > Open > Open).
#  1) installs Miniforge (conda) in your home folder if it is not there yet;
#  2) creates the environment "mothermachine" with all the packages (~10 min, needs internet).
# Running it again updates the packages.
cd "$(dirname "$0")" || exit 1
echo
echo " Mother machine pipeline - installation"
echo " --------------------------------------"
buscar_conda() {
    CONDA=""
    for d in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/opt/miniconda3" "$HOME/opt/anaconda3" \
             "/opt/homebrew/Caskroom/miniforge/base" "/usr/local/Caskroom/miniforge/base" "/opt/miniconda3" "/opt/anaconda3"; do
        if [ -x "$d/bin/conda" ]; then CONDA="$d/bin/conda"; return; fi
    done
    command -v conda >/dev/null 2>&1 && CONDA="$(command -v conda)"
}
buscar_conda
if [ -z "$CONDA" ]; then
    echo " conda (Miniforge) is not installed on this computer."
    read -r -p " Download and install Miniforge now (about 100 MB, in your home folder)? [y/n] " r
    if [[ ! "$r" =~ ^[YySs] ]]; then
        echo " Install Miniforge yourself (https://github.com/conda-forge/miniforge) and run this file again."
        read -r -p " Press Enter to close."; exit 1
    fi
    so="$(uname)"; [ "$so" = "Darwin" ] && so="MacOSX"
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${so}-$(uname -m).sh"
    echo " Downloading $url"
    if ! curl -L --fail -o "/tmp/Miniforge3.sh" "$url"; then
        echo " The download failed: check the internet connection."; read -r -p " Press Enter to close."; exit 1
    fi
    bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3" || { echo " The installation of Miniforge failed."; read -r -p " Press Enter to close."; exit 1; }
    buscar_conda
fi
echo " conda: $CONDA"
echo
if "$CONDA" env list | grep -q "^mothermachine "; then
    echo " Updating the environment \"mothermachine\"..."
    "$CONDA" env update -f environment.yml --prune || { echo " Failed (see above)."; read -r -p " Press Enter to close."; exit 1; }
else
    echo " Creating the environment \"mothermachine\" (about 10 minutes)..."
    "$CONDA" env create -f environment.yml || { echo " Failed (see above)."; read -r -p " Press Enter to close."; exit 1; }
fi
echo
echo " Checking the installation..."
"$CONDA" run -n mothermachine python scripts/check_install.py
chmod +x start_mac.command 2>/dev/null
echo
echo " =========================================================================="
echo "  Installed. To use the pipeline, double-click start_mac.command in this"
echo "  folder (you can drag it to the Dock or make an alias on the Desktop)."
echo " =========================================================================="
read -r -p " Press Enter to close."
