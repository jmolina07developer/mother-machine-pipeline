#!/bin/bash
# Mother machine pipeline - opens the graphical interface in the web browser.
# Keep this window open while you use it; close it to stop the interface.
cd "$(dirname "$0")" || exit 1
BASE=""
for d in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/opt/miniconda3" "$HOME/opt/anaconda3" \
         "/opt/homebrew/Caskroom/miniforge/base" "/usr/local/Caskroom/miniforge/base" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -f "$d/etc/profile.d/conda.sh" ]; then BASE="$d"; break; fi
done
if [ -z "$BASE" ] && command -v conda >/dev/null 2>&1; then BASE="$(conda info --base)"; fi
if [ -z "$BASE" ]; then echo " conda was not found: double-click install_mac.command first."; read -r -p " Press Enter to close."; exit 1; fi
# shellcheck disable=SC1091
source "$BASE/etc/profile.d/conda.sh"
if ! conda activate mothermachine; then
    echo " The environment \"mothermachine\" does not exist: double-click install_mac.command first."
    read -r -p " Press Enter to close."; exit 1
fi
echo
echo " Mother machine pipeline"
echo " The interface opens in your web browser at http://localhost:8501"
echo " Keep this window open while you use it; close it to stop the interface."
echo
abrir="open"; command -v open >/dev/null 2>&1 || abrir="xdg-open"
(sleep 5; "$abrir" http://localhost:8501) >/dev/null 2>&1 &
cd scripts || exit 1
python -m streamlit run interfaz.py --server.headless true --server.address localhost --server.port 8501 --browser.gatherUsageStats false --client.toolbarMode viewer
