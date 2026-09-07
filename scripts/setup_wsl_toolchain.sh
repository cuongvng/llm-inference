#!/usr/bin/env bash
# Run with: sudo bash scripts/setup_wsl_toolchain.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run me with sudo: sudo bash $0" >&2
  exit 1
fi

apt-get update
apt-get install -y --no-install-recommends ca-certificates gpg wget build-essential

# --- CMake (Kitware apt repo) -----------------------------------------------
cmake_too_old() {
  command -v cmake >/dev/null || return 0
  local have
  have=$(cmake --version | head -1 | awk '{print $3}')
  [[ "$(printf '%s\n3.18\n' "$have" | sort -V | head -1)" != "3.18" ]]
}

if cmake_too_old; then
  wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main" \
    > /etc/apt/sources.list.d/kitware.list
  apt-get update
  apt-get install -y cmake
fi

# --- CUDA toolkit 12.2 (NVIDIA WSL-Ubuntu repo, no driver) ------------------
if ! command -v nvcc >/dev/null && [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
    -O /tmp/cuda-keyring.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update
  apt-get install -y cuda-toolkit-12-2
fi

# --- PATH ---------------------------------------------------------------------
PROFILE_SNIPPET=/etc/profile.d/cuda.sh
cat > "$PROFILE_SNIPPET" <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
EOF
chmod 0644 "$PROFILE_SNIPPET"

echo
echo "done. open a new shell (or: source $PROFILE_SNIPPET), then:"
echo "  nvcc --version && cmake --version"
echo "  cmake --preset relwithdebinfo && cmake --build --preset relwithdebinfo && ctest --preset relwithdebinfo"
