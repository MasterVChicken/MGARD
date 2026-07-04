#!/bin/bash
#SBATCH --job-name=mgard-merge-test
#SBATCH --partition=chen
#SBATCH --account=cdux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/home/jieyang/dev/MGARD/merge-test-%j.log

# Build MGARD-X (CUDA sm_90 + SERIAL, MDR, tests) after the
# sc26-blockmagrd-v1 merge and run the mgard-x unit tests on the H100.
# Environment mirrors build_hopper.sbatch (the known-good recipe on this
# cluster); dependency handling mirrors build_test_mgard.sh (reuse installed
# zstd/protobuf/Catch2 when present).

set -e
source /etc/profile
module purge
module load cuda/12.4.1
module load mpi/gcc/8.5.0/mpich/4.1.1
# drop spack's zstd (1.5.5) from pkg-config search so MGARD links our built
# 1.5.6; CMake still finds it via CMAKE_PREFIX_PATH
unset PKG_CONFIG_PATH

echo "=== env ==="
which nvcc && nvcc --version | grep release
which mpicxx && mpicxx --version | head -1
cmake --version | head -1
gcc --version | head -1
nvidia-smi --query-gpu=name --format=csv,noheader

cd /gpfs/home/jieyang/dev/MGARD

build_dir=./build-cuda-hopper
install_dir=./install-cuda-hopper
num_build_procs=${SLURM_CPUS_PER_TASK:-32}

export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib64:$LD_LIBRARY_PATH
export CC=gcc
export CXX=g++
export CUDACXX=nvcc

# ZSTD (reused if already installed)
if [ ! -f "${install_dir}/lib/libzstd.so" ]; then
  zstd_src_dir=${build_dir}/zstd/src
  [ -d "${zstd_src_dir}" ] || git clone -b v1.5.6 https://github.com/facebook/zstd.git ${zstd_src_dir}
  cmake -S ${zstd_src_dir}/build/cmake -B ${build_dir}/zstd/build \
      -DZSTD_MULTITHREAD_SUPPORT=ON -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_INSTALL_PREFIX=${install_dir}
  cmake --build ${build_dir}/zstd/build -j ${num_build_procs}
  cmake --install ${build_dir}/zstd/build
fi

# Protobuf (reused if already installed)
if [ ! -f "${install_dir}/lib64/libprotobuf.so" ] && [ ! -f "${install_dir}/lib/libprotobuf.so" ]; then
  protobuf_src_dir=${build_dir}/protobuf/src
  [ -d "${protobuf_src_dir}" ] || git clone -b v3.21.12 --recurse-submodules https://github.com/protocolbuffers/protobuf.git ${protobuf_src_dir}
  cmake -S ${protobuf_src_dir} -B ${build_dir}/protobuf/build \
      -Dprotobuf_BUILD_SHARED_LIBS=ON -Dprotobuf_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=${install_dir}
  cmake --build ${build_dir}/protobuf/build -j ${num_build_procs}
  cmake --install ${build_dir}/protobuf/build > /dev/null 2>&1
fi

# Catch2 (reused if already installed)
if [ ! -f "${install_dir}/lib64/libCatch2.a" ] && [ ! -f "${install_dir}/lib/libCatch2.a" ]; then
  catch2_src_dir=${build_dir}/catch2/src
  [ -d "${catch2_src_dir}" ] || git clone -b v3.3.2 --depth 1 https://github.com/catchorg/Catch2.git ${catch2_src_dir}
  cmake -S ${catch2_src_dir} -B ${build_dir}/catch2/build \
      -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${install_dir}
  cmake --build ${build_dir}/catch2/build -j ${num_build_procs}
  cmake --install ${build_dir}/catch2/build > /dev/null 2>&1
fi

# MGARD (fresh build tree so the merged CMake config is fully re-generated)
mgard_build_dir=${build_dir}/mgard-merge
rm -rf ${mgard_build_dir}
cmake -S . -B ${mgard_build_dir} \
    -DCMAKE_PREFIX_PATH="$(pwd)/${install_dir}" \
    -DBUILD_TESTING=ON \
    -DMGARD_ENABLE_CUDA=ON \
    -DMGARD_ENABLE_SERIAL=ON \
    -DMGARD_ENABLE_OPENMP=OFF \
    -DMGARD_ENABLE_MDR=ON \
    -DCMAKE_CUDA_ARCHITECTURES="90" \
    -DMGARD_ENABLE_AUTO_TUNING=OFF \
    -DMGARD_ENABLE_EXTERNAL_COMPRESSOR=OFF \
    -DMGARD_ENABLE_DOCS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${install_dir}
time cmake --build ${mgard_build_dir} -j ${num_build_procs}
echo "=== BUILD OK ==="

ctest --test-dir ${mgard_build_dir} -R mgard-x --output-on-failure
echo "=== CTEST EXIT: $? ==="
