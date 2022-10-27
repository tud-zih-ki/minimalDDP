module purge

ml modenv/hiera 
ml GCC/11.3.0
ml OpenMPI/4.1.4
ml CUDA/11.7.0
ml cuDNN/8.4.1.50-CUDA-11.7.0
ml NCCL/2.12.12-CUDA-11.7.0
ml Python/3.9.6-bare

VENV="./py396_gcc113_cu117" 
source "${VENV}/bin/activate"

set -euxo pipefail


export CC=gcc
export CXX=g++
export FC=gfortran

export C99=$CC

export I_MPI_CC=$CC
export I_MPI_CXX=$CXX
export I_MPI_FC=$FC

set -euxo pipefail

NPROCS=64

COMPILER=gcc # intel, gcc
MPI_TYPE=openmpi3 #intel3, openmpi3


SCOREP_VERSION=7.1
SCOREP_MASTER_VER=3eb34e6f
LIBUNWIND_VERSION=1.6.2

INSTALL_PATH=./installed
mkdir -p ${INSTALL_PATH}


wget https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/branches/master/sources.${SCOREP_MASTER_VER}.tar.gz
wget https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/scorep-${SCOREP_VERSION}/scorep-${SCOREP_VERSION}.tar.gz 
wget http://download.savannah.nongnu.org/releases/libunwind/libunwind-${LIBUNWIND_VERSION}.tar.gz

echo "Extracting Score-P" ${SCOREP_VERSION}
tar -xzf scorep-${SCOREP_VERSION}.tar.gz
tar -xzf sources.${SCOREP_MASTER_VER}.tar.gz

mkdir -p libunwind
tar -xzf libunwind-${LIBUNWIND_VERSION}.tar.gz --directory libunwind --strip-components=1

rm -f *.tar.gz

pushd libunwind
    
./configure --prefix=$PWD/../${INSTALL_PATH} 
make -j ${NPROCS} 
make install 
    
popd 
    

# install scorep

pushd scorep-${SCOREP_VERSION}
#pushd sources.${SCOREP_MASTER_VER}
mkdir -p _build
pushd _build
../configure --prefix=$PWD/../../${INSTALL_PATH} \
         --enable-shared \
         --without-shmem \
         --with-mpi=${MPI_TYPE} \
         --with-nocross-compiler-suite=${COMPILER} \
         --with-libcupti=/sw/installed/CUDA/11.7.0/extras/CUPTI/ \
         --with-libcupti-lib=/sw/installed/CUDA/11.7.0/extras/CUPTI/lib64 \
         --enable-cuda \
         --with-libunwind=$PWD/../../${INSTALL_PATH} \
         CC=$CC \
         CXX=$CXX \
         FC=$FC \
         MPI_CC=$I_MPI_CC \
         MPI_CXX=$I_MPI_CXX \
         MPI_FC=$I_MPI_FC

make -j ${NPROCS} 
make install 
popd
popd

rm -f scoreprc
cat <<EOF > scoreprc
export PATH=$PWD/${INSTALL_PATH}/bin:\$PATH
export LD_LIBRARY_PATH=$PWD/${INSTALL_PATH}/lib:\$LD_LIBRARY_PATH
EOF

source ./scoreprc
pip install scorep

