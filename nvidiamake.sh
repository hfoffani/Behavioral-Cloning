export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cuda-install-samples-8.0.sh ~
cd ~/NVIDIA_CUDA-*_Samples/1_Utilities/deviceQuery/
make
~/NVIDIA_CUDA-*_Samples/bin/x86_64/linux/release/deviceQuery

