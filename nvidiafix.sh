wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
rm cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo apt-get update
sudo apt-get install cuda

# sudo apt-get update && sudo apt-get -y upgrade \
    # install the package maintainer's version (of /boot/grub/menu.lst)
# sudo apt-get install -y linux-image-extra-`uname -r`
# sudo apt-get update
# wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb

# sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
# sudo apt-get update
# sudo apt-get install -y cuda
