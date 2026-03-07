<!-- MarkdownTOC -->

- [swap](#swa_p_)
- [init](#ini_t_)
- [flags](#flags_)
- [persistence](#persistence_)
- [power_limit](#power_limit_)
  - [pc       @ power_limit](#pc___power_limi_t_)
- [sudo_no_pwd](#sudo_no_pwd_)
- [ssh](#ssh_)
- [nomachine](#nomachine_)
- [rustdesk self-host](#rustdesk_self_hos_t_)
  - [docker       @ rustdesk_self-host](#docker___rustdesk_self_host_)
  - [deb       @ rustdesk_self-host](#deb___rustdesk_self_host_)
  - [client       @ rustdesk_self-host](#client___rustdesk_self_host_)
    - [x99       @ client/rustdesk_self-host](#x99___client_rustdesk_self_hos_t_)
- [docker](#docke_r_)
  - [nvidia-container-toolkit       @ docker](#nvidia_container_toolkit___docker_)
- [install nvidia drivers](#install_nvidia_driver_s_)
  - [418_for_cuda_10.1       @ install_nvidia_drivers](#418_for_cuda_10_1___install_nvidia_drivers_)
  - [410_for_cuda_10.0       @ install_nvidia_drivers](#410_for_cuda_10_0___install_nvidia_drivers_)
  - [396_for_cuda_9       @ install_nvidia_drivers](#396_for_cuda_9___install_nvidia_drivers_)
  - [20.04       @ install_nvidia_drivers](#20_04___install_nvidia_drivers_)
    - [470_for_cuda_11       @ 20.04/install_nvidia_drivers](#470_for_cuda_11___20_04_install_nvidia_drivers_)
    - [others       @ 20.04/install_nvidia_drivers](#others___20_04_install_nvidia_drivers_)
- [cuda](#cud_a_)
  - [bugs       @ cuda](#bugs___cuda_)
  - [CUDA_12.2:       @ cuda](#cuda_12_2____cuda_)
  - [cuda-12.3       @ cuda](#cuda_12_3___cuda_)
    - [cuda-13.1       @ cuda-12.3/cuda](#cuda_13_1___cuda_12_3_cuda_)
  - [24.04       @ cuda](#24_04___cuda_)
    - [all       @ 24.04/cuda](#all___24_04_cuda_)
      - [12.2       @ all/24.04/cuda](#12_2___all_24_04_cuda_)
      - [12.3       @ all/24.04/cuda](#12_3___all_24_04_cuda_)
  - [cudnn-deb       @ cuda](#cudnn_deb___cuda_)
- [disks](#disks_)
- [tmux](#tmu_x_)
- [git](#git_)
- [protobuf compiler](#protobuf_compiler_)
- [misc packages](#misc_packages_)
  - [18.04       @ misc_packages](#18_04___misc_package_s_)
- [update pip](#update_pi_p_)
  - [24.04       @ update_pip](#24_04___update_pip_)
  - [18.04       @ update_pip](#18_04___update_pip_)
  - [all       @ update_pip](#all___update_pip_)
- [setup python 3](#setup_python_3_)
  - [20.04       @ setup_python_3](#20_04___setup_python_3_)
    - [python_3.6       @ 20.04/setup_python_3](#python_3_6___20_04_setup_python_3_)
    - [python_3.8       @ 20.04/setup_python_3](#python_3_8___20_04_setup_python_3_)
    - [python_3.10       @ 20.04/setup_python_3](#python_3_10___20_04_setup_python_3_)
      - [fix_corruption_when_upgrading_18.04_to_20.04       @ python_3.10/20.04/setup_python_3](#fix_corruption_when_upgrading_18_04_to_20_04___python_3_10_20_04_setup_python_3_)
  - [install_core_library       @ setup_python_3](#install_core_library___setup_python_3_)
      - [python_3.6       @ install_core_library/setup_python_3](#python_3_6___install_core_library_setup_python_3_)
    - [Ubuntu_14.04       @ install_core_library/setup_python_3](#ubuntu_14_04___install_core_library_setup_python_3_)
  - [install_packages       @ setup_python_3](#install_packages___setup_python_3_)
    - [linux       @ install_packages/setup_python_3](#linux___install_packages_setup_python_3_)
    - [all       @ install_packages/setup_python_3](#all___install_packages_setup_python_3_)
      - [pycocotools       @ all/install_packages/setup_python_3](#pycocotools___all_install_packages_setup_python_3_)
      - [lapsolver       @ all/install_packages/setup_python_3](#lapsolver___all_install_packages_setup_python_3_)
      - [paramiko_issue       @ all/install_packages/setup_python_3](#paramiko_issue___all_install_packages_setup_python_3_)
    - [windows       @ install_packages/setup_python_3](#windows___install_packages_setup_python_3_)
      - [AES       @ windows/install_packages/setup_python_3](#aes___windows_install_packages_setup_python_3_)
      - [x32       @ windows/install_packages/setup_python_3](#x32___windows_install_packages_setup_python_3_)
  - [ffmpeg       @ setup_python_3](#ffmpeg___setup_python_3_)
  - [pytorch_and_vis_tools       @ setup_python_3](#pytorch_and_vis_tools___setup_python_3_)
    - [gpu       @ pytorch_and_vis_tools/setup_python_3](#gpu___pytorch_and_vis_tools_setup_python_3_)
      - [grs_/_ubuntu_18.04_/_cuda11       @ gpu/pytorch_and_vis_tools/setup_python_3](#grs___ubuntu_18_04___cuda11___gpu_pytorch_and_vis_tools_setup_python_3_)
    - [latest       @ pytorch_and_vis_tools/setup_python_3](#latest___pytorch_and_vis_tools_setup_python_3_)
    - [v1.0       @ pytorch_and_vis_tools/setup_python_3](#v1_0___pytorch_and_vis_tools_setup_python_3_)
      - [linux_python3.8/cuda_10.0       @ v1.0/pytorch_and_vis_tools/setup_python_3](#linux_python3_8_cuda_10_0___v1_0_pytorch_and_vis_tools_setup_python_3_)
      - [linux_python3.6/cuda_10.0       @ v1.0/pytorch_and_vis_tools/setup_python_3](#linux_python3_6_cuda_10_0___v1_0_pytorch_and_vis_tools_setup_python_3_)
        - [apex       @ linux_python3.6/cuda_10.0/v1.0/pytorch_and_vis_tools/setup_python_3](#apex___linux_python3_6_cuda_10_0_v1_0_pytorch_and_vis_tools_setup_python_3_)
    - [windows_python3.7/cuda_10.0       @ pytorch_and_vis_tools/setup_python_3](#windows_python3_7_cuda_10_0___pytorch_and_vis_tools_setup_python_3_)
  - [opencv       @ setup_python_3](#opencv___setup_python_3_)
    - [4.5.4.60       @ opencv/setup_python_3](#4_5_4_60___opencv_setup_python_3_)
    - [4.1.0       @ opencv/setup_python_3](#4_1_0___opencv_setup_python_3_)
    - [3.4.11.45       @ opencv/setup_python_3](#3_4_11_45___opencv_setup_python_3_)
    - [3.4.5       @ opencv/setup_python_3](#3_4_5___opencv_setup_python_3_)
      - [uninstall       @ 3.4.5/opencv/setup_python_3](#uninstall___3_4_5_opencv_setup_python_3_)
    - [3.4.4.19_CC       @ opencv/setup_python_3](#3_4_4_19_cc___opencv_setup_python_3_)
  - [tensorflow       @ setup_python_3](#tensorflow___setup_python_3_)
    - [check_version       @ tensorflow/setup_python_3](#check_version___tensorflow_setup_python_3_)
    - [check_gpu       @ tensorflow/setup_python_3](#check_gpu___tensorflow_setup_python_3_)
    - [install       @ tensorflow/setup_python_3](#install___tensorflow_setup_python_3_)
    - [1.14       @ tensorflow/setup_python_3](#1_14___tensorflow_setup_python_3_)
      - [python3.8       @ 1.14/tensorflow/setup_python_3](#python3_8___1_14_tensorflow_setup_python_3_)
      - [from_source       @ 1.14/tensorflow/setup_python_3](#from_source___1_14_tensorflow_setup_python_3_)
    - [windows       @ tensorflow/setup_python_3](#windows___tensorflow_setup_python_3_)
    - [v1.6_for_cuda_9.0       @ tensorflow/setup_python_3](#v1_6_for_cuda_9_0___tensorflow_setup_python_3_)
      - [python_3.5       @ v1.6_for_cuda_9.0/tensorflow/setup_python_3](#python_3_5___v1_6_for_cuda_9_0_tensorflow_setup_python_3_)
      - [python_3.6       @ v1.6_for_cuda_9.0/tensorflow/setup_python_3](#python_3_6___v1_6_for_cuda_9_0_tensorflow_setup_python_3_)
    - [v1.6_for_cuda_8.0       @ tensorflow/setup_python_3](#v1_6_for_cuda_8_0___tensorflow_setup_python_3_)
    - [v1.4_for_cuda_8.0       @ tensorflow/setup_python_3](#v1_4_for_cuda_8_0___tensorflow_setup_python_3_)
  - [theano_and_keras       @ setup_python_3](#theano_and_keras___setup_python_3_)
- [setup python 2](#setup_python_2_)
  - [install_packages       @ setup_python_2](#install_packages___setup_python_2_)
    - [windows       @ install_packages/setup_python_2](#windows___install_packages_setup_python_2_)
  - [install_opencv       @ setup_python_2](#install_opencv___setup_python_2_)
    - [4.1.0       @ install_opencv/setup_python_2](#4_1_0___install_opencv_setup_python_2_)
    - [3.4.5       @ install_opencv/setup_python_2](#3_4_5___install_opencv_setup_python_2_)
    - [uninstall       @ install_opencv/setup_python_2](#uninstall___install_opencv_setup_python_2_)
  - [tensorflow       @ setup_python_2](#tensorflow___setup_python_2_)
    - [1.14       @ tensorflow/setup_python_2](#1_14___tensorflow_setup_python_2_)
    - [v1.6_for_cuda_9.0       @ tensorflow/setup_python_2](#v1_6_for_cuda_9_0___tensorflow_setup_python_2_)
    - [v1.6_for_cuda_8.0       @ tensorflow/setup_python_2](#v1_6_for_cuda_8_0___tensorflow_setup_python_2_)
    - [v1.4_for_cuda_8.0       @ tensorflow/setup_python_2](#v1_4_for_cuda_8_0___tensorflow_setup_python_2_)
    - [windows/python3.7       @ tensorflow/setup_python_2](#windows_python3_7___tensorflow_setup_python_2_)
      - [no_gpu       @ windows/python3.7/tensorflow/setup_python_2](#no_gpu___windows_python3_7_tensorflow_setup_python_2_)
- [misc tools](#misc_tool_s_)
  - [jedit       @ misc_tools](#jedit___misc_tools_)
  - [imagemagick_7       @ misc_tools](#imagemagick_7___misc_tools_)
  - [pycharm       @ misc_tools](#pycharm___misc_tools_)
  - [jpeg4py       @ misc_tools](#jpeg4py___misc_tools_)
- [multi_python_issues](#multi_python_issues_)
  - [python_3.6_and_3.5       @ multi_python_issues](#python_3_6_and_3_5___multi_python_issue_s_)
- [optional](#optiona_l_)
  - [network_printers       @ optional](#network_printers___optional_)
  - [install_opencv_from_source       @ optional](#install_opencv_from_source___optional_)
    - [windows       @ install_opencv_from_source/optional](#windows___install_opencv_from_source_optiona_l_)
    - [linux       @ install_opencv_from_source/optional](#linux___install_opencv_from_source_optiona_l_)
      - [3.4.5       @ linux/install_opencv_from_source/optional](#3_4_5___linux_install_opencv_from_source_optiona_l_)
      - [3.4.1       @ linux/install_opencv_from_source/optional](#3_4_1___linux_install_opencv_from_source_optiona_l_)
      - [cuda       @ linux/install_opencv_from_source/optional](#cuda___linux_install_opencv_from_source_optiona_l_)
      - [no_cuda       @ linux/install_opencv_from_source/optional](#no_cuda___linux_install_opencv_from_source_optiona_l_)
      - [2.4.13.6       @ linux/install_opencv_from_source/optional](#2_4_13_6___linux_install_opencv_from_source_optiona_l_)
      - [compile_and_install       @ linux/install_opencv_from_source/optional](#compile_and_install___linux_install_opencv_from_source_optiona_l_)
  - [setup_point_grey_ethernet_camera       @ optional](#setup_point_grey_ethernet_camera___optional_)
    - [ffmpeg       @ setup_point_grey_ethernet_camera/optional](#ffmpeg___setup_point_grey_ethernet_camera_optiona_l_)
    - [sdk       @ setup_point_grey_ethernet_camera/optional](#sdk___setup_point_grey_ethernet_camera_optiona_l_)
    - [python_wrappers       @ setup_point_grey_ethernet_camera/optional](#python_wrappers___setup_point_grey_ethernet_camera_optiona_l_)
  - [setting_up_detector_training       @ optional](#setting_up_detector_training___optional_)
- [issues](#issue_s_)
  - [broken_software_updates_in_16.04_due_to_python_35/36_mismatch       @ issues](#broken_software_updates_in_16_04_due_to_python_35_36_mismatch___issues_)
- [gcc](#gcc_)
  - [6.3       @ gcc](#6_3___gc_c_)
  - [7.0       @ gcc](#7_0___gc_c_)
- [matlab engine](#matlab_engine_)
  - [mdp_mot_devkit       @ matlab_engine](#mdp_mot_devkit___matlab_engin_e_)
- [vot_devkit       @ matlab_engine](#vot_devkit___matlab_engin_e_)

<!-- /MarkdownTOC -->

<a id="swa_p_"></a>
# swap
dd if=/dev/zero of=/home/abhineet/swapfile64.img bs=1024 count=64M
mkswap /home/abhineet/swapfile64.img

sudo nano /etc/fstab

Add this line at the end:
/home/abhineet/swapfile64.img swap swap sw 0 0

sudo swapon /home/abhineet/swapfile64.img
(this command will need to be run again on each reboot)

<a id="ini_t_"></a>
# init
sudo apt-get purge unattended-upgrades
sudo apt install nautilus-share
sudo apt install thunar-archive-plugin
sudo apt install nvtop

<a id="flags_"></a>
# flags
CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=1
DISPLAY=localhost:10.0

<a id="persistence_"></a>
# persistence
sudo nvidia-smi -pm 1 

<a id="power_limit_"></a>
# power_limit
sudo nvidia-smi -i 0,1 -pl 100
sudo nvidia-smi -i 0,1 -pl 150
sudo nvidia-smi -i 0,1 -pl 160
sudo nvidia-smi -i 0,1 -pl 170
sudo nvidia-smi -i 0,1 -pl 175
sudo nvidia-smi -i 0,1 -pl 180
sudo nvidia-smi -i 0,1 -pl 200
sudo nvidia-smi -i 0,1 -pl 250
sudo nvidia-smi -i 0,1 -pl 300
sudo nvidia-smi -i 0,1 -pl 350
sudo nvidia-smi -i 0 -pl 350
sudo nvidia-smi -i 1 -pl 350

sudo nvidia-smi -i 0,1,2 -pl 150
sudo nvidia-smi -i 0,1,2 -pl 170
sudo nvidia-smi -i 0,1,2 -pl 175
sudo nvidia-smi -i 0,1,2 -pl 200

sudo nvidia-smi -i 1 -pl 100
sudo nvidia-smi -i 1 -pl 120
sudo nvidia-smi -i 2 -pl 120
sudo nvidia-smi -i 0 -pl 100
sudo nvidia-smi -i 0 -pl 120
sudo nvidia-smi -i 0 -pl 150
sudo nvidia-smi -i 0 -pl 180
sudo nvidia-smi -i 1 -pl 150
sudo nvidia-smi -i 0 -pl 200
sudo nvidia-smi -i 0 -pl 250
sudo nvidia-smi -i 0 -pl 300
sudo nvidia-smi -i 0 -pl 350

watch nvidia-smi

<a id="pc___power_limi_t_"></a>
## pc       @ power_limit-->install
nvidia-smi -i 0 -pl 300
nvidia-smi -i 1,2 -pl 100

<a id="sudo_no_pwd_"></a>
# sudo_no_pwd
https://linuxhint.com/setup-sudo-no-password-linux/
sudo visudo
add at end of file (and that means the end, i.e. after the @includedir /etc/sudoers.d line);

abhineet ALL=(ALL) NOPASSWD:ALL

<a id="ssh_"></a>
# ssh
https://www.cyberciti.biz/faq/ubuntu-linux-install-openssh-server/
sudo apt-get install openssh-server
sudo systemctl enable ssh
sudo systemctl enable ssh --now
sudo systemctl start ssh

<a id="nomachine_"></a>
# nomachine
https://www.nomachine.com/getting-started-with-nomachine
https://downloads.nomachine.com/linux/?id=1
sudo dpkg -i nomachine_8.13.1_1_amd64.deb

<a id="rustdesk_self_hos_t_"></a>
# rustdesk self-host
https://rustdesk.com/docs/en/self-host/rustdesk-server-oss/install/

sudo ufw allow 21114:21119/tcp
sudo ufw allow 21116/udp
sudo ufw enable
sudo ufw disable

<a id="docker___rustdesk_self_host_"></a>
## docker       @ rustdesk_self-host-->install
https://rustdesk.com/docs/en/self-host/rustdesk-server-oss/docker/

<a id="deb___rustdesk_self_host_"></a>
## deb       @ rustdesk_self-host-->install
https://github.com/rustdesk/rustdesk-server/releases/tag/1.1.15

wget https://github.com/rustdesk/rustdesk-server/releases/download/1.1.15/rustdesk-server-hbbr_1.1.15_amd64.deb
wget https://github.com/rustdesk/rustdesk-server/releases/download/1.1.15/rustdesk-server-hbbs_1.1.15_amd64.deb
wget https://github.com/rustdesk/rustdesk-server/releases/download/1.1.15/rustdesk-server-utils_1.1.15_amd64.deb

sudo dpkg -i rustdesk-server-hbbr_1.1.15_amd64.deb
sudo dpkg -i rustdesk-server-hbbs_1.1.15_amd64.deb
sudo dpkg -i rustdesk-server-utils_1.1.15_amd64.deb

<a id="client___rustdesk_self_host_"></a>
## client       @ rustdesk_self-host-->install
https://rustdesk.com/docs/en/self-host/client-configuration/#2-manual-config
<a id="x99___client_rustdesk_self_hos_t_"></a>
### x99       @ client/rustdesk_self-host-->install
./var/lib/rustdesk-server/id_ed25519.pub

<a id="docke_r_"></a>
# docker
https://docs.docker.com/engine/install/ubuntu/
```
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

<a id="nvidia_container_toolkit___docker_"></a>
## nvidia-container-toolkit       @ docker-->install
https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container/58432877#58432877
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update 
sudo apt-get install -y nvidia-container-toolkit
```


<a id="install_nvidia_driver_s_"></a>
# install nvidia drivers
**DO NOT RESTART OR LOGOUT BEFORE noveau HAS BEEN REMOVED AND BLACKLISTED OR YOU WON'T BE ABLE TO LOG BACK IN (OR EVEN ACCESS TERMINAL) in 18.04.**

1. uninstall noveau drivers:
```
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get remove --purge '^libnvidia-.*'
sudo apt-get remove --purge '^cuda-.*'
```

messed up nvidia xorg, e.g. blank screen at login as if login bar is out of sight and nomachine not working correctly
```
sudo mv /etc/X11/xorg.conf /etc/X11/xorg.conf.backup
sudo reboot
```

2. blacklist noveau drivers using instructions here:

https://askubuntu.com/questions/841876/how-to-disable-nouveau-kernel-driver

Create a file

```
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```
with the following contents:

```
blacklist nouveau
options nouveau modeset=0

```
Regenerate the kernel initramfs:

```
sudo update-initramfs -u
```
and finally: reboot

```
sudo shutdown -r now


sudo shutdown now
```

3. install nvidia drivers:
__new system : might be better to directly install cuda which also installs driver__


```
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update 
ubuntu-drivers devices
```

<a id="418_for_cuda_10_1___install_nvidia_drivers_"></a>
## 418_for_cuda_10.1       @ install_nvidia_drivers-->install
```
sudo apt-get install nvidia-418
```
<a id="410_for_cuda_10_0___install_nvidia_drivers_"></a>
## 410_for_cuda_10.0       @ install_nvidia_drivers-->install

```
sudo apt-get install nvidia-410
```

<a id="396_for_cuda_9___install_nvidia_drivers_"></a>
## 396_for_cuda_9       @ install_nvidia_drivers-->install
```
sudo apt-get install nvidia-396
```

<a id="20_04___install_nvidia_drivers_"></a>
## 20.04       @ install_nvidia_drivers-->install

<a id="470_for_cuda_11___20_04_install_nvidia_drivers_"></a>
### 470_for_cuda_11       @ 20.04/install_nvidia_drivers-->install
```
sudo apt install nvidia-driver-470
```
<a id="others___20_04_install_nvidia_drivers_"></a>
### others       @ 20.04/install_nvidia_drivers-->install
```
sudo apt install nvidia-driver-510
sudo apt install nvidia-driver-515
sudo apt install nvidia-driver-520
sudo apt install nvidia-driver-525
sudo apt install nvidia-driver-535
sudo apt install nvidia-driver-545
sudo apt install nvidia-driver-550
sudo apt install nvidia-driver-555
sudo apt install nvidia-driver-560
```

4. restart
```
sudo shutdown -r now
```

<a id="cud_a_"></a>
# cuda
nvcc --version
/usr/local/cuda/bin/nvcc --version

<a id="bugs___cuda_"></a>
## bugs       @ cuda-->install
`CUDA stops working after waking up from suspend`
https://askubuntu.com/questions/1228423/how-do-i-fix-cuda-breaking-after-suspend

sudo nano /etc/modprobe.d/nvidia-power-management.conf

options nvidia NVreg_PreserveVideoMemoryAllocations=1 NVreg_TemporaryFilePath=/tmp

sudo update-initramfs -u
sudo shutdown -r now

no need to enable the suspend service in the newest drivers
sudo systemctl enable nvidia-suspend.service

<a id="cuda_12_4__"></a>
<a id="cuda_12_2____cuda_"></a>
## CUDA_12.2:       @ cuda-->install
`from p2s`
https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install cuda-12-2

sudo apt-get remove cuda-12-2

```
<a id="cuda_12_3___cuda_"></a>
## cuda-12.3       @ cuda-->install
`needed for tensorflow 2.20`
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.2-545.23.08-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.2-545.23.08-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

export LD_LIBRARY_PATH=/usr/local/cuda-12.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda-12.3/bin
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.3
```
<a id="cuda_13_1___cuda_12_3_cuda_"></a>
### cuda-13.1       @ cuda-12.3/cuda-->install
`for ubuntu 24.04 / mint 22.3`
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.1.1/local_installers/cuda-repo-ubuntu2404-13-1-local_13.1.1-590.48.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-13-1-local_13.1.1-590.48.01-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1

wget https://developer.download.nvidia.com/compute/cudnn/9.19.1/local_installers/cudnn-local-repo-ubuntu2404-9.19.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.19.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.19.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-13

```
<a id="24_04___cuda_"></a>
## 24.04       @ cuda-->install
`nsight-systems-2023.2.3 : Depends: libtinfo5 but it is not installable`
https://askubuntu.com/a/1493087
sudo nano /etc/apt/sources.list
deb http://old-releases.ubuntu.com/ubuntu/ lunar universe
sudo apt update

<a id="all___24_04_cuda_"></a>
### all       @ 24.04/cuda-->install
download cudnn tar from
https://developer.nvidia.com/rdp/cudnn-download
```
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
```
and extract into cuda/targets/ folder
```
tar xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
```
<a id="12_2___all_24_04_cuda_"></a>
#### 12.2       @ all/24.04/cuda-->install
```
sudo mv cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/* /usr/local/cuda-12.2/targets/x86_64-linux/lib/
sudo mv cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/* /usr/local/cuda-12.2/targets/x86_64-linux/include/
```
<a id="12_3___all_24_04_cuda_"></a>
#### 12.3       @ all/24.04/cuda-->install
```
sudo mv cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/* /usr/local/cuda-12.3/targets/x86_64-linux/lib/
sudo mv cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/* /usr/local/cuda-12.3/targets/x86_64-linux/include/

```

add following environment variables to bashrc as well as the pycharm debug configuration window
```
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda-12.2/bin
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.2
```

<a id="cudnn_deb___cuda_"></a>
## cudnn-deb       @ cuda-->install
`does not work`
https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=12&target_type=deb_local

wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12

<a id="disks_"></a>
# disks
sudo apt-get install gnome-disk-utility

<a id="tmu_x_"></a>
# tmux
sudo apt-get install tmux
chmod +x ~/.tmux/plugins/tpm/tpm

<a id="git_"></a>
# git
sudo apt-get install git
git config --global credential.helper store

<a id="protobuf_compiler_"></a>
# protobuf compiler
sudo apt-get install protobuf-compiler

<a id="misc_packages_"></a>
# misc packages
sudo apt-get install libboost-all-dev
sudo apt install libboost1.58-dev
sudo apt-get install libzip-dev
apt-get install libhdf5-dev
apt-get install libleveldb-dev
apt-get install libsnappy-dev
apt-get install liblmdb-dev
sudo apt-get install libatlas-base-dev

<a id="18_04___misc_package_s_"></a>
## 18.04       @ misc_packages-->install
sudo apt install caffe-cuda

<a id="update_pi_p_"></a>
# update pip

<a id="24_04___update_pip_"></a>
## 24.04       @ update_pip-->install
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
sudo apt install python3.10

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.10


<a id="18_04___update_pip_"></a>
## 18.04       @ update_pip-->install
```
sudo apt-get install python3-distutils
sudo apt-get install python3-apt
sudo apt install python3-testresources

sudo apt-get install python3.6-distutils
sudo apt install python3-testresources
```

<a id="all___update_pip_"></a>
## all       @ update_pip-->install

```
wget https://bootstrap.pypa.io/get-pip.py

wget https://bootstrap.pypa.io/pip/3.7/get-pip.py

python3 get-pip.py
python36 get-pip.py
python3.7 get-pip.py
python3.9 get-pip.py
python3.10 get-pip.py
python3.11 get-pip.py
python3.12 get-pip.py
/usr/bin/python3 get-pip.py

python -m pip install --upgrade pip
```

<a id="setup_python_3_"></a>
# setup python 3

<a id="20_04___setup_python_3_"></a>
## 20.04       @ setup_python_3-->install

```
ln -sf /usr/bin/python3.6 /usr/bin/python3
ln -sf /usr/bin/python3.8 /usr/bin/python3

ln -sf /usr/bin/python3.6 /usr/bin/python36

sudo update-alternatives --install /usr/bin/python36 python36 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/python36 python36 /usr/bin/python3.6 2
sudo update-alternatives --config python3

```

<a id="python_3_6___20_04_setup_python_3_"></a>
### python_3.6       @ 20.04/setup_python_3-->install
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

sudo apt-get install python3.6-dev
sudo apt-get install python3.6-tk
```

<a id="python_3_8___20_04_setup_python_3_"></a>
### python_3.8       @ 20.04/setup_python_3-->install
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

sudo apt-get install python3.8-dev
sudo apt-get remove python3.8-dev
sudo apt-get install python3.8-tk
```
<a id="python_3_10___20_04_setup_python_3_"></a>
### python_3.10       @ 20.04/setup_python_3-->install
```
sudo apt-get install python3.10-dev
sudo apt-get remove python3.10-dev
sudo apt-get install python3.10-tk
```

<a id="fix_corruption_when_upgrading_18_04_to_20_04___python_3_10_20_04_setup_python_3_"></a>
#### fix_corruption_when_upgrading_18.04_to_20.04       @ python_3.10/20.04/setup_python_3-->install
https://stackoverflow.com/questions/61627422/upgrade-ubuntu-18-04-to-20-04-but-packages-remain-bionic1

apt-cache showpkg python3.8

sudo apt install libpython3.8:amd64=3.8.2-1ubuntu1 libpython3.8-dev:amd64=3.8.2-1ubuntu1 libpython3.8-minimal:amd64=3.8.2-1ubuntu1 libpython3.8-stdlib:amd64=3.8.2-1ubuntu1 python3.8=3.8.2-1ubuntu1 python3.8-minimal=3.8.2-1ubuntu1

<a id="install_core_library___setup_python_3_"></a>
## install_core_library       @ setup_python_3-->install

```
sudo apt-get install python3-dev
```

<a id="python_3_6___install_core_library_setup_python_3_"></a>
#### python_3.6       @ install_core_library/setup_python_3-->install
```
sudo apt-get install python3.6-dev
```

<a id="ubuntu_14_04___install_core_library_setup_python_3_"></a>
### Ubuntu_14.04       @ install_core_library/setup_python_3-->install
On Ubuntu 14.04, running the above command usually installs python 3.4.3 which is too old to run the labeling tool and some of the batch scripts.
Following commands should be used to install python 3.5 instead:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.5
```

<a id="install_packages___setup_python_3_"></a>
## install_packages       @ setup_python_3-->install

<a id="linux___install_packages_setup_python_3_"></a>
### linux       @ install_packages/setup_python_3-->install

```
apt-get install -y python3-tk

python36 -m pip install pycocotools

apt-get install -y python3.6-apt

cd /usr/lib/python3/dist-packages
sudo ln -s apt_pkg.cpython-36m-x86_64-linux-gnu.so

```
<a id="all___install_packages_setup_python_3_"></a>
### all       @ install_packages/setup_python_3-->install
```
python3 -m pip install cython numpy scipy 

python3 -m pip install scikit-learn scikit-image scikit-video pandas matplotlib screeninfo imageio pillow imutils prettytable color_transfer colorlog lxml tabulate tqdm paramiko xlwt contextlib2 paramparse psutil keyboard mouse Pillow mss dictances packaging jpeg4py xgboost pyperclip pyautogui spacy docstring_parser PyYAML easydict pykalman portalocker gdown imagesize terminaltables pycocotools paramparse

python3 -m pip install pyqt5
python3 -m pip install -U Pillow

```
<a id="pycocotools___all_install_packages_setup_python_3_"></a>
#### pycocotools       @ all/install_packages/setup_python_3-->install

__pycocotools/_mask.c:4:10: fatal error: Python.h: No such file or directory__

```
sudo apt update
sudo apt-get install python3.6-dev
pip install cython
pip install pycocotools
```

<a id="lapsolver___all_install_packages_setup_python_3_"></a>
#### lapsolver       @ all/install_packages/setup_python_3-->install
__install_cmake__
```
apt install cmake
```

```
python36 -m pip install lapsolver

```

might need scipy 1.3.3 due to [this](https://discuss.pytorch.org/t/torchvisions-inception-v3-takes-much-longer-to-load-than-other-models/68756/13) issue:

```
python36 -m pip install --upgrade scipy==1.3.3
```

**Note**: 

1. PyQt5 is known to have compatibility issues with the version of freetype library that comes with Ubuntu 14.04 that may prevent the labeling tool from working.

2. There might be a PIL version related error. Fix it by uninstalling the old version:

```
sudo apt install python3-pil
```

and running the python36 -m pip command again

<a id="paramiko_issue___all_install_packages_setup_python_3_"></a>
#### paramiko_issue       @ all/install_packages/setup_python_3-->install

AttributeError: cffi library '_sodium' has no function, constant or global variable named 'crypto_aead_chacha20poly1305_ietf_keybytes'

```
python36 -m pip install --upgrade PyNaCl
```

<a id="windows___install_packages_setup_python_3_"></a>
### windows       @ install_packages/setup_python_3-->install
__vcredist__
```
https://aka.ms/vs/16/release/vc_redist.x64.exe
```

```
python3 -m pip install pywin32 pywinauto pyautogui pyexcel pyexcel-ods3 winrt openpyxl

```
<a id="aes___windows_install_packages_setup_python_3_"></a>
#### AES       @ windows/install_packages/setup_python_3-->install
__Build Tools for Visual Studio 2019__
```
https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
```

```
python36 -m pip install pycryptodomex --no-binary :all:
```
<a id="x32___windows_install_packages_setup_python_3_"></a>
#### x32       @ windows/install_packages/setup_python_3-->install
```
python3x32 -m pip install cython numpy scipy 
python3x32 -m pip install scikit-learn scikit-image pandas matplotlib screeninfo imageio pillow imutils prettytable color_transfer lxml tabulate tqdm paramiko xlwt contextlib2 paramparse psutil keyboard mouse Pillow mss dictances packaging jpeg4py lapsolver xgboost pyperclip portalocker

python3x32 -m pip install pycocotools
python3x32 -m pip install pyqt5
python3x32 -m pip install -U Pillow

python3x32 -m pip install pywin32 pywinauto
wget https://bootstrap.pypa.io/get-pip.py

python3x32 get-pip.py
```


<a id="ffmpeg___setup_python_3_"></a>
## ffmpeg       @ setup_python_3-->install
```
sudo apt update
sudo apt install ffmpeg
```


<a id="pytorch_and_vis_tools___setup_python_3_"></a>
## pytorch_and_vis_tools       @ setup_python_3-->install

<a id="gpu___pytorch_and_vis_tools_setup_python_3_"></a>
### gpu       @ pytorch_and_vis_tools/setup_python_3-->install
```
python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

python3 -m pip install torch==1.12.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

```

<a id="grs___ubuntu_18_04___cuda11___gpu_pytorch_and_vis_tools_setup_python_3_"></a>
#### grs_/_ubuntu_18.04_/_cuda11       @ gpu/pytorch_and_vis_tools/setup_python_3-->install

```
python3 -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

<a id="latest___pytorch_and_vis_tools_setup_python_3_"></a>
### latest       @ pytorch_and_vis_tools/setup_python_3-->install

```
python36 -m pip install torch torchvision tensorboardX visdom pytorch-ignite

python36 -m pip install torch torchvision tensorboardX visdom pytorch-ignite
python36 -m pip uninstall torch torchvision

python2 -m pip install torch torchvision tensorboardX visdom pytorch-ignite

```

<a id="v1_0___pytorch_and_vis_tools_setup_python_3_"></a>
### v1.0       @ pytorch_and_vis_tools/setup_python_3-->install

versions > 1.0.0 have compatibility issues with yolo 3 causing NaN loss

```

python36 -m pip install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
python36 -m pip install torchvision==0.2.2
python36 -m pip install packaging tensorboardX visdom pytorch-ignite pytorch-lightning

python2 -m pip install torch torchvision tensorboardX visdom
```


<a id="linux_python3_8_cuda_10_0___v1_0_pytorch_and_vis_tools_setup_python_3_"></a>
#### linux_python3.8/cuda_10.0       @ v1.0/pytorch_and_vis_tools/setup_python_3-->install
```

python36 -m pip install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp38-cp38m-linux_x86_64.whl
python36 -m pip install torchvision==0.2.2
python36 -m pip install packaging tensorboardX visdom pytorch-ignite pytorch-lightning

```

<a id="linux_python3_6_cuda_10_0___v1_0_pytorch_and_vis_tools_setup_python_3_"></a>
#### linux_python3.6/cuda_10.0       @ v1.0/pytorch_and_vis_tools/setup_python_3-->install
```
python36 -m pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
python36 -m pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
```

<a id="apex___linux_python3_6_cuda_10_0_v1_0_pytorch_and_vis_tools_setup_python_3_"></a>
##### apex       @ linux_python3.6/cuda_10.0/v1.0/pytorch_and_vis_tools/setup_python_3-->install
```
git clone https://www.github.com/nvidia/apex
cd apex
python36 setup.py install --cuda_ext --cpp_ext

python2 setup.py install --cuda_ext --cpp_ext

python36 - m "import torch; print(torch.__version__)"
```

<a id="windows_python3_7_cuda_10_0___pytorch_and_vis_tools_setup_python_3_"></a>
### windows_python3.7/cuda_10.0       @ pytorch_and_vis_tools/setup_python_3-->install
```

https://download.pytorch.org/whl/cu100/

python36 -m pip install https://download.pytorch.org/whl/cu100/torch-1.2.0-cp37-cp37m-win_amd64.whl

python36 -m pip install https://download.pytorch.org/whl/cu100/torchvision-0.4.0-cp37-cp37m-win_amd64.whl

```
<a id="opencv___setup_python_3_"></a>
## opencv       @ setup_python_3-->install

<a id="4_5_4_60___opencv_setup_python_3_"></a>
### 4.5.4.60       @ opencv/setup_python_3-->install
needed for fiftyone
```
python3 -m pip install opencv-python==4.5.4.60 opencv-contrib-python==4.5.4.60

python3 -m pip install opencv-python

```

<a id="4_1_0___opencv_setup_python_3_"></a>
### 4.1.0       @ opencv/setup_python_3-->install
```
python36 -m pip install opencv-python==4.1.0.25 opencv-contrib-python==4.1.0.25
python36 -m pip install opencv-python==4.1.0.25 opencv-contrib-python==4.1.0.25

```

<a id="3_4_11_45___opencv_setup_python_3_"></a>
### 3.4.11.45       @ opencv/setup_python_3-->install
```
python -m pip install opencv-python==3.4.11.45 opencv-contrib-python==3.4.11.45
```
<a id="3_4_5___opencv_setup_python_3_"></a>
### 3.4.5       @ opencv/setup_python_3-->install
```
python3 -m pip install opencv-python==3.4.18.65 opencv-contrib-python==3.4.18.65

python3 -m pip install opencv-python==3.4.5.20 opencv-contrib-python==3.4.5.20
python36 -m pip install opencv-python opencv-contrib-python

python3x32 -m pip install opencv-python==3.4.5.20 opencv-contrib-python==3.4.5.20

python36 -m pip uninstall opencv-python opencv-contrib-python
```
<a id="uninstall___3_4_5_opencv_setup_python_3_"></a>
#### uninstall       @ 3.4.5/opencv/setup_python_3-->install
```
python36 -m pip uninstall opencv-python opencv-contrib-python
```
opencv 4 might have compatibility issues so opencv 3 is recommended

<a id="3_4_4_19_cc___opencv_setup_python_3_"></a>
### 3.4.4.19_CC       @ opencv/setup_python_3-->install
```
python36 -m pip install opencv-python==3.4.4.19 opencv-contrib-python==3.4.4.19
```

<a id="tensorflow___setup_python_3_"></a>
## tensorflow       @ setup_python_3-->install

<a id="check_version___tensorflow_setup_python_3_"></a>
### check_version       @ tensorflow/setup_python_3-->install
```
python3 -c "import tensorflow as tf; print(tf.__version__)"
python3 -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

<a id="check_gpu___tensorflow_setup_python_3_"></a>
### check_gpu       @ tensorflow/setup_python_3-->install
```
python3 -c "import tensorflow as tf; sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))"
```

<a id="install___tensorflow_setup_python_3_"></a>
### install       @ tensorflow/setup_python_3-->install

```
python3 -m pip install --upgrade tensorflow_gpu tensorboard
python36 -m pip install --upgrade tensorflow_gpu
```


<a id="1_14___tensorflow_setup_python_3_"></a>
### 1.14       @ tensorflow/setup_python_3-->install

__Note__
reuires cuda 10.0 / cudnn 7: cuda_10_0___ubuntu_18_04_install_cud_a_, cuda_10_0___install_cudn_n_

```
python36 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.14.0-cp36-cp36m-linux_x86_64.whl

python36 -m pip install --user gast==0.2.2

python36 -m pip install tensorboard
```

<a id="python3_8___1_14_tensorflow_setup_python_3_"></a>
#### python3.8       @ 1.14/tensorflow/setup_python_3-->install

```
python36 -m pip install --upgrade tensorflow_gpu==1.14.0

pip3 install --user gast==0.2.2
```

<a id="from_source___1_14_tensorflow_setup_python_3_"></a>
#### from_source       @ 1.14/tensorflow/setup_python_3-->install

```

python36 -m pip install keras_applications==1.0.4 --no-deps
python36 -m pip install keras_preprocessing==1.0.2 --no-deps

python2 -m pip install keras_applications==1.0.4 --no-deps
python2 -m pip install keras_preprocessing==1.0.2 --no-deps

wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh
chmod +x bazel-0.25.2-installer-linux-x86_64.sh
./bazel-0.25.2-installer-linux-x86_64.sh --user

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.14
./configure

bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
python36 -m pip install /tmp/tensorflow_pkg/tensorflow-1.14.0-cp36-cp36m-linux_x86_64.whl
python36 -m pip install /tmp/tensorflow_pkg/tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl
```

<a id="windows___tensorflow_setup_python_3_"></a>
### windows       @ tensorflow/setup_python_3-->install

```
python36 -m pip install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.2.0-cp37-cp37m-win_amd64.whl
python36 -m pip install --upgrade tensorflow_gpu==1.13.0
python36 -m pip install --upgrade tensorflow_gpu==1.14.0

pip install --upgrade tensorflow_gpu==1.13.0

```

<a id="v1_6_for_cuda_9_0___tensorflow_setup_python_3_"></a>
### v1.6_for_cuda_9.0       @ tensorflow/setup_python_3-->install

<a id="python_3_5___v1_6_for_cuda_9_0_tensorflow_setup_python_3_"></a>
#### python_3.5       @ v1.6_for_cuda_9.0/tensorflow/setup_python_3-->install

```
python36 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp35-cp35m-linux_x86_64.whl

```

<a id="python_3_6___v1_6_for_cuda_9_0_tensorflow_setup_python_3_"></a>
#### python_3.6       @ v1.6_for_cuda_9.0/tensorflow/setup_python_3-->install

```
python36 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp36-cp36m-linux_x86_64.whl
```

<a id="v1_6_for_cuda_8_0___tensorflow_setup_python_3_"></a>
### v1.6_for_cuda_8.0       @ tensorflow/setup_python_3-->install
Tensorflow does not provide an official installer for v1.6 that supports cuda 8.0 so a prebuilt installer available [here](https://drive.google.com/open?id=1m0hDMsmRn1LufagccUP5I9RoOiJPc-ML) must be used instead.

```
python2 -m pip install --upgrade tensorflow-1.6.0-cp35-cp35m-linux_x86_64.whl
```

Please ensure that a file called ```/usr/local/cuda/lib64/libcudnn.6.0``` is available on the system. If not, run this *before* running the pip command:

```
ln -s /usr/local/cuda-8.0/lib64/libcudnn.6 /usr/local/cuda-8.0/lib64/libcudnn.6.0
```

<a id="v1_4_for_cuda_8_0___tensorflow_setup_python_3_"></a>
### v1.4_for_cuda_8.0       @ tensorflow/setup_python_3-->install

```
python36 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0rc1-cp34-cp34m-linux_x86_64.whl
```



<a id="theano_and_keras___setup_python_3_"></a>
## theano_and_keras       @ setup_python_3-->install

```

git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray

mkdir Build
cd Build

cmake .. -DCMAKE_BUILD_TYPE=Release # or Debug if you are investigating a crash
make
make install
cd ..

python2 setup.py build
python2 setup.py install

python36 setup.py build
python36 setup.py install

python2 -m pip install Theano
python2 -m pip install keras

python36 -m pip install Theano
python36 -m pip install keras

```


<a id="setup_python_2_"></a>
# setup python 2

```
sudo apt-get install python-dev
```

<a id="install_packages___setup_python_2_"></a>
## install_packages       @ setup_python_2-->install

```
sudo apt-get install python-tk
python2 -m pip install cython numpy scipy sklearn scikit-image pandas matplotlib  pillow imutils prettytable color_transfer tqdm

python2 -m pip install screeninfo  lxml tabulate paramiko xlwt contextlib2 paramparse psutil keyboard mouse Pillow dictances packaging jpeg4py pyperclip spacy docstring_parser easydict
python2 -m pip install pycocotools
python2 -m pip install PyQt4


```

<a id="windows___install_packages_setup_python_2_"></a>
### windows       @ install_packages/setup_python_2-->install

```
python2 -m pip install pywin32 pywinauto

```


<a id="install_opencv___setup_python_2_"></a>
## install_opencv       @ setup_python_2-->install

<a id="4_1_0___install_opencv_setup_python_2_"></a>
### 4.1.0       @ install_opencv/setup_python_2-->install

```
python2 -m pip install opencv-python==4.1.0.25 opencv-contrib-python==4.1.0.25
```

<a id="3_4_5___install_opencv_setup_python_2_"></a>
### 3.4.5       @ install_opencv/setup_python_2-->install

```
python2 -m pip install opencv-python==3.4.5.20 opencv-contrib-python==3.4.5.20

```
<a id="uninstall___install_opencv_setup_python_2_"></a>
### uninstall       @ install_opencv/setup_python_2-->install
```
python2 -m pip uninstall opencv-python opencv-contrib-python

```

opencv 4 might have compatibility issues so opencv 3 is recommended

<a id="tensorflow___setup_python_2_"></a>
## tensorflow       @ setup_python_2-->install

<a id="1_14___tensorflow_setup_python_2_"></a>
### 1.14       @ tensorflow/setup_python_2-->install
```
python2 -m pip install tensorflow_gpu==1.14.0
```

<a id="v1_6_for_cuda_9_0___tensorflow_setup_python_2_"></a>
### v1.6_for_cuda_9.0       @ tensorflow/setup_python_2-->install
```
python2 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp27-none-linux_x86_64.whl
```

<a id="v1_6_for_cuda_8_0___tensorflow_setup_python_2_"></a>
### v1.6_for_cuda_8.0       @ tensorflow/setup_python_2-->install
Tensorflow does not provide an official installer for v1.6 that supports cuda 8.0 so a prebuilt installer available [here](https://drive.google.com/open?id=11F69SNYVE4nY7Pfw6XOgQtfS_1fCFnaJ) must be used.

```
python2 -m pip install --upgrade tensorflow-1.6.0-cp27-cp27mu-linux_x86_64.whl
```
Please ensure that a file called ```/usr/local/cuda-8.0/lib64/libcudnn.6.0``` is available on the system. If not, run this *before* running the pip command:

```
ln -s /usr/local/cuda-8.0/lib64/libcudnn.6 /usr/local/cuda-8.0/lib64/libcudnn.6.0
```

<a id="v1_4_for_cuda_8_0___tensorflow_setup_python_2_"></a>
### v1.4_for_cuda_8.0       @ tensorflow/setup_python_2-->install

```
python2 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0rc1-cp27-none-linux_x86_64.whl
```

<a id="windows_python3_7___tensorflow_setup_python_2_"></a>
### windows/python3.7       @ tensorflow/setup_python_2-->install

```
python36 -m pip install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.13.1-cp37-cp37m-win_amd64.whl

python36 -m pip install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.6.0-cp37-cp37m-win_amd64.whl
```

<a id="no_gpu___windows_python3_7_tensorflow_setup_python_2_"></a>
#### no_gpu       @ windows/python3.7/tensorflow/setup_python_2-->install

```
python36 -m pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.13.1-cp37-cp37m-win_amd64.whl

python36 -m pip install --upgrade https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/libsvm-3.23-cp37-cp37m-win_amd64.whl
```
<a id="misc_tool_s_"></a>
# misc tools
<a id="jedit___misc_tools_"></a>
## jedit       @ misc_tools-->install

```
sudo apt-get install jedit
```

<a id="imagemagick_7___misc_tools_"></a>
## imagemagick_7       @ misc_tools-->install

```
wget https://www.imagemagick.org/download/ImageMagick.tar.gz
tar xvzf ImageMagick.tar.gz
cd ImageMagick-7.0.8-60/
./configure 
make -j8
sudo make install 
sudo ldconfig /usr/local/lib

```

7.0.8-60 might need adapting based on the latest available version that gets downloaded.

<a id="pycharm___misc_tools_"></a>
## pycharm       @ misc_tools-->install

```
sudo snap install pycharm-community --classic
```


<a id="jpeg4py___misc_tools_"></a>
## jpeg4py       @ misc_tools-->install

```
python36 -m pip install jpeg4py 
sudo apt-get install libturbojpeg
```

<a id="multi_python_issues_"></a>
# multi_python_issues

<a id="python_3_6_and_3_5___multi_python_issue_s_"></a>
## python_3.6_and_3.5       @ multi_python_issues-->install
```
cd /usr/lib/python3/dist-packages

ln -s /usr/bin/python3.5 /usr/bin/python3

sudo cp _dbus_bindings.cpython-35m-x86_64-linux-gnu.so _dbus_bindings.cpython-36m-x86_64-linux-gnu.so
sudo cp _dbus_glib_bindings.cpython-35m-x86_64-linux-gnu.so cups.cpython-36m-x86_64-linux-gnu.so
sudo cp cups.cpython-35m-x86_64-linux-gnu.so cups.cpython-36m-x86_64-linux-gnu.so

```

<a id="optiona_l_"></a>
# optional 

<a id="network_printers___optional_"></a>
## network_printers       @ optional-->install

```
 ipp://cups.cs.ualberta.ca/printers/csc228-cpy
 ipp://cups.cs.ualberta.ca/printers/cs351
```

<a id="install_opencv_from_source___optional_"></a>
## install_opencv_from_source       @ optional-->install

<a id="windows___install_opencv_from_source_optiona_l_"></a>
### windows       @ install_opencv_from_source/optional-->install

getting 3.x.x to work with VS 2017 or newer:

1. add 
```
  elseif(MSVC_VERSION MATCHES "^192[0-9]$")
    set(OpenCV_RUNTIME vc16)
```
right after
```
  elseif(MSVC_VERSION EQUAL 1900)
    set(OpenCV_RUNTIME vc14)
  elseif(MSVC_VERSION MATCHES "^191[0-9]$")
    set(OpenCV_RUNTIME vc15)
```
in `OpenCVConfig.cmake`.
2. move bin, lib and include folders to `x64/vc16/` after creating the same.

<a id="linux___install_opencv_from_source_optiona_l_"></a>
### linux       @ install_opencv_from_source/optional-->install

add one of these at the top of CMakeLists.txt to avoid errors originating from too new C++ versions

```
set (CMAKE_CXX_STANDARD 98)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD 14)

```

```
cd ~
sudo apt-get install dos2unix

dos2unix acamp_code/install_cv_345.sh
chmod +x acamp_code/install_cv_345.sh

acamp_code/install_cv_345.sh

```

detailed commands if this doesn't work:

```
sudo apt-get -y update

sudo apt-get -y upgrade

sudo apt-get -y install build-essential checkinstall cmake pkg-config yasm
sudo apt-get -y install git gfortran

sudo apt-get -y install libjpeg8-dev libjasper-dev libpng12-dev 

sudo apt-get -y install libtiff5-dev
 
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev

sudo apt-get -y install libxine2-dev libv4l-dev

sudo apt-get -y install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev

sudo apt-get -y install qt5-default libgtk2.0-dev libtbb-dev

sudo apt-get -y install libatlas-base-dev

sudo apt-get -y install libfaac-dev libmp3lame-dev libtheora-dev

sudo apt-get -y install libvorbis-dev libxvidcore-dev

sudo apt-get -y install libopencore-amrnb-dev libopencore-amrwb-dev

sudo apt-get -y install x264 v4l-utils
 
sudo apt-get -y install libprotobuf-dev protobuf-compiler

sudo apt-get -y install libgoogle-glog-dev libgflags-dev

sudo apt-get -y install libgphoto2-dev libhdf5-dev doxygen

sudo apt-get -y install pyqt5-dev-tools

```

<a id="3_4_5___linux_install_opencv_from_source_optiona_l_"></a>
#### 3.4.5       @ linux/install_opencv_from_source/optional-->install
```
wget https://github.com/opencv/opencv/archive/3.4.5.zip

unzip 3.4.5.zip

rm 3.4.5.zip

wget https://github.com/opencv/opencv_contrib/archive/3.4.5.zip

unzip 3.4.5.zip

rm 3.4.5.zip

cd opencv-3.4.5

mkdir build && cd build



cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.5/modules \
      -D WITH_CUDA=ON ..   

```


<a id="3_4_1___linux_install_opencv_from_source_optiona_l_"></a>
#### 3.4.1       @ linux/install_opencv_from_source/optional-->install

something weird and anoying in MDP causes segfault when compiled with 3.4.5

```
wget https://github.com/opencv/opencv/archive/3.4.1.zip

unzip 3.4.1.zip

rm 3.4.1.zip

wget https://github.com/opencv/opencv_contrib/archive/3.4.1.zip

unzip 3.4.1.zip

rm 3.4.1.zip

cd opencv-3.4.1

mkdir build && cd build

```

<a id="cuda___linux_install_opencv_from_source_optiona_l_"></a>
#### cuda       @ linux/install_opencv_from_source/optional-->install
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D BUILD_opencv_cudacodec=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.1/modules \
      -D CUDA_ARCH_BIN=8.6 \
      -D WITH_CUDA=ON .. 
```
<a id="no_cuda___linux_install_opencv_from_source_optiona_l_"></a>
#### no_cuda       @ linux/install_opencv_from_source/optional-->install
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D BUILD_opencv_cudacodec=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.1/modules \
      -D CUDA_ARCH_BIN=8.6 \
      -D WITH_CUDA=OFF ..   
```

<a id="2_4_13_6___linux_install_opencv_from_source_optiona_l_"></a>
#### 2.4.13.6       @ linux/install_opencv_from_source/optional-->install
something weird and anoying in MDP causes segfault when compiled with 3.4.5

```
wget https://github.com/opencv/opencv/archive/2.4.13.6.zip

unzip 2.4.13.6.zip

rm 2.4.13.6.zip

wget https://github.com/opencv/opencv_contrib/archive/2.4.13.6.zip

unzip 2.4.13.6.zip

rm 2.4.13.6.zip

cd opencv-2.4.13.6

mkdir build && cd build



cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-2.4.13.6/modules \
        -D WITH_CUDA=OFF ..   

```

       
<a id="compile_and_install___linux_install_opencv_from_source_optiona_l_"></a>
#### compile_and_install       @ linux/install_opencv_from_source/optional-->install

```
make -j16
make -j12
sudo make install

sudo make uninstall

```

<a id="setup_point_grey_ethernet_camera___optional_"></a>
## setup_point_grey_ethernet_camera       @ optional-->install

1. download Spinnaker SDK, its python wrapper and ffmpeg dependencies from here:

https://www.ptgrey.com/support/downloads

after setting "Blackfly S" in "Product Family"

2. run:

<a id="ffmpeg___setup_point_grey_ethernet_camera_optiona_l_"></a>
### ffmpeg       @ setup_point_grey_ethernet_camera/optional-->install
```
unzip FFmpeg-5156578d1f486163d5b83f1d63246cd23d107933.zip

cd FFmpeg-5156578d1f486163d5b83f1d63246cd23d107933

./configure

make

sudo make install
```
<a id="sdk___setup_point_grey_ethernet_camera_optiona_l_"></a>
### sdk       @ setup_point_grey_ethernet_camera/optional-->install
```
tar -xvzf spinnaker-1.13.0.31-amd64-pkg.tar.gz

cd spinnaker-1.13.0.31-amd64

./install_spinnaker.sh
```
<a id="python_wrappers___setup_point_grey_ethernet_camera_optiona_l_"></a>
### python_wrappers       @ setup_point_grey_ethernet_camera/optional-->install
```
tar -xvf spinnaker_python_1.13.0.31_linux_x86_64.tar

tar -xvzf spinnaker_python-1.13.0.31-cp27-cp27mu-linux_x86_64.tar.gz

python2 -m pip install spinnaker_python-1.13.0.31-cp27-cp27mu-linux_x86_64.whl

tar -xvzf spinnaker_python-1.13.0.31-cp35-cp35m-linux_x86_64.tar.gz

python36 -m pip install spinnaker_python-1.13.0.31-cp35-cp35m-linux_x86_64.whl
```

<a id="setting_up_detector_training___optional_"></a>
## setting_up_detector_training       @ optional-->install

1. Clone this repo to the home folder:
```
https://github.com/abhineet123/models
```
2. Run this from ~/models/research:
```
protoc object_detection/protos/*.proto --python_out=.
```
If this produces an error, follow the instructions under ```Manual protobuf-compiler installation and usage``` section [here](https://github.com/abhineet123/models/blob/master/research/object_detection/g3doc/installation.md) 

3. Add `~/models/research` and `~/models/research/slim` to PYTHONPATH (already done in the bashrc lines mentioned before).

<a id="issue_s_"></a>
# issues 

<a id="broken_software_updates_in_16_04_due_to_python_35_36_mismatch___issues_"></a>
## broken_software_updates_in_16.04_due_to_python_35/36_mismatch       @ issues-->install

```
cd /usr/lib/python3/dist-packages/
ln -s _dbus_bindings.cpython-35m-x86_64-linux-gnu.so _dbus_bindings.so
ln -s _dbus_glib_bindings.cpython-35m-x86_64-linux-gnu.so _dbus_glib_bindings.so

update-manager
software-properties-gtk

```
<a id="gcc_"></a>
# gcc

<a id="6_3___gc_c_"></a>
## 6.3       @ gcc-->install
Remove any previous gcc-6 version installed:
```
sudo apt-get remove --purge gcc-6 g++-6 gcc-6-multilib
sudo apt autoremove
```
Add the gcc-6.3 PPA:
```
sudo add-apt-repository ppa:jonathonf/gcc-6.3
sudo apt-get update
```
Install the specific gcc-6.3 version, 6.3.0-21ubuntu1~16.04.york0 in our case:

```
VER=6.3.0-21ubuntu1~16.04.york0 && sudo apt-get install gcc-6-base=$VER gcc-6=$VER g++-6=$VER cpp-6=$VER libgcc-6-dev=$VER libstdc++-6-dev=$VER libasan3=$VER
```
Check the version of the just installed gcc with gcc-6 -v. Expected:

gcc version 6.3.0 20170628 (Ubuntu 6.3.0-21ubuntu1~16.04.york0)


```

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-6 

sudo update-alternatives --config gcc
gcc --version
g++ --version
```


<a id="7_0___gc_c_"></a>
## 7.0       @ gcc-->install

```
sudo apt-get install -y software-properties-common python-software-properties
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-7 -y

```

either 

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7 
```

or

```
sudo update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
    --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-7 \
    --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-7 \
    --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-7

sudo update-alternatives \
    --install /usr/bin/g++ g++ /usr/bin/g++-7 60 \
    --slave /usr/bin/g++-ar g++-ar /usr/bin/g++-ar-7 \
    --slave /usr/bin/g++-nm g++-nm /usr/bin/g++-nm-7 \
    --slave /usr/bin/g++-ranlib g++-ranlib /usr/bin/g++-ranlib-7

```

<a id="matlab_engine_"></a>
# matlab engine

```
cd /usr/local/MATLAB/R2020a/extern/engines/python
python36 setup.py install

```

<a id="mdp_mot_devkit___matlab_engin_e_"></a>
## mdp_mot_devkit       @ matlab_engine-->install

```
cd evaluation/devkit/matlab_devkit
matlab
compile
```



<a id="vot_devkit___matlab_engin_e_"></a>
# vot_devkit       @ matlab_engine-->install

```
mkvirtualenv vot
python3 -m pip install git+https://github.com/votchallenge/vot-toolkit-python

vot initialize vot2017 --workspace /data/vot2017
vot initialize vot2018 --workspace /data/vot2018

```






