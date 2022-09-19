#!/bin/bash

source ~/.bashrc
echo "Start to intall the run package"
WORK_DIR=start_1.7
RUN_DIR=run
mindspore_file=mindspore_ascend-1.7.0-cp37-cp37m-linux_aarch64.whl
LOCAL_DIR=$(cd "$(dirname "$0")";pwd)
echo $LOCAL_DIR

echo "===current dir="
ls ./${WORK_DIR}/${RUN_DIR}

pip install ./${WORK_DIR}/${mindspore_file} -i http://100.125.33.126:8888/repository/pypi/simple --trusted-host=100.125.33.126
sudo chmod +755 -R /usr/local/Ascend/nnae
sudo rm -rf /usr/local/Ascend/nnae

sudo chmod +x ./${WORK_DIR}/${RUN_DIR}/*.run
sudo bash ./${WORK_DIR}/${RUN_DIR}/Ascend* --full --quiet

export HCCL_CONNECT_TIMEOUT=1800 # 通信建链最长等待时间，单位s

echo "======/usr/local/Ascend======"
ls -al /usr/local/Ascend
echo "======/usr/local/Ascend/ascend-toolkit/======"
ls -al /usr/local/Ascend/ascend-toolkit/
echo "======/usr/local/Ascend/ascend-toolkit/latest======"
ls -al /usr/local/Ascend/ascend-toolkit/latest
echo "======/usr/local/Ascend/driver/lib64========"
ls -al /usr/local/Ascend/driver/lib64
echo "======/usr/local/Ascend/driver/lib64/common======="
ls -al /usr/local/Ascend/driver/lib64/common
echo "=======/usr/local/Ascend/driver/lib64/driver======="
ls -al /usr/local/Ascend/driver/lib64/driver
echo "============/usr/local/Ascend/ascend-toolkit/5.1.RC1============="
ls -al /usr/local/Ascend/ascend-toolkit/5.1.RC1
sudo mkdir /usr/local/Ascend/nnae
sudo chmod +755 -R /usr/local/Ascend/nnae
#sudo mkdir /usr/local/Ascend/nnae/latest
#sudo chmod +755 -R /usr/local/Ascend/nnae/latest
sudo ln -s /usr/local/Ascend/ascend-toolkit/5.1.RC1 /usr/local/Ascend/nnae/latest
echo "======/usr/local/Ascend/nnae======"
ls -al /usr/local/Ascend/nnae
echo "======/usr/local/Ascend/nnae/latest======"
ls -al /usr/local/Ascend/nnae/latest
echo "======/usr/local/Ascend/nnae/latest/lib64/libhccl.so======"
ls -al /usr/local/Ascend/nnae/latest/lib64/libhccl.so

# sudo cp -fp ${LOCAL_DIR}/${WORK_DIR}/libhccl.so /usr/local/Ascend/nnae/latest/lib64/libhccl.so
echo "======/usr/local/Ascend/nnae/latest/lib64/libhccl.so======"
ls -al /usr/local/Ascend/nnae/latest/lib64/libhccl.so

echo "======/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm.py======"
ls -al /usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm.py

echo "======/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm.py======"
ls -al /usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm_x_backprop_v2.py


sudo cp -fp ${LOCAL_DIR}/${WORK_DIR}/layer_norm.py /usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm.py
sudo cp -fp ${LOCAL_DIR}/${WORK_DIR}/layer_norm_x_backprop_v2.py /usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm_x_backprop_v2.py

chmod +777 /usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm_x_backprop_v2.py
chmod +777 /usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm.py

echo "======/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm.py====new=="
ls -al /usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm.py

echo "======/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm.py====new=="
ls -al /usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/layer_norm_x_backprop_v2.py

ls -al ${LOCAL_DIR}/${WORK_DIR}/custom_tune_bank_new

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:$LD_LIBRARY_PATH
export PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:$PATH
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest:$ASCEND_HOME_PATH

echo "-------------------uninstall te topi and hccl--------------------------"
sudo pip uninstall te -y
sudo pip uninstall topi -y
sudo pip uninstall hccl -y
echo "-------------------install te topi and hccl--------------------------"
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/topi-0.4.0-py3-none-any.whl 
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-0.4.0-py3-none-any.whl 
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-0.1.0-py3-none-any.whl 
pip install /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_parser-0.1-py3-none-any.whl


export GLOG_v=3 # mindspore日志开关，1：Info, 2:Warning, 3:Error
export ASCEND_GLOBAL_LOG_LEVEL=3 # 底层软件的日志级别开关 1：Info, 2:Warning, 3:Error
export ASCEND_GLOBAL_EVENT_ENABLE=1 # 底层软件的日志event日志开关 0：disable, 1:enable
export ASCEND_SLOG_PRINT_TO_STDOUT=0 # 是否把底层日志重定向到打屏，0：disable, 1:enable

export ENABLE_TUNE_BANK=True
export TUNE_BANK_PATH=${LOCAL_DIR}/${WORK_DIR}/custom_tune_bank_new

env

mkdir -p /cache/ckpts
mkdir -p /home/work/sfs/cache/${BATCH_JOB_ID}/1
mkdir -p /home/work/sfs/cache/${BATCH_JOB_ID}/2

sudo chmod +777 -R /cache/ckpts
sudo chmod +777 -R /home/work/sfs/cache/${BATCH_JOB_ID}

export GROUP_INFO_FILE=/home/work/sfs/cache/${BATCH_JOB_ID}/group_info_file.pb
