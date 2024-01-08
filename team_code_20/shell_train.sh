#!/bin/bash

export CARLA_ROOT=${1:-/home/hcis-s15/Documents/projects/SRL/carla}
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:../scenario_runner

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hcis-server/miniconda3/lib

export OMP_NUM_THREADS=40  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=42353467 --rdzv_backend=c10d train.py --id stp3_wp --batch_size 16 --setting all --root_dir /media/user/data3/SRL/SRL_data/training_data_final --logdir /media/user/data3/SRL/checkpoints --load_file /media/user/data3/SRL/checkpoints/stp3/model_0030.pth --use_bev_semantic 0 --use_semantic 0 --use_depth 0 --use_route 0 --detect_boxes 0 --use_controller_input_prediction 0 --use_wp_gru 1 --use_discrete_command 1 --use_tp 1 --continue_epoch 0 --cpu_cores 40 --freeze_backbone 1
