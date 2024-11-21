# export SLURM_JOB_ID=3729308
gpu_num=4
srun --partition=MoE --mpi=pmi2 --gres=gpu:${gpu_num} -n1 --ntasks-per-node=1 --job-name=cgj --kill-on-bad-exit=1 --quotatype=auto \
torchrun --nnodes=1 --nproc_per_node=${gpu_num} --master_port=29520 sample_ddp.py # sample_npz.py