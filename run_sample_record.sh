srun --partition=MoE --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=cgj --kill-on-bad-exit=1 --quotatype=auto \
python sample_record.py --ckpt /mnt/petrelfs/chenguanjie/cgj/DiT-skip/ckpt/DiT-400k.pt --model DiT-cache-3

# /mnt/petrelfs/chenguanjie/cgj/DiT-skip/ckpt/DiT-800k-origin.pt
#  /mnt/petrelfs/chenguanjie/cgj/DiT-skip/ckpt/DiT-400k.pt