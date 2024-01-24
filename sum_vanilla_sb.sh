#!/bin/bash
#SBATCH --job-name=sum
#SBATCH --output=sum.txt
#SBATCH --mail-user=tim.kolber@stud.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --error=/pfs/data5/home/hd/hd_hd/hd_go226/outputs/slurm/slurm-%j.err
#SBATCH --output=/pfs/data5/home/hd/hd_hd/hd_go226/outputs/slurm/slurm-%j.out
# JOB STEPS
python /pfs/data5/home/hd/hd_hd/hd_go226/bachelor/prompting_experiments.py --temp 0.7 --top_p 0.95 --rep_penalty 1.15 --results vicuna1.1_vanilla_august --prompt_mode vanilla --model_name lmsys/vicuna-13b-v1.1 --ds /pfs/data5/home/hd/hd_hd/hd_go226/bachelor/august_results.xlsx

