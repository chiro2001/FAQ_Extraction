#!/bin/bash
#SBATCH -J qa                          # 作业名为 test
#SBATCH -o slurm-%j.out                       # 屏幕上的输出文件重定向到 slurm-%j.out , %j 会替换成jobid
#SBATCH -e slurm-%j.err                       # 屏幕上的错误输出文件重定向到 slurm-%j.err , %j 会替换成jobid
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --cpus-per-task=4                     # 单任务使用的 CPU 核心数为 4
#SBATCH -t 8:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:tesla_p100-pcie-16gb:1
#SBATCH --mem=10GB

source ~/.bashrc

conda init bash
conda activate hsj-py3.8

python -u run_qa.py config/qa_config.json &> roberta.log
