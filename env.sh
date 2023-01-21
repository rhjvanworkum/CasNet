# include python path
PYTHONPATH="/home/ruard/code/caschnet/:/home/ruard/code/schnetpack/src/:/home/ruard/code/caschnet/phisnet_fork/:$PYTHONPATH"
export PYTHONPATH

# include PySCF path
PYSCF_EXT_PATH="/home/ruard/anaconda3/envs/caschnet/lib/python3.8/site-packages/pyscf"
export PYSCF_EXT_PATH

# set files base dir
export base_dir='/home/ruard/Documents/experiments/fulvene/'

# set WANDB project for training logs
export WANDB_PROJECT='caschnet'

# set PySCF OMP threads for parallelization
export OMP_NUM_THREADS=2