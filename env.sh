# include python path
PYTHONPATH="/home/ruard/code/caschnet/:/home/ruard/code/caschnet/schnetpack/src/:$PYTHONPATH"
export PYTHONPATH
PYSCF_EXT_PATH="/home/ruard/anaconda3/envs/caschnet/lib/python3.8/site-packages/pyscf"
export PYSCF_EXT_PATH

# set files base dir
export base_dir='/home/rhjvanworkum/'

# set WANDB project for training logs
export WANDB_PROJECT='caschnet'

# set PySCF OMP threads for parallelization
export OMP_NUM_THREADS=1