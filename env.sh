# include python path
PYTHONPATH="/mnt/c/users/rhjva/imperial/caschnet/:/mnt/c/users/rhjva/imperial/caschnet/schnetpack/src/:$PYTHONPATH"
export PYTHONPATH

# set files base dir
export base_dir='/mnt/c/users/rhjva/imperial/fulvene/'

# set WANDB project for training logs
export WANDB_PROJECT='caschnet_pyscf'

# set PySCF OMP threads for parallelization
export OMP_NUM_THREADS=16