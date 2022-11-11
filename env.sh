# include python path
PYTHONPATH="/home/ubuntu/caschnet/:/home/ubuntu/caschnet/schnetpack/src/:$PYTHONPATH"
export PYTHONPATH

# set files base dir
export base_dir='/home/ubuntu/fulvene/'

# set WANDB project for training logs
export WANDB_PROJECT='caschnet'

# set PySCF OMP threads for parallelization
export OMP_NUM_THREADS=16