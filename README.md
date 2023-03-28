# CasNet
A repository used to perform experiments for the work on CasNet.

### 1. Installation & Setup
type in command line:
- git clone git@github.com:rhjvanworkum/CasNet.git
- git submodule init
- git submodule update
- conda env create -f environment.yml
- conda activate casnet
- cd pyscf
- pip install -e . (this might take a while)
- source env.sh (after setting variables in env.sh file)

This should set up the whole environment 
(Watch out: We're our own fork a of PySCF that returns macro, micro & inner iterations of the CASSCF object)

<br>
<br>

### 2. Performing CASSCF calculations in PySCF
##### 2.1 Run calculations
`python data/casscf/pyscf/run_casscf_calculations.py --geometry_folder geometries/geom_scan_200/ --output_folder pyscf/geom_scan_200_sto_6g/ --basis sto_6g `
##### 2.2 Analyze results
`python data/check_casscf_calculations.py --output_folder pyscf/geom_scan_200_sto_6g/`
##### 2.3 Save results in ASE DB
`python data/db/save_casscf_calculations_to_db.py --geometry_folder geometries/geom_scan_200/ --output_folder pyscf/geom_scan_200_sto_6g/`
##### 2.4 Generate split
check data/splits/ to see example of split generating scripts

<br>
<br>

### 3. Training models
##### 3.1 Train ML-MO model
`python train_model.py --db_name geom_scan_200_sto_6g.db --split_name geom_scan_200.npz --property mo_coeffs_adjusted --model_name gs200_sto_6g_F`
##### 3.2 Train ML-F model
`python train_model.py --db_name geom_scan_200_sto_6g.db --split_name geom_scan_200.npz --property F --model_name gs200_sto_6g_F`
##### 3.3 Train PhiSNet model
Add a configuration in phisnet_fork/configurations/ folder (e.g. named 'test_conf.txt') and then type:\
`python phisnet_fork/train.py @phisnet_fork/configurations/test_conf.txt`

<br>
<br>

### 4. Evaluating models
##### 4.1 Evaluate model loss on train/val/test set
`python evaluation/compute_model_loss.py --db_name fulvene_geom_scan_250.db --split_name fulvene_gs_250_inter.npz --mo_model geom_scan_250/fulvene_gs250_MO --F_model geom_scan_250/fulvene_gs250_F --phisnet_model geom_scan_250/fulvene_gs250_phisnet`
##### 4.2 Evaluate orbitals on validation/test set
`python evaluation/compute_SCF_iterations.py  --geometry_folder geometries/geom_scan_200/ --split_name geom_scan_200.npz --mode ML-MO --model gs200_sto_6g_F --basis sto_6g`

`python evaluation/compute_CASCI_energy_errors.py  --geometry_folder geometries/geom_scan_200/ --split_name geom_scan_200.npz --mode ML-MO --model gs200_sto_6g_F --basis sto_6g`
##### 4.3 Look at predicted orbitals for a particular geometry
`python evaluation/paper_figures/plot_orbitals.py` \
(check this file for setting geometry & orbital model & MO indices)