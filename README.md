# caschnet
make changes to pyscf & re-compile
(orbitalMixer: https://acdc.alcf.anl.gov/mdf/detail/orbital_mixer_v1.1/)

### 1. Installation & Setup
build pyscf from scratch:
- sudo apt-get install libblas-dev liblapack-dev

!! note pyscf in this folder is not used, one in site-packages is used

<br>
<br>

### 2. Performing CASSCF calculations in PySCF
##### 2.1 Run calculations
`python data/run_casscf_calculations.py --geometry_folder geometries/geom_scan_200/ --output_folder pyscf/geom_scan_200_sto_6g/ --basis sto_6g --no-parallel`
##### 2.2 Analyze results
`python data/check_casscf_calculations.py --output_folder pyscf/geom_scan_200_sto_6g/`
##### 2.3 Save results in ASE DB
`python data/save_casscf_calculations_to_db.py --geometry_folder geometries/geom_scan_200/ --output_folder pyscf/geom_scan_200_sto_6g/`
##### 2.4 Generate split
`python data/generate_split.py --name geom_scan_200 --n 200 --train_split 0.9 --val_split 0.1 --test_split 0.0`

<br>
<br>

### 3. Training & evaluating models
`python train_model.py --db_name geom_scan_200_sto_6g.db --split_name geom_scan_200.npz --property F --model_name gs200_sto_6g_F`

##### 3.2 Look at predicted orbitals for a particular geometry
`python evaluation/write_orbital_guesses_to_molden.py  --geometry_path geometries/geom_scan_200/geometry_10.xyz --mo_model gs200_sto_6g_MO --F_model gs200_sto_6g_F --basis sto_6g `

##### 3.3 Evaluate orbital convergence on validation/test set
- `python evaluation/evaluate_orbital_guesses_convergence.py  --geometry_folder geometries/geom_scan_200/ --split_name geom_scan_200.npz --mo_model gs200_sto_6g_MO --F_model gs200_sto_6g_F --basis sto_6g`
- `python evaluation/evaluate_orbital_guesses_energies.py  --geometry_folder geometries/geom_scan_200/ --split_name geom_scan_200.npz --mo_model gs200_sto_6g_MO --F_model gs200_sto_6g_F --basis sto_6g`


necessary packages:
- schnetpack
- apsw
- pyscf
- sympy
- matscipy
- dirsync
- fasteners
- torch_ema
- hydra
- pytorch_lightning