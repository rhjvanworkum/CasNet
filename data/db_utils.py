import os
from typing import List
from ase.db import connect
from ase.io.extxyz import read_xyz
from tqdm import tqdm
import tempfile
import numpy as np
import re
import h5py

def parse_property_string(prop_str):
    """
    Generate valid property string for extended xyz files.
    (ref. https://libatoms.github.io/QUIP/io.html#extendedxyz)
    Args:
        prop_str (str): Valid property string, or appendix of property string
    Returns:
        valid property string
    """
    if prop_str.startswith("Properties="):
        return prop_str
    return "Properties=species:S:1:pos:R:3:" + prop_str


def xyz_to_extxyz(
    xyz_path,
    extxyz_path,
    atomic_properties="Properties=species:S:1:pos:R:3",
):
    """
    Convert a xyz-file to extxyz.
    Args:
        xyz_path (str): path to the xyz file
        extxyz_path (str): path to extxyz-file
        atomic_properties (str): property-string
        molecular_properties (list): molecular properties contained in the
            comment line
    """
    # ensure valid property string
    atomic_properties = parse_property_string(atomic_properties)
    new_file = open(extxyz_path, "w")
    with open(xyz_path, "r") as xyz_file:
        while True:
            first_line = xyz_file.readline()
            if first_line == "":
                break
            n_atoms = int(re.findall(r'\d+', first_line)[0])
            _ = xyz_file.readline().strip("/n").split()

            comment = ''
            new_file.writelines(str(n_atoms) + "\n")
            new_file.writelines(" ".join([atomic_properties, comment]) + "\n")
            for i in range(n_atoms):
                line = xyz_file.readline()
                new_file.writelines(line)
            break
    new_file.close()


def extxyz_to_db(extxyz_path, db_path, idx, molecular_properties=[]):
    r"""
    Convertes en extxyz-file to an ase database
    Args:
        extxyz_path (str): path to extxyz-file
        db_path(str): path to sqlite database
    """
    with connect(db_path, use_lock_file=False) as conn:
        with open(extxyz_path) as f:
            for at in tqdm(read_xyz(f, index=slice(None)), "creating ase db"):
                data = {}
                for property in molecular_properties:
                  data.update(property)
                conn.write(at, data=data, idx=idx)

def xyz_to_db(
    xyz_path,
    db_path,
    idx,
    atomic_properties="Properties=species:S:1:pos:R:3",
    molecular_properties=[],
):
    """
    Convertes a xyz-file to an ase database.
    Args:
        xyz_path (str): path to the xyz file
        db_path(str): path to sqlite database
        atomic_properties (str): property-string
        molecular_properties (list): molecular properties contained in the
            comment line
    """
    # build temp file in extended xyz format
    extxyz_path = os.path.join(tempfile.mkdtemp(), "temp.extxyz")
    xyz_to_extxyz(xyz_path, extxyz_path, atomic_properties)
    # build database from extended xyz
    extxyz_to_db(extxyz_path, db_path, idx, molecular_properties)