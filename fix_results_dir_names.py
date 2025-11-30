"""Quick script to rename the results dir with missing "_" before "ped" and "lr"."""
from pathlib import Path

"""Adjust with the name of the results directory containing subdirectories missing 
underscore"""
DIR_PATH = Path("Results/20251130_033304")

# Substrings missing the underscore
SUBS_MISSING_UNDERSCORE = ["ped_", "lr_"]

# For each results sub dir
for dir_name in DIR_PATH.iterdir():
    
    # If the subdir has missing underscore, rename it correctly
    if dir_name.is_dir():
        for sub in SUBS_MISSING_UNDERSCORE:
            new_name = dir_name.name.replace(sub, "_" + sub, 1)
            if new_name != dir_name.name: 
                dir_name.rename(dir_name.with_name(new_name))