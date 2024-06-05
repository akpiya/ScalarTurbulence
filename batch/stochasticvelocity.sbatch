#!/bin/bash

#SBATCH --job-name=stochasticfield
#SBATCH --output=log/stochasticfield.out
#SBATCH --account=pi-vitelli
#SBATCH --time=00:30:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12

