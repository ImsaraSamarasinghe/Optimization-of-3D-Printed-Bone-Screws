#!/bin/bash

# change directory to PNG_shear
cd PNG_shear

# delete all png files
rm *.png

# change to constraint_history
cd ..
cd constraint_history

# delete all png files
rm *.png

# change to newIDPresults
cd ..
cd newIDPresults

# delete all .png and .pvd files
rm *.png
rm *.pvd

# change to PNG_rho
cd ..
cd PNG_rho

# delete all pvd and vtu
rm *.pvd
rm *.vtu

# move to forces file
cd ..
cd forces

# delete all png in forces file
rm *.png

# come back to main dir
cd ..

# run the code
export OMP_NUM_THREADS=1
python IDP.py