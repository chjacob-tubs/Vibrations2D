#!/usr/bin/env python
# use: ./start_VCI.py
# start this script from harmonic calculation folder

import os
import sys # sys.exit()

initdir = os.getcwd()

# Here are the scripts
folderwithscripts = '/home/julia/2d-ir-spectroscopy/skripte'
resultsfolder = os.path.join(initdir,'results_VCI_exciton')

exciton = os.path.join(folderwithscripts,'3_calc_excitonmodel.py')
vciharm = os.path.join(folderwithscripts,'3_calctransmat_harm.py')
vcianharm = os.path.join(folderwithscripts,'3_calctransmat.py')


# Check if there is an folder for anharmonic calculations.
# Check if needed files are in anharm folder.
# If not: start get_potentials script to get needed files.
if os.path.exists(os.path.join(initdir,'anharm')):
    os.chdir(os.path.join(initdir,'anharm'))
    #print(os.listdir(os.getcwd()))
    
    resultsfiles = ['V1_g16.npy','Dm1_g16.npy','V2_g16.npy','Dm2_g16.npy']
    if all(elem in os.listdir(os.getcwd()) for elem in resultsfiles):
        print('Found all expected files:',resultsfiles)
    else:
        print('Starting get_potentials.py script:')
        os.system('python 2_get_potentials.py')
    os.chdir('..')

else:
    print('ERROR: No anharm folder found.')
    sys.exit()

# Make new results folder for the VCI calculation.
# Check if such folder already exists. 
# Start three VCI scripts:
#     - Exciton Model
#     - VCI Harmonic Model
#     - VCI Anharmonic Model
# Save output to new files. 
if os.path.exists(resultsfolder):
    print('ERROR: results folder already exists!')
else:
    print('Successfully added new folder',resultsfolder)
    os.makedirs(resultsfolder)

    print('Starting exciton model calculations.')
    os.system('python '+exciton+' > '+os.path.join(resultsfolder,'calc_exciton.out'))
    
    print('Starting VCI harmonic calculation.')
    os.system('python '+vciharm+' > '+os.path.join(resultsfolder,'calc_transmatharm.out'))
    
    print('Starting VCI anharmonic calculation.')
    os.system('python '+vcianharm+' > '+os.path.join(resultsfolder,'calc_transmat.out'))
    
    print('All done.')