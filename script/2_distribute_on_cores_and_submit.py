#!/usr/bin/env python
# ./distribute_on_cores.py <queue> <how_many_cores> <jobrunner>
#

from glob import glob
import sys
import shutil
import os
import time

queue = sys.argv[1]
nodes = int(sys.argv[2])
runner = sys.argv[3]
runscript = 'Calculate_Potentials.pyadf'

print('***** Starting Script *****')

initdir = os.getcwd()
endir = os.path.join(initdir,'energies')
if not os.path.exists(endir):
    print('  Made energies folder.')
    os.mkdir(endir)

print('  Cores: ', nodes)
print('  Runner:', runner)
print('  Energies stored to: ', endir)

print('*****')

# looks for already existing folders
folders = 0
for folder in os.listdir():
    if folder.isdigit() and int(folder) > folders:
        folders = int(folder)

if folders == 0 :
    
    for i in range(nodes) :
        
        dirname = str(i+1)
        os.mkdir(dirname)

        shutil.copy('restart', dirname)
        shutil.copy('snf.out', dirname)
        shutil.copy('coord', dirname)
        if os.path.isfile('subsets.npy'):
            shutil.copy('subsets.npy', dirname)

        os.chdir(dirname)

        f = open(os.path.join(initdir,runner))
        lines = f.readlines()
        for i,l in enumerate(lines):
            if 'actcore' in l:
                lines[i] = 'actcore = ' + dirname + '\n'
                break
        for i,l in enumerate(lines):
            if 'totcores' in l:
                lines[i] = 'totcores = ' + str(nodes) + '\n'
                break
        for i,l in enumerate(lines):
            if 'enpath = ' in l:
                lines[i] = 'enpath = \'' + endir + '\'\n'
                break
        f.close()
        f = open(runscript,'w')
        for l in lines:
            f.write(l)
        f.close()
        
        print('Starting',runscript,'script in folder',dirname+'.')

        os.system('tcsub -q %s -c 1 -a pyadf %s' %(queue,runscript))
        time.sleep(2)
        print()

        os.chdir('..')
        
elif folders == nodes :
    
    print('Found',nodes,'folders!')
    print('WARNING: restarting all',runscript,'scripts in preexisting folders! \n')
    
    for i in range(nodes) :
        dirname = str(i+1)
        os.chdir(dirname)
        
        checkfiles = ['coord','snf.out','restart',runscript]
        if all(elem in os.listdir(os.getcwd()) for elem in checkfiles):
            
            print('Restarting',runscript,'script in folder',dirname+'.')
            os.system('tcsub -q %s -c 1 -a pyadf %s' %(queue,runscript))
            time.sleep(2)
            print()
        
        os.chdir('..')
    
else:
    print('WARNING: Found',folders,'folders, but was to make',nodes,'folders. Please check.')
    
print('*****   End Script    *****')