#!/usr/env/bin python

import numpy as np
from glob import glob

printoutput = None

def countelements(subset):
    counter = 0
    for i in subset:
        if isinstance(i,int):
            counter += 1
        else:
            for j in i:
                counter += 1
    return counter

#subsets = np.load('subsets.npy')
subsets = [[0],[1,2]]
nmodes = countelements(subsets)
#nmodes = 3
ngrid = 16
firstfile = nmodes * ngrid

with open('energies/E0.dat') as f:
    lines = f.readlines()

if printoutput: print(lines[0].split())
pot = float(lines[0].split()[0])
dmx = float(lines[0].split()[1])
dmy = float(lines[0].split()[2])
dmz = float(lines[0].split()[3])
if printoutput: print('Energy and dipole moment',pot,dmx,dmy,dmz)


E0 = pot
d0 = np.array([dmx,dmy,dmz])

v1 = np.zeros((nmodes,ngrid))
dm1 = np.zeros((nmodes,ngrid,3))

firstcount = 0
for fname in glob('./energies/v1*.dat'):

    f = open(fname,'r')
    fname = fname[:-4]
    mode = fname.split('_')[1]
    mode = int(mode)
    gp   = int(fname.split('_')[2])
    if printoutput: print('Mode : %i %i' %(mode,gp))
    lines = f.readlines()
    if printoutput: print(lines[0].split())
    pot = float(lines[0].split()[0])
    dmx = float(lines[0].split()[1])
    dmy = float(lines[0].split()[2])
    dmz = float(lines[0].split()[3])
    if printoutput: print('Energy and dipole moment',pot,dmx,dmy,dmz)
    v1[mode,gp] = pot - E0
    dm1[mode,gp,0] = dmx - d0[0]
    dm1[mode,gp,1] = dmy - d0[1]
    dm1[mode,gp,2] = dmz - d0[2]
    firstcount +=1

if printoutput: print('%i files found, should be %i' %(firstcount,firstfile))
if printoutput: print('Saving potential to file')
np.save('V1_g16.npy',v1)
np.save('Dm1_g16.npy',dm1)

v2 = np.zeros((nmodes,nmodes,ngrid,ngrid))
dm2 = np.zeros((nmodes,nmodes,ngrid,ngrid,3))

secondcount = 0
secondfile = nmodes*(nmodes-1)/2 * ngrid**2
for fname in glob('energies/v2*.dat'):

    f = open(fname,'r')
    fname = fname[:-4]
    model = fname.split('_')[1]
    model = int(model)
    moder = fname.split('_')[2]
    moder = int(moder)
    gpl   = int(fname.split('_')[3])
    gpr   = int(fname.split('_')[4])
    if printoutput: print('Modes : %i (%i) %i (%i) ' %(model,gpl,moder,gpr))
    lines = f.readlines()
    if printoutput: print(lines[0].split())
    pot = float(lines[0].split()[0])
    dmx = float(lines[0].split()[1])
    dmy = float(lines[0].split()[2])
    dmz = float(lines[0].split()[3])
    if printoutput: print('Energy and dipole moment',pot,dmx,dmy,dmz)
    v2[model,moder,gpl,gpr] = pot - v1[model,gpl] - v1[moder,gpr] - E0
    v2[moder,model,gpr,gpl] = pot - v1[model,gpl] - v1[moder,gpr] - E0
    dm2[model,moder,gpl,gpr,0] = dmx - dm1[model,gpl,0] - dm1[moder,gpr,0] - d0[0]
    dm2[model,moder,gpl,gpr,1] = dmy - dm1[model,gpl,1] - dm1[moder,gpr,1] - d0[1]
    dm2[model,moder,gpl,gpr,2] = dmz - dm1[model,gpl,2] - dm1[moder,gpr,2] - d0[2]
    dm2[moder,model,gpr,gpl,0] = dmx - dm1[model,gpl,0] - dm1[moder,gpr,0] - d0[0]
    dm2[moder,model,gpr,gpl,1] = dmy - dm1[model,gpl,1] - dm1[moder,gpr,1] - d0[1]
    dm2[moder,model,gpr,gpl,2] = dmz - dm1[model,gpl,2] - dm1[moder,gpr,2] - d0[2]
    f.close()
    secondcount += 1


if printoutput: print('%i files found, should be %i' %(secondcount,secondfile))
if printoutput: print('Saving potentials to file')

np.save('V2_g16.npy',v2)
np.save('Dm2_g16.npy',dm2)
    
print('*********  start get potentials script  *********')
if firstcount == firstfile:
    print('V1 potentials', firstcount, firstfile)
else:
    print('WARNING: Found',firstcount,'V1 potentials, but should be',firstfile,'!')
    print('Missing',abs(firstfile-firstcount),'files.')

if secondcount == secondfile:
    print('V2 potentials', secondcount, int(secondfile))
else:
    print('WARNING: Found',secondcount,'V2 potentials, but should be',secondfile,'!')
    print('Missing',abs(secondfile-secondcount),'files.')
print('*********   end get potentials script   *********')