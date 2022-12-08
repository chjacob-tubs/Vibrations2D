import numpy as np
from numpy import linalg as LA

from Irspec2d import *

# FREQUENCY DOMAIN FUNCTIONS

class freqdomain(Calc2dir_base):
    
    def __init__(self, freqmat, dipoles):
        
        super().__init__(freqmat, dipoles)
    
    def calc_excitation(self,intmat):
        '''
        Takes the energy levels and the intensity matrix in order to find 
        the excited state absorption processes that occur in an 2D IR
        experiment. 
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''
        exc_x = [] # excitation coords
        exc_y = [] 
        exc_i = [] # intensity

        for i in range(self.noscill+1):
                for j in range(len(intmat)):
                    if i>0 and j>i:

                        x_coor = self.freqmat[0][i]-self.freqmat[0][0]
                        y_coor = self.freqmat[0][j]-self.freqmat[0][i]
                        exc_inten = intmat[i][j]

                        exc_y.append(y_coor)
                        exc_x.append(x_coor)
                        exc_i.append(exc_inten)

                        print('Excitation from energy level',i,'to',j,'at (',np.around(x_coor,2),',',np.around(y_coor,2),') rcm and intensity: ',np.around(exc_inten,2))
                        
        return (exc_x, exc_y, exc_i)
    
    def calc_stimulatedemission(self,intmat):
        '''
        Takes the energy levels and the intensity matrix in order to find
        the stimulated emission processes that occur in an 2D IR experiment.
        In order to match the experiment the stimulated emission can only 
        happen in transition to the ground state energy level!
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''
        emi_x = [] # stimulated emission coords
        emi_y = [] 
        emi_i = [] # intensity

        for i in range(self.noscill+1):
            for j in range(len(intmat)):
                if j==0 and i>j:

                    x_coor = self.freqmat[0][i]-self.freqmat[0][j]
                    y_coor = self.freqmat[0][i]-self.freqmat[0][j]
                    emi_inten = intmat[i][j]

                    emi_y.append(y_coor)
                    emi_x.append(x_coor)
                    emi_i.append(emi_inten)

                    print('Stim. emission from energy level',i,'to',j,'at (',np.around(x_coor,2),',',np.around(y_coor,2),') rcm and intensity: ',np.around(emi_inten,2))
        return (emi_x, emi_y, emi_i)

    def calc_bleaching(self,intmat):
        '''
        Takes the energy levels and the intensity matrix in order to find
        the bleaching processes that occur in an 2D IR experiment.
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''

        ble_x = [] # bleaching coords
        ble_y = [] 
        ble_i = [] # intensity

        for i in range(self.noscill+1):
            for j in range(len(intmat)):
                if i==0 and j>0 and j<=self.noscill:
                    for k in range(1,self.noscill+1):

                        x_coor = self.freqmat[0][k]-self.freqmat[0][i]
                        y_coor = self.freqmat[0][j]-self.freqmat[0][i]
                        ble_inten = -intmat[i][j]

                        ble_x.append(x_coor)
                        ble_y.append(y_coor)
                        ble_i.append(ble_inten)

                        print('Bleaching from energy level 0 to',j,'at (',np.around(x_coor,2),',',np.around(y_coor,2),') rcm and intensity: ',np.around(ble_inten,2))

        return (ble_x, ble_y, ble_i)
                      
    def calc_all_2d_process(self):
        '''
        Calculates all processes that can occur within a
        2D IR experiment from the energy levels and the
        intensity matrix. 
        
        @return: x- and y-coordinates and intensities of all processes
        @rtype: three tuples of lists

        '''
        intmat = self.calc_trans2int()
        
        exc_x = [] # excitation coords
        exc_y = [] 
        exc_i = [] # intensity
        
        emi_x = [] # stimulated emission coords
        emi_y = [] 
        emi_i = [] # intensity
        
        ble_x = [] # bleaching coords
        ble_y = [] 
        ble_i = [] # intensity
        
        for i in range(self.noscill+1):
            for j in range(len(intmat)):

                # look for excitation
                if i>0 and j>i:
                    e_x = self.freqmat[0][i]-self.freqmat[0][0]
                    e_y = self.freqmat[0][j]-self.freqmat[0][i]
                    e_i = intmat[i][j]
                    
                    exc_x.append(e_x)
                    exc_y.append(e_y)
                    exc_i.append(e_i)
                    
                    # print('excitation',i,j,'  ( '+str(e_x)+' | '+str(e_y)+' )',' :',e_i)

                # look for stimulated emission
                if j==0 and i>j:
                    s_x = self.freqmat[0][i]-self.freqmat[0][j]
                    s_y = self.freqmat[0][i]-self.freqmat[0][j]
                    s_i = intmat[i][j]
                    
                    emi_x.append(s_x)
                    emi_y.append(s_y)
                    emi_i.append(s_i)
                    
                    # print('emission',i,j,'  ( '+str(s_x)+' | '+str(s_y)+' )',' :',s_i)

                # look for bleaching
                if i==0 and j>0 and j<=self.noscill:
                    for k in range(1,self.noscill+1):
                        b_x = self.freqmat[0][k]-self.freqmat[0][i]
                        b_y = self.freqmat[0][j]-self.freqmat[0][i]
                        b_i = -intmat[i][j]
                    
                        ble_x.append(b_x)
                        ble_y.append(b_y)
                        ble_i.append(b_i)
                    
                        # print('bleaching',i,j,'  ( '+str(b_x)+' | '+str(b_y)+' )',' :',b_i)

        exc = (exc_x,exc_y,exc_i)
        ste = (emi_x,emi_y,emi_i)
        ble = (ble_x,ble_y,ble_i)
        
        return exc, ste, ble