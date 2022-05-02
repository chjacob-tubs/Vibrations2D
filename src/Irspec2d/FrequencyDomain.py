import numpy as np
from numpy import linalg as LA

from Calc2dir import *

# FREQUENCY DOMAIN FUNCTIONS

class freqdomain(Calc2dir_base):
    
    def __init__(self, freqmat, dipoles):
        
        super().__init__(freqmat, dipoles)
    
    def calc_excitation(self,intmat):
        '''
        Takes the energy levels and the intensity matrix in order to find 
        the excited state absorption processes that occur in an 2D IR
        experiment. 
        
        @param noscill: number of oscillators
        @type noscill: integer
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''
        exc_x = [] # excitation coords
        exc_y = [] 
        exc_i = [] # intensity

        for i in range(len(intmat)):
            if intmat[0][i] and i<=self.noscill:
                for j in range(len(intmat)):
                    if j>i:

                        y_coor = self.freqmat[0][j]-self.freqmat[0][i]
                        x_coor = self.freqmat[0][i]-self.freqmat[0][0]
                        exc_inten = intmat[i][j]

                        exc_y.append(y_coor)
                        exc_x.append(x_coor)
                        exc_i.append(exc_inten)

                        # print('Excitation from energy level',i,'to',j,'at (',x_coor,',',y_coor,') rcm and intensity: ',exc_inten)
                        
        return (exc_x, exc_y, exc_i)
    
    def calc_stimulatedemission(self,intmat):
        '''
        Takes the energy levels and the intensity matrix in order to find
        the stimulated emission processes that occur in an 2D IR experiment.
        In order to match the experiment the stimulated emission can only 
        happen in transition to the ground state energy level!
        
        @param noscill: number of oscillators
        @type noscill: integer
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''
        emi_x = [] # stimulated emission coords
        emi_y = [] 
        emi_i = [] # intensity

        for i in range(len(intmat)):
            for j in range(len(intmat)):
                if j==0 and i>j and i<=self.noscill:

                    y_coor = self.freqmat[0][i]-self.freqmat[0][j]
                    x_coor = self.freqmat[0][i]-self.freqmat[0][j]
                    emi_inten = intmat[j][i]

                    emi_y.append(y_coor)
                    emi_x.append(x_coor)
                    emi_i.append(emi_inten)

                    # print('Stimulated emission from energy level',i,'to',j,'at (',x_coor,',',y_coor,') rcm and intensity: ',emi_inten)
        return (emi_x, emi_y, emi_i)

    def calc_bleaching(self,intmat):
        '''
        Takes the energy levels and the intensity matrix in order to find
        the bleaching processes that occur in an 2D IR experiment.
        
        @param noscill: number of oscillators
        @type noscill: integer
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''

        ble_x = [] # excitation coords
        ble_y = [] 
        ble_i = [] # intensity

        for i in range(len(intmat)):
            if intmat[0][i] != 0 and i<=self.noscill:
                
                y_coor = self.freqmat[0][i]-self.freqmat[0][0]
                ble_inten = intmat[0][i]
                
                for j in range(len(intmat)):
                    if intmat[0][j] != 0 and j<=self.noscill:
                        x_coor = self.freqmat[0][j]-self.freqmat[0][0]
                        ble_x.append(x_coor)
                        ble_y.append(y_coor)
                        ble_i.append(ble_inten)
                        
                        # print('Bleaching from energy level 0 to',i,'at (',x_coor,',',y_coor,') rcm and intensity: ',ble_inten)

        return (ble_x, ble_y, ble_i)
                      
    def calc_all_2d_process(self):
        '''
        Calculates all processes that can occur within a
        2D IR experiment from the energy levels and the
        intensity matrix. 
        
        @return: x- and y-coordinates and intensities of all processes
        @rtype: three tuples of lists

        '''
        intmat = self.calc_trans2int(self.freqmat,self.dipoles)
        
        exc = self.calc_excitation(intmat)
        ste = self.calc_stimulatedemission(intmat)
        ble = self.calc_bleaching(intmat)
        
        return exc, ste, ble