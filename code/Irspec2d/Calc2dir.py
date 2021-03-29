import numpy as np

class basics():
    '''
    This class includes all basic functions that are needed for the calculation of 2D IR spectra.
    
    '''
    
    def __init__(self, freqmat, intmat):
        '''
        Add text. 
        
        '''
        self.intmat = intmat
        self.freqmat = freqmat
        self.nmodes = self.calc_nmodes()
        self.noscill = self.calc_num_oscill()
        
        if self.check_symmetry(self.intmat) == False:
            print('Intensity matrix is not (skew-)symmetrical. Please check!')
        if self.check_symmetry(self.freqmat) == False:
            print('Intensity matrix is not (skew-)symmetrical. Please check!')
        
        
    def calc_nmodes(self):
        '''
        Returns the number of modes.
        
        '''
        if len(self.intmat) == len(self.freqmat):
            return len(self.intmat)
    
    def calc_num_oscill(self):
        '''
        Calculates the number of oscillators n_oscill based on a 
        given number of modes n_modes. This is based on the assumption 
        that there are 
           n_modes = 1 + 2n_oscill + (n_oscill*(n_oscill-1))/2 
        modes. There is one ground state (0) plus n first excited states 
        plus n second excited states plus (n_oscill*(n_oscill-1))/2 combination 
        states. 

        '''
        return (-3. + np.sqrt(8.*self.nmodes +1.) ) /2. 
    
    def check_symmetry(self, a, tol=1e-5):
        '''
        Checks if a given matrix a is symmetric or skew-symmetric.
        Returns True/False.

        '''
        return np.all(np.abs(abs(a)-abs(a).T) < tol)
    



class calc2dir(basics):
    '''
    This class is supposed to evaluate 2D IR spectra in a simple approach.
    
    '''
    
    def __init__(self, freqmat, intmat, verbose_all=False):
        
        super().__init__(freqmat, intmat)
        self.freqs = self.freqmat[0]
        self.verbose_all = verbose_all
    
    def calc_excitation(self,verbose=False):
        '''
        Takes the energy levels and the intensity matrix in order to find 
        the excited state absorption processes that occur in an 2D IR
        experiment. 

        '''
        exc_x = [] # excitation coords
        exc_y = [] 
        exc_i = [] # intensity

        for i in range(len(self.intmat)):
            if self.intmat[0][i] and i<=self.noscill:
                for j in range(len(self.intmat)):
                    if j>i:

                        y_coor = self.freqs[j]-self.freqs[i]
                        x_coor = self.freqs[i]-self.freqs[0]
                        exc_inten = self.intmat[i][j]

                        exc_y.append(y_coor)
                        exc_x.append(x_coor)
                        exc_i.append(exc_inten)

                        if self.verbose_all == True or verbose == True : print('Excitation from energy level',i,'to',j,'at (',x_coor,',',y_coor,') rcm and intensity: ',exc_inten)
        return (exc_x, exc_y, exc_i)
    
    def calc_stimulatedemission(self,verbose=False):
        '''
        Takes the energy levels and the intensity matrix in order to find
        the stimulated emission processes that occur in an 2D IR experiment.
        In order to match the experiment the stimulated emission can only 
        happen in transition to the ground state energy level!

        '''
        emi_x = [] # excitation coords
        emi_y = [] 
        emi_i = [] # intensity

        for i in range(len(self.intmat)):
            for j in range(len(self.intmat)):
                if j==0 and i>j and i<=self.noscill:

                    y_coor = self.freqs[i]-self.freqs[j]
                    x_coor = self.freqs[i]-self.freqs[j]
                    emi_inten = self.intmat[j][i]

                    emi_y.append(y_coor)
                    emi_x.append(x_coor)
                    emi_i.append(emi_inten)

                    if self.verbose_all == True or verbose == True : print('Stimulated emission from energy level',i,'to',j,'at (',x_coor,',',y_coor,') rcm and intensity: ',emi_inten)
        return (emi_x, emi_y, emi_i)

    def calc_bleaching(self,verbose=False):
        '''
        Takes the energy levels and the intensity matrix in order to find
        the bleaching processes that occur in an 2D IR experiment.

        '''

        ble_x = [] # excitation coords
        ble_y = [] 
        ble_i = [] # intensity

        for i in range(len(self.intmat)):
            if self.intmat[0][i] != 0 and i<=self.noscill:
                
                y_coor = self.freqs[i]-self.freqs[0]
                ble_inten = self.intmat[0][i]
                
                for j in range(len(self.intmat)):
                    if self.intmat[0][j] != 0 and j<=self.noscill:
                        x_coor = self.freqs[j]-self.freqs[0]
                        ble_x.append(x_coor)
                        ble_y.append(y_coor)
                        ble_i.append(ble_inten)
                        if self.verbose_all == True or verbose == True : print('Bleaching from energy level 0 to',i,'at (',x_coor,',',y_coor,') rcm and intensity: ',ble_inten)

        return (ble_x, ble_y, ble_i)
                      
    def calc_all_2d_process(self,verbose=False):
        '''
        Calculates all processes that can occur within a
        2D IR experiment from the energy levels and the
        intensity matrix. 

        '''
        return self.calc_excitation(verbose=verbose), self.calc_stimulatedemission(verbose=verbose), self.calc_bleaching(verbose=verbose)