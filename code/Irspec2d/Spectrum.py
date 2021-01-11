import numpy as np
import matplotlib.pyplot as plt
from Irspec2d import Calc2dir

class spectrum():
    '''
    This class generates data for plotting or plotts the spectra.

    '''
    def __init__(self, freqmat, intmat):
        '''
        Add text.
        
        '''
        self.intmat = intmat
        self.freqmat = freqmat
        self.freqs = self.freqmat[0]
        self.ints = self.intmat[0]
        self.noscill = Calc2dir.calc2dir(self.freqmat,self.intmat).calc_num_oscill()
        
    def calc_gauss(self,ints,x,freqs):
        '''
        Returns a value for a gaussian function.

        '''
        return ints*np.exp(-0.1*(x-freqs)**2)
    
    def set_xmin(self):
        '''
        Returns the first energy level. This is not the ground state (would be zero). 
        
        '''
        return self.freqs[1]
    
    def set_xmax(self):
        '''
        Returns the highest first excited state energy level. 
        
        '''
        return self.freqs[int(self.noscill)]
    
    def calc_gauss_spectrum1d(self,dim,margin=50):
        '''
        Sums over a len(freq) gaussian functions.
        Returns x and y values.

        '''
        xmin=self.set_xmin()-margin
        xmax=self.set_xmax()+margin
        
        x = np.linspace(xmin,xmax,dim)
        y = np.zeros(dim)

        for j in range(len(self.freqs)):
            ys = []
            for i in range(dim):
                ys.append(self.calc_gauss(self.ints[j],x[i],self.freqs[j]))
                y[i] += ys[i]
        return x,y
    
    def calc_lorentz(self,ints,x,x0,y,y0):
        '''
        Returns a value for a 2D lorentzian function.

        '''
        return (ints)/(2*np.pi) * 1/( ((x-x0)**2 + (y-y0)**2 + 100 ) )

    def calc_sum_lorentz(self,dim,x_val,y_val,intensity,xx,yy):
        '''
        Takes the number of data points dim, peaks (x_val and y_val)
        the intensities and the meshgrid xx,yy and calculates the 2D
        lorentian function on the given grid. 

        '''
        z_val = np.zeros((dim,dim))
        for i in range(len(x_val)):
            z_val += self.calc_lorentz(intensity[i],xx,x_val[i],yy,y_val[i])
        return z_val
    
    def calc_lorentz_spectrum2d(self,lorentzmin,lorentzmax,dim):
        '''
        Adds up the Lorentz functions for all processes and returns
        the values on an xx,yy grid. 
        
        '''
        
        exc_x, exc_y, exc_i, emi_x, emi_y, emi_i, ble_x, ble_y, ble_i = Calc2dir.calc2dir(self.freqmat,self.intmat).calc_all_2d_process()
        
        x = np.linspace(lorentzmin, lorentzmax, dim)
        y = np.linspace(lorentzmin, lorentzmax, dim)
        xx, yy = np.meshgrid(x, y)

        exc_z = self.calc_sum_lorentz(dim,exc_x,exc_y,exc_i,xx,yy)
        ble_z = self.calc_sum_lorentz(dim,ble_x,ble_y,ble_i,xx,yy)
        emi_z = self.calc_sum_lorentz(dim,emi_x,emi_y,emi_i,xx,yy)

        z = np.zeros((dim,dim))
        z += exc_z - ble_z - emi_z
        return x,y,z
    
    def find_minmax(self,z):
        '''
        Searches for the highest absolute value.
        (Needed for plotting: sets the vmax/vmin values when 
         using the contourf plot)

        '''
        print(np.asarray(z).min(),np.asarray(z).max())
        if abs(np.asarray(z).max()) > abs(np.asarray(z).min()):
            z_val = abs(np.asarray(z).max())
            print('chose max: ', z_val)
        if abs(np.asarray(z).max()) < abs(np.asarray(z).min()):
            z_val = abs(np.asarray(z).min())
            print('chose min ', z_val)
        return z_val
    
    def plot_2dspectrum_dots(self,xmin,xmax,ble_x,ble_y,ble_i,exc_x,exc_y,exc_i,emi_x,emi_y,emi_i):
        '''
        Example plot of the positions for the different processes. 
        
        '''
        fig = plt.figure(figsize=(15,15))
        plt.plot([0,xmax], [0,xmax], ls="--", c=".5")
        plt.plot(ble_x,ble_y, "x", color='gold', label='bleaching', markersize=20)
        plt.plot(exc_x,exc_y, "bx", label='excitation', markersize=10)
        plt.plot(emi_x,emi_y, "gx", label='stim. emiss.', markersize=10)
        plt.scatter(exc_x,exc_y,s=exc_i, color='b', alpha=0.5)
        plt.scatter(ble_x,ble_y,s=ble_i, color='gold', alpha=0.5)
        plt.scatter(emi_x,emi_y,s=emi_i, color='g', alpha=0.5)
        plt.xlim(xmin,xmax)
        plt.ylim(xmin,xmax)
        plt.grid(True)
        plt.ylabel('Probe',fontsize=34)
        plt.xlabel('Pump',fontsize=34)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.legend(fontsize=20,loc='lower right')
        return