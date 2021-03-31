import numpy as np
import matplotlib.pyplot as plt

class spectrum():
    '''
    This class generates data for plotting or plotts the spectra.

    '''
    
    def __init__(self,verbose_all=False):
        self.verbose_all = verbose_all
    
    def find_xmin(self,freqs):
        '''
        Returns the first energy level. This is not the ground state (would be zero). 
        
        '''
        if len(freqs.shape)==1:
            if freqs[0]==0:
                return freqs[1]
            else:
                return freqs[0]
        if len(freqs.shape)==2:
            if freqs[0][0]==0:
                return freqs[0][1]
            else:
                return freqs[0][0]
        else:
            print('Could not compute the minimum value. Check input.')
    
    def find_xmax(self,freqs,n):
        '''
        Returns the highest first excited state energy level. 
        
        '''
        if len(freqs.shape)==1:
            return freqs[n]
        if len(freqs.shape)==2:
            return freqs[0][n]
        else:
            print('Could not compute the maximum value. Check input.')
        
    def calc_gauss(self,intensity,x,frequency):
        '''
        Returns a value for a gaussian function.

        '''
        return intensity*np.exp(-0.1*(x-frequency)**2)
    
    def calc_gauss_spectrum1d(self,dim,ints,freqs,margin=50):
        '''
        Sums over a len(freq) gaussian functions.
        Returns x and y values.

        '''
        xmin=self.find_xmin(freqs)-margin
        xmax=self.find_xmax(freqs)+margin
        
        x = np.linspace(xmin,xmax,dim)
        y = np.zeros(dim)

        for j in range(len(freqs)):
            ys = []
            for i in range(dim):
                ys.append(self.calc_gauss(ints[j],x[i],freqs[j]))
                y[i] += ys[i]
        return x,y
    
    def plot_2dspectrum_dots(self,xmin,xmax,exc,emi,ble):
        '''
        Example plot of the positions for the different processes. 
        
        '''
        exc_x,exc_y,exc_i = exc
        emi_x,emi_y,emi_i = emi
        ble_x,ble_y,ble_i = ble
        
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
    
    def find_minmax(self,z,verbose=False):
        '''
        Searches for the highest absolute value.
        (Needed for plotting: sets the vmax/vmin values when 
         using the contourf plot)

        '''
        if self.verbose_all == True or verbose == True : print(np.asarray(z).min(),np.asarray(z).max())
        
        if abs(np.asarray(z).max()) > abs(np.asarray(z).min()):
            z_val = abs(np.asarray(z).max())
            if self.verbose_all == True or verbose == True : print('chose max: ', z_val)
        
        if abs(np.asarray(z).max()) < abs(np.asarray(z).min()):
            z_val = abs(np.asarray(z).min())
            if self.verbose_all == True or verbose == True : print('chose min: ', z_val)
        
        return z_val
    
    def calc_lorentz_spectrum2d(self,exc,emi,ble,lorentzmin,lorentzmax,dim):
        '''
        Adds up the Lorentz functions for all processes and returns
        the values on an xx,yy grid. 
        
        '''
        exc_x,exc_y,exc_i = exc
        emi_x,emi_y,emi_i = emi
        ble_x,ble_y,ble_i = ble
        
        
        x = np.linspace(lorentzmin, lorentzmax, dim)
        y = np.linspace(lorentzmin, lorentzmax, dim)
        xx, yy = np.meshgrid(x, y)

        exc_z = self.calc_sum_lorentz(dim,exc_x,exc_y,exc_i,xx,yy)
        ble_z = self.calc_sum_lorentz(dim,ble_x,ble_y,ble_i,xx,yy)
        emi_z = self.calc_sum_lorentz(dim,emi_x,emi_y,emi_i,xx,yy)

        z = np.zeros((dim,dim))
        z += exc_z - ble_z - emi_z
        return x,y,z