import pytest
import numpy as np
import math
from numpy import linalg as LA

import Irspec2d.Calc2dir as Calc2dir
    
class Testspectra():
    
    def test_test(self):
        assert 5 == 5 
        
    def test_set_line_spacing(self):
        '''
        Use this for matplotlib.pyplot contour plots in order to set the 
        number of lines. 
        
        Usage: set_line_spacing(maximum,number)
        Example: plt.contour(x,y,z,set_line_spacing(abs(z.max()),20))

        '''
        output1 = Calc2dir.spectra.set_line_spacing(3,5)
        expected_out1 = [-3. , -2.4, -1.8, -1.2, -0.6,  0.6,  1.2,  1.8,  2.4,  3. ]
        np.testing.assert_almost_equal(output1, expected_out1, decimal=4)
        
        output2 = Calc2dir.spectra.set_line_spacing(10,10)
        expected_out2 = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        np.testing.assert_almost_equal(output2, expected_out2, decimal=4)
    
    def test_gauss_func(self):
        '''
        Computes a single value at position x for a 
        1D gaussian type function.
        
        Usage: gauss_func(intensity,x,x0,halfwidth)
        
        '''
        output1 = Calc2dir.spectra.gauss_func(1,5,0,10)
        expected_out1 = 0.004412712003053031
        np.testing.assert_almost_equal(output1, expected_out1, decimal=10)
        
        output2 = Calc2dir.spectra.gauss_func(5,7,2,50)
        expected_out2 = 0.001716818464510043
        np.testing.assert_almost_equal(output2, expected_out2, decimal=10)
        
    def test_gauss2d_func(self):
        '''
        Computes a single value at position x for a 
        1D gaussian type function.
        
        Usage: gauss2d_func(intensity,x,x0,y,y0,halfwidth)
        
        '''
        output1 = Calc2dir.spectra.gauss2d_func(1,5,0,5,0,10)
        expected_out1 = 0.0022063560015265155
        np.testing.assert_almost_equal(output1, expected_out1, decimal=10)
        
        output2 = Calc2dir.spectra.gauss2d_func(5,7,2,7,2,50)
        expected_out2 = 0.0016698719733144564
        np.testing.assert_almost_equal(output2, expected_out2, decimal=10)
    
    def test_lorentz_func(self):
        '''
        Computes a single value at position x for a 
        1D lorentzian type function.
        
        Usage: lorentz_func(intensity,x,x0,halfwidth)
        
        '''
        output1 = Calc2dir.spectra.lorentz_func(1,5,0,10)
        expected_out1 = 0.03183098861837907
        np.testing.assert_almost_equal(output1, expected_out1, decimal=10)
        
        output2 = Calc2dir.spectra.lorentz_func(5,7,2,50)
        expected_out2 = 0.06121343965072898
        np.testing.assert_almost_equal(output2, expected_out2, decimal=10)
        
    def lorentz2d_func(self):
        '''
        Computes a single value at grid x,y for a 
        2D lorentzian type function.
        
        Usage: lorentz2d_func(intensity,x,x0,y,y0,halfwidth)
        
        '''
        output1 = Calc2dir.spectra.lorentz2d_func(1,5,0,5,0,10)
        expected_out1 = 0.02122065907891938
        np.testing.assert_almost_equal(output1, expected_out1, decimal=10)
        
        output2 = Calc2dir.spectra.lorentz2d_func(5,7,2,7,2,50)
        expected_out2 = 0.02122065907891938
        np.testing.assert_almost_equal(output2, expected_out2, decimal=10)
    
    
    def test_get_norm_1d_spectrum(self):
        '''
        Sums up all gauss/lorentz functions for each peak and sets the highest value to one.
        
        Usage: get_norm_1d_spectrum(xmin,xmax,freqs,ints,steps=5000,halfwidth=5,ftype='gauss')
        
        '''
        xmin = 0 
        xmax = 100
        
        freqs = np.linspace(xmin,xmax,10)
        ints = np.linspace(xmin,xmax,10)+5
        
        x_gauss = [0, 2.04081633, 4.08163265, 6.12244898, 8.16326531, 10.20408163, 12.24489796, 14.28571429, 16.32653061, 18.36734694, 20.40816327, 22.44897959, 24.48979592, 26.53061224, 28.57142857, 30.6122449 , 32.65306122, 34.69387755, 36.73469388, 38.7755102 , 40.81632653, 42.85714286, 44.89795918, 46.93877551, 48.97959184, 51.02040816, 53.06122449, 55.10204082, 57.14285714, 59.18367347, 61.2244898 , 63.26530612, 65.30612245, 67.34693878, 69.3877551 , 71.42857143, 73.46938776, 75.51020408, 77.55102041, 79.59183673, 81.63265306, 83.67346939, 85.71428571, 87.75510204, 89.79591837, 91.83673469, 93.87755102, 95.91836735, 97.95918367, 100]
        y_gauss = [0.047619173077411286, 0.030020590415917585, 0.008144946368995114, 0.010456481847797778, 0.05856140746396076, 0.14005930586896556, 0.13305610644604457, 0.050419282902728006, 0.013003094785508599, 0.050335260513379304, 0.1799940195997473, 0.25778558687650543, 0.14664278304669065, 0.035244414846434466, 0.032492148118181796, 0.16071054335148258, 0.3468163800397442, 0.29733638710635685, 0.10183995896243049, 0.027010580522910006, 0.11010971769946082, 0.3561183439670827, 0.46028137660119817, 0.23634299321556382, 0.05288269717558807, 0.06282091848080085, 0.28939131019935194, 0.563714682681903, 0.4361550069670883, 0.13532645796884749, 0.04189760351020143, 0.18998009754318088, 0.5558817752340665, 0.6483975254635579, 0.30058242058480705, 0.06421950495883742, 0.10464765021796511, 0.44584062956573356, 0.7838780550944255, 0.5473581196259256, 0.15430484726376173, 0.06047755041208205, 0.2931543384839422, 0.7753843816228342, 0.8162145776421201, 0.34171749699977505, 0.0722446074440935, 0.16133886622568525, 0.6301786169338374, 1.0]
        expected_out_gauss = (x_gauss,y_gauss)
        output_gauss = Calc2dir.spectra.get_norm_1d_spectrum(xmin,xmax,freqs,ints,steps=50,halfwidth=5,ftype='gauss')
        
        np.testing.assert_almost_equal(output_gauss, expected_out_gauss, decimal=7)
        
        x_lorentz = [0, 2.04081633, 4.08163265, 6.12244898, 8.16326531, 10.20408163, 12.24489796, 14.28571429, 16.32653061, 18.36734694, 20.40816327, 22.44897959, 24.48979592, 26.53061224, 28.57142857, 30.6122449 , 32.65306122, 34.69387755, 36.73469388, 38.7755102 , 40.81632653, 42.85714286, 44.89795918, 46.93877551, 48.97959184, 51.02040816, 53.06122449, 55.10204082, 57.14285714, 59.18367347, 61.2244898 , 63.26530612, 65.30612245, 67.34693878, 69.3877551 , 71.42857143, 73.46938776, 75.51020408, 77.55102041, 79.59183673, 81.63265306, 83.67346939, 85.71428571, 87.75510204, 89.79591837, 91.83673469, 93.87755102, 95.91836735, 97.95918367, 100.]
        y_lorentz = [0.06228356039393276, 0.0488464604565308, 0.04177851667241133, 0.050803924398031254, 0.08248259185200384, 0.15237767689980522, 0.14958242190737656, 0.09399703307793744, 0.08293999953468087, 0.10936373745726007, 0.19581394804958216, 0.28147776942246944, 0.18149952224496094, 0.1258306138959152, 0.13310018320286057, 0.20754368891248534, 0.37019763573974745, 0.32155189490435593, 0.1916543789442308, 0.16274255711428154, 0.21156630261350332, 0.3786307006042548, 0.49480486167315957, 0.2994014150686517, 0.20731784311833923, 0.2206309053159919, 0.34996581057613796, 0.6012863481200488, 0.4703128960174491, 0.2795591222887323, 0.24183331614061485, 0.3226041159024232, 0.5826368354745055, 0.6879100618147889, 0.39938790573309224, 0.2823119800512739, 0.309594306601867, 0.5069382924491993, 0.8315485023303244, 0.5907656721726464, 0.3527206242608812, 0.3135877146455436, 0.4354800698068898, 0.7959351408896913, 0.842201274069423, 0.467598587051229, 0.33138938501226467, 0.37456442100675363, 0.6445191149903975, 1.0]
        expected_out_lorentz = (x_lorentz,y_lorentz)
        output_lorentz = Calc2dir.spectra.get_norm_1d_spectrum(xmin,xmax,freqs,ints,steps=50,halfwidth=5,ftype='lorentz')
        
        np.testing.assert_almost_equal(output_lorentz, expected_out_lorentz, decimal=7)


    def test_get_2d_spectrum(self):
        '''
        Automatically fits 2d lorentz or gaussian function onto given
        2d coordiantes for the three processes. 
        
        Usage: get_2d_spectrum(xmin,xmax,exc,ble,emi,steps=2000,halfwidth=15,ftype='gauss')
        
        '''
        # These values are completly "random".
        val = np.linspace(0,100,10)
        proc = (val,val,val+5)
        
        # Gauss type function
        exp_out_x_gauss = [0.0, 25.0, 50.0, 75.0, 100.0]
        exp_out_y_gauss = [0.0, 25.0, 50.0, 75.0, 100.0]
        exp_out_z_gauss = [[0.5295254403663637, 4.177219182249698e-31, 2.170139872992363e-67, 1.2478299491939515e-137, 1.200937912538466e-267], [4.177219182249698e-31, 3.4872803232747197e-07, 3.2222087341792506e-17, 4.983310565337642e-87, 2.1084023279485208e-137], [2.170139872992363e-67, 3.2222087341792506e-17, 8.007871333798731e-27, 5.4444216543030255e-17, 3.666788061262749e-67], [1.2478299491939515e-137, 4.983310565337642e-87, 5.4444216543030255e-17, 5.892301235877681e-07, 3.2025347063916625e-30], [1.200937912538466e-267, 2.1084023279485208e-137, 3.666788061262749e-67, 3.2025347063916625e-30, 4.059695042808788]]
        out_x_gauss, out_y_gauss, out_z_gauss = Calc2dir.spectra.get_2d_spectrum(0,100,proc,proc,proc,steps=5,halfwidth=5,ftype='gauss')
        
        np.testing.assert_almost_equal(out_x_gauss, exp_out_x_gauss, decimal=7)
        np.testing.assert_almost_equal(out_y_gauss, exp_out_y_gauss, decimal=7)
        np.testing.assert_almost_equal(out_z_gauss, exp_out_z_gauss, decimal=7)
        
        # Lorentz type function
        exp_out_x_lorentz = [0.0, 25.0, 50.0, 75.0, 100.0]
        exp_out_y_lorentz = [0.0, 25.0, 50.0, 75.0, 100.0]
        exp_out_z_lorentz = [[1.9518420947039785, 0.08238223004327774, 0.07018237442543628, 0.05757203487028467, 0.05002358239311952], [0.08238223004327774, 0.3185934324950019, 0.1763865358568427, 0.09742975322955487, 0.07594655832861882], [0.07018237442543628, 0.1763865358568427, 0.22974184779329007, 0.26430483344983885, 0.12938401981993064], [0.05757203487028467, 0.09742975322955487, 0.26430483344983885, 0.5735994060430677, 0.26160012955996975], [0.05002358239311952, 0.07594655832861882, 0.12938401981993064, 0.26160012955996975, 14.766835459903769]]
        out_x_lorentz, out_y_lorentz, out_z_lorentz = Calc2dir.spectra.get_2d_spectrum(0,100,proc,proc,proc,steps=5,halfwidth=5,ftype='lorentz')
        
        np.testing.assert_almost_equal(out_x_lorentz, exp_out_x_lorentz, decimal=7)
        np.testing.assert_almost_equal(out_y_lorentz, exp_out_y_lorentz, decimal=7)
        np.testing.assert_almost_equal(out_z_lorentz, exp_out_z_lorentz, decimal=7)
    
    def test_norm_2d_spectrum(self):
        '''
        Sets the highest value of a matrix to one.
        
        Usage: norm_2d_spectrum(z, max_z)
        '''
        input_matrix1 = [[5,10],[7,2]]
        input_max1 = 10
        expected_output1 = [[0.5,1],[0.7,0.2]]
        output1 = Calc2dir.spectra.norm_2d_spectrum(input_matrix1, input_max1)
        print(output1)
        assert output1 == expected_output1

        input_matrix2 = [[1,2,3],[4,5,6],[7,8,10]]
        input_max2 = 10
        expected_output2 = [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,1]]
        output2 = Calc2dir.spectra.norm_2d_spectrum(input_matrix2, input_max2)
        assert output2 == expected_output2
        