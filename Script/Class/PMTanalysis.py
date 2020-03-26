import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
import scipy as sp
from scipy.signal import savgol_filter
import lmfit as lm
import configparser as cp
import datetime 

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
#mpl.use('Agg')

class PMTanalyzer():
    def __init__(self, path_to_config):
        '''
        Initialization for the class.  
        Args:
            1 anything yet!            
        Note:
            1. Open of the configuration file where fundamental parameters are defined like the calibration run ID and the light level
        '''
        self.config_file = path_to_config + '/configuration.ini'
        self.date = datetime.datetime.today().strftime('%Y%m%d')
        print(self.config_file)
        
        Config = cp.ConfigParser()
        Config.read(self.config_file)
 
        self.run_spe_topbulk = Config.get('runs_pars','run_spe_topbulk')
        self.run_spe_topring = Config.get('runs_pars','run_spe_topring')
        self.run_gain_step0  = Config.get('runs_pars', 'run_gain_step0')
        self.run_gain_step1  = Config.get('runs_pars', 'run_gain_step1')
        self.run_gain_step2  = Config.get('runs_pars', 'run_gain_step2')
        self.run_gain_step3  = Config.get('runs_pars', 'run_gain_step3')
        self.run_gain_step4  = Config.get('runs_pars', 'run_gain_step4')
        self.light_level = Config.get('runs_pars','light_level')
        
    def get_signal(self, data):
        '''
        Function which subtract the signal shift and invert it.
        '''
        data_ = np.copy(data)
        for r in data_:
            bsl = np.sum(r['data'][0:50])/(50.-0.)
            r['data'] = np.abs(r['data'] - bsl)
            
        return data_
    
    def get_baseline(self, data, n_samples=40):
        '''
        Function which estimates the baseline rms within the specified number of samples.

        Args:
            1. data: waveform from a PMT 
            2. n_samples: first n samples which are used for the rms is estimate.     
        Note:
            1. the RMS is only computed for samples <= zero.
        Output:
            1. array with PMT channel entry and signal baseline (in ADC usually) and error of each waveform.
        '''
        datatype = [('channel', np.int16), ('baseline', np.float32), ('baseline error', np.float32)]
        baseline = np.zeros(len(data), dtype=datatype)

        for i in range(len(data)):
            d_b = data['data'][i] + data['baseline'][i]%1
            mask = np.where(d_b[:n_samples]<=0)
            baseline['baseline'][i]       = np.mean(np.sqrt(d_b[mask]**2))
            baseline['baseline error'][i] = np.std(np.sqrt(d_b[mask]**2))/(len(d_b[mask])-1)

        return baseline

    def get_amplitude(self, data, window=None):
        '''
        Look for signal amplitude in a given window. 
    
        Args:
            1. data: waveform from a PMT
            2. window: define the sample range where look for the signal amplitude. If None, all the waveform it will be scan.
        Note:
            1. the input data is suppose to have 'channel' and 'data' entries, defined as are in strax.
            3. before use this function, the baseline has to be subtracted from the waveform and the signal inverted
        Output:
            1. array with PMT channel entry and signal amplitude (in ADC usually) of each waveform.
        '''
        datatype = [('channel', np.int16), ('amplitude', np.float32), ('sample of amplitude', np.float32)]
        amplitude = np.zeros(len(data), dtype = datatype)
        for i, wf in enumerate(data['data']):
            if window != None: 
                amplitude['channel'][i]   = data['channel'][i]
                amplitude['amplitude'][i] = np.max(wf[int(window[0]):int(window[1])])
                amplitude['sample of amplitude'][i] = np.argmax(wf[int(window[0]):int(window[1])])
            else:
                amplitude['channel'][i]   = data['channel'][i]
                amplitude['amplitude'][i] = np.max(wf)
                amplitude['sample of amplitude'][i] = np.argmax(wf)
        return amplitude
    
    def get_speinput(self, amplitude, channel=None):
        ''' 
        Function that finds the following values from an input SPE spectrum histogram:
        - Top of noise peak
        - Valley between noise and SPE peak
        - Top of SPE peak
        
        Args:
            1. amplitude: signal amplitude waveform from a PMT. The array should have channel variable.
            2. channel: PMT n° to analize. If it is None the amplitude array passed is suppose to be only for one PMT
        Note:
            1. this function is optimized for single photoelectron amplitude spectrum. I am not sure it would work for the area spectrum. On the other hand it should work even if there is not any PE signal.
        Output:
            1. fit_input: list with topnoise, valley, endvalley, spebump, endbin-1 bins index
            2. binning: bins to use if you want to visualize the output in a histogram
            3. check: this varible is meant to check if the spectrum has (1) or not (0) an SPE signal
        '''
        if channel != None:
            amplitude  = amplitude[amplitude['channel']==channel]['amplitude']
        bins_tmp      = np.max(amplitude)
        H_tmp, B_tmp  = np.histogram(amplitude, range=(0, bins_tmp), bins=int(bins_tmp/2.5))
        topnoise_tmp  = int(np.argmax(H_tmp))
        check         = 1
        
        try:
            rebin = B_tmp[int(np.where(H_tmp[topnoise_tmp+1:]<=0)[0][0])]
            H, B  = np.histogram(amplitude, range=(0, rebin), bins=int(rebin/2.5))
            binning = B
            nbins = len(B)

            topnoise = np.argmax(H)                           # idx of noise peak
            endbin   = int(rebin/2.5)                         # where bins start to be empty

        except:
            fit_input, binning, check = [topnoise_tmp, 0, 0, 0, int(bins_tmp/2.5)-1], B_tmp, 0
            rebin = 0
            
        if rebin != 0:
            valley, i = 0, topnoise+1
            while valley == 0 and i<endbin-topnoise:         # valley defined as the minumum between O PE and 1 PE
                if H[i] < H[(i+1)]:
                    valley = i
                else: i = i + 1
                               
            endvalley, i = 0, (topnoise+valley+1)
            while endvalley == 0 and i<endbin-topnoise-valley:
                if H[(i+1)]/H[i] > 1.1:
                    endvalley = i
                else: i = i + 1
            
            spebump = 0
            if valley != 0 or endvalley != 0:
                spebump  = int(np.argmax(H[endvalley+3:])) + endvalley+3  # SPE defined as second max after noise
            else: spebump = 0
            
            if endvalley >= spebump or H[endvalley]>=H[spebump]:
                check = 0
            if H[spebump] < 10:
                check = 0
            if valley == 0 or endvalley == 0:
                check = 0
            
            fit_input = [topnoise, valley, endvalley, spebump, endbin-1]
        return fit_input, binning, check
    
    def get_sperough(self, amplitude, channel=None):
        ''' 
        Function that computes a rough fit of SPE signal given the input from get_speinput function.
        
        Args:
            1. amplitude: signal amplitude waveform from a PMT. The array should have channel variable.
            2. channel: PMT n° to analize. If it is None the amplitude array passed is suppose to be only for one PMT
        Note:
            1. this function is optimized for single photoelectron amplitude spectrum. I am not sure it would work for the area spectrum. On the other hand it should work even if there is not any PE signal.
        Output:
            1. gauss: lmfit model if checkSPE is 1, otherwise is 0
            2. result_fit: lmfit fit result if checkSPE is 1, otherwise is 0
        '''
        inputSPE, binsSPE, checkSPE = self.get_speinput(amplitude, channel)
        gauss = lm.models.GaussianModel(prefix='g_')
                
        if channel != None:
            amplitude  = amplitude[amplitude['channel']==channel]['amplitude']
            
        H, B = np.histogram(amplitude,  bins=binsSPE)
        topnoise,valley, endvalley, spebump, endbin = inputSPE
        idx_1 = spebump - int((spebump-endvalley)/2)
        idx_2 = spebump + int((endbin-spebump)/2)
        
        if idx_1 < endvalley:
            idx_1 = endvalley
        if idx_2 > endbin:
            idx_2 = endbin
            
        if checkSPE == 1:      
            #gauss.set_param_hint('g_height', value=H[spebump], min=H[spebump]-30, max=H[spebump]+30)
            #gauss.set_param_hint('g_center', value=B[spebump], min=B[spebump]-30, max=B[spebump]+30)
            #gauss.set_param_hint('g_sigma', value=idx_2-idx_1, max=(idx_2-idx_1)+30)
            params = gauss.guess(H[idx_1:idx_2], x=B[idx_1:idx_2])
            result_fit = gauss.fit(H[idx_1:idx_2], x=B[idx_1:idx_2], 
                                   params=params,
                                   weights=1.0/np.sqrt(H[idx_1:idx_2]))
        else:
            #if channel != None:
            #    print('PMT: ', int(channel), '\t :(')
            #print('No SPE')
            gauss  = 0
            result_fit = 0
        return gauss, result_fit, B[idx_1:idx_2]
    
    def get_ledwindow(self, amplitude, channels):
        ''' 
        Function that find the LED window usign the SPE rough fit.
        
        Args:
            1. amplitude: signal amplitude waveform from a PMT. The array should have channel variable.
            2. channel: list of PMTs n° to analize.
        Note:
            
        Output:
            1. df_fit: data frame with fitted value for each PMTs
            2. df_led: data frame with time information, in sample, of led signal
            3. window: led signal window defined as integer of the mean of df_led +/- one standar deviation
        '''
        df_fit = pd.DataFrame({'channel': [ ], 'normalization': [ ], 'mean': [ ], 'sigma': [ ], 'chi_red': [ ]})
        df_led = pd.DataFrame({'channel': [ ], 'idx_LED': [ ]}) 
        for ch in channels:
            _, fit, __ = self.get_sperough(amplitude, ch)
            if fit != 0:
                popt   = fit.best_values
                df_fit = df_fit.append({'channel': int(ch),
                                       'normalization': popt['g_amplitude']/(np.sqrt(2*np.pi)*popt['g_sigma']),
                                       'mean': popt['g_center'],
                                       'sigma': popt['g_sigma'],
                                       'chi_red': fit.chisqr/fit.nfree},
                                       ignore_index=True)
            else:
                df_fit = df_fit.append({'channel': int(ch),
                                       'normalization': np.NaN,
                                       'mean': np.NaN,
                                       'sigma': np.NaN,
                                       'chi_red': np.NaN},
                                       ignore_index=True)

            mean    = df_fit[df_fit['channel']==ch]['mean'].iloc[0]
            sigma   = df_fit[df_fit['channel']==ch]['sigma'].iloc[0]
            PMT     = amplitude[amplitude['channel']==ch]
            mask    = np.where((PMT['amplitude'] < mean + sigma) & (PMT['amplitude'] > mean - sigma))

            idx_led = PMT['sample of amplitude'][mask]
            d_temp = pd.DataFrame({'channel': [ ], 'idx_led': [ ]})
            if len(idx_led)==0:
                d_temp['channel'] = ch
                d_temp['idx_led'] = np.NaN
            else:
                d_temp['idx_led'] = idx_led
                d_temp['channel'] = np.ones_like(idx_led) * ch
            df_led = df_led.append(d_temp, ignore_index=True)
            del d_temp, idx_led    

        median = np.nanmedian(df_led['idx_led'])
        std = np.nanstd(df_led['idx_led'])
        window = [int(median-0.5*std),int(median+std)]       
        
        length = window[1]-window[0]
        window_noise = [50, 50+length]
        
        Config = cp.ConfigParser()
        Config.read(self.config_file)
        Config.set("window_pars", "led_windows_left", str(window[0]))          # update
        Config.set("window_pars", "led_windows_right", str(window[1]))         # update
        Config.set("window_pars", "noise_windows_left", str(window_noise[0]))  # update
        Config.set("window_pars", "noise_windows_right", str(window_noise[1])) # update
        with open(self.config_file, 'w+') as configfile:
            Config.write(configfile)
            
        return df_fit, df_led, window
    
    def get_scalingspectrum(self, data_led, data_noise, n_channel_s = np.arange(0, 249, 1)):
        ''' 
        Function that subtract out the contribution of the noise to the amplitude spectrum. 
        Then it scale down the off-time amplitude spectrum such that the total counts below 
        the (3-7) ADC count threshold is the same as in the LED spectrum.
        
        Args:
            1. data_led: signal amplitude array from a PMT(s). The array should have channel variable.
            2. data_noise: noise amplitude array from a PMT(s). The array should have channel variable.
            3. n_channel_s: list of PMTs n° to analize.
        Note:
            1. the fraction of SPE signals with amplitude below a threshold of (3-7) ADC counts is assume to be small.
            2. the spectrum also contains contributions of 2 or more photoelectrons. From the scaling down factor 
               of the noise s, assuming a Poisson distribution of photoelectrons we estimate that the average number of 
               photoelectrons (occupancy) in the LED run was lambda = -ln(s) = 0.566.
            3. the fraction of events with 2 or more photoelectrons is then 1-exp(-lambda)(1+lambda) = 0.111. The contribution 
               of 2 or more photoelectrons leads to a slight over-estimate in the acceptances calculated.
        Output:
            1. SPE: array with histograms info.
        '''
        
        datatype = [('channel', np.int16),
                    ('spectrum led', object),   ('bins led', object),
                    ('spectrum noise', object), ('bins noise', object),
                    ('spectrum noise scaled 3 bin', object), ('occupancy 3 bin', np.float32),
                    ('spectrum noise scaled 4 bin', object), ('occupancy 4 bin', np.float32),
                    ('spectrum noise scaled 5 bin', object), ('occupancy 5 bin', np.float32),
                    ('spectrum noise scaled 6 bin', object), ('occupancy 6 bin', np.float32),
                    ('spectrum noise scaled 7 bin', object), ('occupancy 7 bin', np.float32)]

        SPE = np.zeros((len(n_channel_s)), dtype = datatype)

        for n_channel in n_channel_s:
            arr_led   = data_led[data_led['channel'] == n_channel]
            arr_noise = data_noise[data_noise['channel'] == n_channel]

            led, bins_led     = np.histogram(arr_led['amplitude_led'], bins=250, range=(0,500))
            noise, bins_noise = np.histogram(arr_noise['amplitude_noise'], bins=250, range=(0,500))

            SPE[n_channel]['channel']        = n_channel
            SPE[n_channel]['spectrum led']   = led
            SPE[n_channel]['bins led']       = bins_led
            SPE[n_channel]['spectrum noise'] = noise
            SPE[n_channel]['bins noise']     = bins_noise

            for i in range(3,8):
                ADC_correction = i
                scaling_coeff = led[:i].sum()/noise[:i].sum() 
                SPE[n_channel]['spectrum noise scaled '+str(i)+' bin'] = noise*scaling_coeff
                SPE[n_channel]['occupancy '+str(i)+' bin'] = -np.log(scaling_coeff)

        return SPE
    
    def get_speacceptance(self, data_spe, data_noise, n_channel_s = np.arange(0, 249, 1)):
            ''' 
            Function that compute SPE acceptance. 

            Args:
                1. data_led: signal amplitude array from a PMT(s). The array should have channel variable.
                2. data_noise: noise amplitude array from a PMT(s). The array should have channel variable.
                3. n_channel_s: list of PMTs n° to analize.
            Note:
                1. the acceptance as a function of amplitude (threshold) is defined as the fraction of 
                   noise-subtracted single photoelectron spectrum above that amplitude.
            Output:
                1. SPE_acceptance: array with histograms info.
            '''

            datatype = [('channel', np.int16),
                        ('Acceptance @ 15 ADC 3 bin', np.float32), ('Threshold for 0.9 acceptance 3 bin', np.float32),
                        ('SPE acceptance 3 bin', object), ('bins SPE acceptance 3 bin', object),
                        ('noise-subtracted spectrum 3 bin', object), ('error of noise-subtracted spectrum 3 bin', object),
                        ('Acceptance @ 15 ADC 4 bin', np.float32), ('Threshold for 0.9 acceptance 4 bin', np.float32),
                        ('SPE acceptance 4 bin', object), ('bins SPE acceptance 4 bin', object),
                        ('noise-subtracted spectrum 4 bin', object), ('error of noise-subtracted spectrum 4 bin', object),
                        ('Acceptance @ 15 ADC 5 bin', np.float32), ('Threshold for 0.9 acceptance 5 bin', np.float32),
                        ('SPE acceptance 5 bin', object), ('bins SPE acceptance 5 bin', object),
                        ('noise-subtracted spectrum 5 bin', object), ('error of noise-subtracted spectrum 5 bin', object),
                        ('Acceptance @ 15 ADC 6 bin', np.float32), ('Threshold for 0.9 acceptance 6 bin', np.float32),
                        ('SPE acceptance 6 bin', object), ('bins SPE acceptance 6 bin', object),
                        ('noise-subtracted spectrum 6 bin', object), ('error of noise-subtracted spectrum 6 bin', object),
                        ('Acceptance @ 15 ADC 7 bin', np.float32), ('Threshold for 0.9 acceptance 7 bin', np.float32),
                        ('SPE acceptance 7 bin', object), ('bins SPE acceptance 7 bin', object),
                        ('noise-subtracted spectrum 7 bin', object), ('error of noise-subtracted spectrum 7 bin', object),
                        ]
            SPE = self.get_scalingspectrum(data_spe, data_noise, n_channel_s)
            SPE_acceptance = np.zeros((len(n_channel_s)), dtype = datatype)
            j=0
            for n_channel in n_channel_s:
                arr = SPE[SPE['channel'] == n_channel]
                SPE_acceptance[j]['channel'] = j

                for i in range(3,8):
                    diff = np.absolute(arr['spectrum led'][0] - arr['spectrum noise scaled '+str(i)+' bin'][0])
                    sigma_diff = np.sqrt(arr['spectrum led'][0] + arr['spectrum noise scaled '+str(i)+' bin'][0])

                    res =  1. - np.cumsum(diff)/np.sum(diff)
                    x = arr['bins led'][0]
                    x_center = 0.5 * (x[1:] + x[:-1])
                    pos_15ADC = np.where(x_center<16)
                    pos_acc90 = np.where(res<0.9)

                    SPE_acceptance[j]['Acceptance @ 15 ADC '+str(i)+' bin'] = res[pos_15ADC[0][-1]]
                    SPE_acceptance[j]['Threshold for 0.9 acceptance '+str(i)+' bin'] = x_center[pos_acc90[0][0]]
                    SPE_acceptance[j]['SPE acceptance '+str(i)+' bin'] = res
                    SPE_acceptance[j]['bins SPE acceptance '+str(i)+' bin'] = x_center
                    SPE_acceptance[j]['noise-subtracted spectrum '+str(i)+' bin'] = diff
                    SPE_acceptance[j]['error of noise-subtracted spectrum '+str(i)+' bin'] = sigma_diff

                j=j+1

            return SPE_acceptance
        
    def get_moments(self, data, channels):
        ''' 
        Function that compute first and second moments (mean and variance) of data distribution.

        Args:
            1. data: PMT(s) array. The array should have channel variable.
            2. channels: list of PMTs n° to analize.

        Note:
            1. this function is used for gain calculation.

        Output:
            1. moments: array with mean and variance.
                
        '''  
        datatype = [('channel', np.int16), 
                    ('mean', np.float32), 
                    ('variance', np.float32)]
        moments =  np.zeros(len(channels), dtype = datatype)

        for i, ch in enumerate(channels):
            area = data[data['channel']==ch]['area']
            hist, bins  = np.histogram(area, range=(-1000, 5000), bins=6000)
            mids = 0.5*(bins[1:] + bins[:-1])

            moments[i]['channel']  = ch
            moments[i]['mean']     = np.average(mids, weights=hist)
            moments[i]['variance'] = np.average((mids - np.average(mids, weights=hist))**2, weights=hist)

        return moments
        
    def get_occupancy(self, data_s, data_b, channels, order=10):
        ''' 
        Function that occupancy (poisson parameter) of data distribution.
        
        #TODO: comments the important steps 
        
        Args:
            1. data_s: signal PMT(s) array.
            2. data_b: noise PMT(s) array.
            3. channels: list of PMTs n° to analize.
            
        Note:
        
        Output:
            1. Occupancy:
                - estimated occupancy: 
                - estimated occupancy error:
                - iteration: 
                - occupancy: 
                - occupancy error:
                - threshold: 
                - occupancy smooth: 
                - scaling factor:
                - entries: 
        '''  
    
        datatype = [('channel', np.int16), 
                    ('estimated occupancy', np.float32), 
                    ('estimated occupancy error', np.float32),
                    ('iteration', np.int16),
                    ('occupancy', object), ('occupancy error', object), ('threshold', object), 
                    ('occupancy smooth', object), ('scaling factor', np.float32), 
                    ('entries', np.int16)]

        Occupancy =  np.zeros(len(channels), dtype = datatype)

        moments_s = self.get_moments(data_s, channels)
        moments_b = self.get_moments(data_b, channels)

        for i, ch in enumerate(channels):
            area_s = data_s[data_s['channel']==ch]['area']
            signal, bins     = np.histogram(area_s, range=(-1000, 5000), bins=6000)

            area_b = data_b[data_b['channel']==ch]['area']
            background, bins = np.histogram(area_b, range=(-1000, 5000), bins=6000)

            E_s = moments_s[moments_s['channel']==ch]['mean']

            if E_s > 0:
                threshold = -150
            else:
                threshold = -50

            ini_threshold = threshold    
            end_threshold = 100
            start = np.digitize(-1000, bins)

            occupancy     = []
            occupancy_err = []
            thr           = []

            tot_entries_b = np.sum(background)

            while threshold < end_threshold: 
                bin_threshold = np.digitize(threshold, bins)
                
                Ab = np.sum(background[start:bin_threshold])
                As = np.sum(signal[start:bin_threshold])

                if Ab > 0 and As > 0:
                    f = Ab/tot_entries_b
                    l = -np.log(As/Ab)
                    l_err = np.sqrt((np.exp(l) + 1. - 2.*(Ab/tot_entries_b))/Ab)

                    if l_err/l <= 0.05:
                        occupancy.append(l)
                        occupancy_err.append(l_err)
                        thr.append(threshold)
                threshold += 1

            num = len(occupancy) - 1
            if num % 2 == 0:
                num = num - 1
            
            occupancy_smooth = savgol_filter(occupancy, num, order)
            dummy = occupancy_smooth.argsort()[::-1]
            for idx in range(0, len(dummy)):
                if occupancy_err[dummy[idx]]/occupancy[dummy[idx]] < 0.01:           
                    estimated_occupancy = occupancy[dummy[idx]]
                    estimated_occupancy_err = occupancy_err[dummy[idx]]
                    itr = dummy[idx]
                    break
                else:
                    estimated_occupancy = 0
                    estimated_occupancy_err = 0
                    itr = 0

            Occupancy[i]['channel']                   = ch
            Occupancy[i]['estimated occupancy']       = estimated_occupancy
            Occupancy[i]['estimated occupancy error'] = estimated_occupancy_err
            Occupancy[i]['iteration']                 = itr
            Occupancy[i]['occupancy']                 = occupancy
            Occupancy[i]['occupancy error']           = occupancy_err
            Occupancy[i]['threshold']                 = thr
            Occupancy[i]['occupancy smooth']          = occupancy_smooth
            Occupancy[i]['scaling factor']            = f
            Occupancy[i]['entries']                   = tot_entries_b

        return Occupancy

    def get_gainconversion(self, mu):
        ''' 
        Function that computed the gain from SPE ADC count.
        
        Args:
            1. mu: SPE ADC signal.
            
        Note:
            
        Output:
            1. gain: multiplication PMT factor.
        '''  
        Z = 50
        A = 10
        e = 1.6021766208e-19
        f = 1e8
        r = 2.25/16384

        gain = mu*r/(Z*A*f*e*1e6)

        return gain

    def get_gain(self, data_s, data_b, channels):
        ''' 
        Function that computed the gain from the occupancy.
        
        #TODO: comments the important steps 
        
        Args:
            1. data_s: signal PMT(s) array.
            2. data_b: noise PMT(s) array.
            3. channels: list of PMTs n° to analize.
            
        Note:
            
        Output:
            1. Gain: multiplication PMT factor.
        '''  
        datatype = [('channel', np.int16), 
                    ('gain', np.float32), 
                    ('gain error', np.float32)]
    
        moments_s = self.get_moments(data_s, channels)
        moments_b = self.get_moments(data_b, channels)

        Occupancy = self.get_occupancy(data_s, data_b, channels)

        Gain = np.zeros(len(channels), dtype = datatype)

        for i, ch in enumerate(channels):
            E_s = moments_s[moments_s['channel']==ch]['mean']
            V_s = moments_s[moments_s['channel']==ch]['variance']
            E_b = moments_b[moments_b['channel']==ch]['mean']
            V_b = moments_b[moments_b['channel']==ch]['variance']
            occupancy = Occupancy[Occupancy['channel']==ch]['estimated occupancy']
            occupancy_err = Occupancy[Occupancy['channel']==ch]['estimated occupancy error']
            tot_N = Occupancy[Occupancy['channel']==ch]['entries']
            f_b = Occupancy[Occupancy['channel']==ch]['scaling factor']

            EPsi = (E_s - E_b)/occupancy
            VPsi = (V_s - V_b)/occupancy - EPsi**2
            EPsi_stat_err = (occupancy*(EPsi**2 + VPsi) + 2.*V_b)/(tot_N*occupancy**2) + (EPsi*EPsi*(np.exp(occupancy) + 1. - 2.*f_b))/(f_b*tot_N*occupancy**2)
            EPsi_sys_err = (E_s - E_b)*occupancy_err/(occupancy**2)

            gain = self.convert_to_gain(EPsi)
            gain_err = self.convert_to_gain(np.sqrt(EPsi_stat_err)) + self.convert_to_gain(EPsi_sys_err)

            Gain[i]['channel']    = ch
            Gain[i]['gain']       = gain
            Gain[i]['gain error'] = gain_err

        return Gain
    
    def get_gainfunction(self, V, A, k):
        ''' 
        Gain function.
        
        Args:
            1. V: voltage value
            2. A k: parameter for gain function
           
        Note:
            
        Output:
            1. gain value

        '''  
        n = 12
        #k = 0.672
        #a = 0.018
        return A * V**(k*n)
    
    def get_HV_for_5e6_gain(self, Gain):
        ''' 
        Function that compute the HV for gain equal 5e6
        
        Args:
            1. Gain: gain obtained at 1500 V
           
        Note:
            
        Output:
            1. HV for gain equal to 5e6

        '''
        return 1500.*pow(5./Gain, 1./(0.672*12.))
    
                
################    
##### Plot #####
################

    def HV_scan(self, HV, Gain, Gain_err):
        ''' 
        Function for fitting gain as a function of HV.
        
        #TODO: write with lmfit
        
        Args:
            1. HV: volatge array at wihch we took gain measurement.
            2. Gain: gain value obtain.
            3: Gain_err: error on gain value.
           
        Note:
            
        Output:
            1. Parameter of fit function.

        '''        
        p0 = [1e-22, 0.7]
        popt, pcov = curve_fit(self.get_gainfunction, HV, Gain, p0= p0, sigma= Gain_err)
        A, k = popt
        dA, dk = [np.sqrt(pcov[j,j]) for j in range(popt.size)]
        resids = Gain - self.get_gainfunction(HV, A, k)
        redchisqr = ((resids/Gain)**2).sum()/float(len(Gain)-2)
        
        print('--Parameters-- \n', popt, '\n--Covariance matrix-- \n', pcov)
        print('Red chi squared: ', redchisqr)
        idx = np.where(np.asarray(HV)==1500)[0][0]
        HV_5e6 = self.get_HV_for_5e6_gain(Gain[idx])
        print('HV for 5e6 gain 1: ', HV_5e6)
        
        return A, dA, k, dk
    
    def plotwf(self, data, channel, event, **kwargs):
        '''
        Given waveform, it plots the ADC counts (1 ADC = 0.137mV) as a fucntion of the sampling (dt = 10 ns)

        Args:
            1. data: waveform from a PMT
            2. channel: PMT channel that you want to plot
            3. event: PMT's event that you want to plot
        Output:
            1. waveform plot
        '''
        wf = data[data['channel']==channel][event]
        start = pd.to_datetime(wf['time'])
        lenght = len(wf['data'])
        x = np.arange(0, lenght, 1)
        plt.figure(figsize=(10,6))
        plt.plot(x, wf['data'], linestyle='steps-mid', 
                 **kwargs)
        plt.xlabel("Sample [dt = 10 ns]", fontsize=14)
        plt.ylabel("ADC counts", fontsize=14)
        plt.tick_params(labelsize=20)
        plt.xlim(0,600)
        plt.ylim(-40,250)
        plt.show()
        
    def wfgiftplot(self, data, channel, window_led=None, window_noise=None):
        '''
        Given waveform, it plots the ADC counts (1 ADC = 0.137mV) as a fucntion of the sampling (dt = 10 ns)

        Args:
            1. data: waveform from a PMT
            2. channel: PMT channel that you want to plot
            3-4. windows: LED and noise window. Look configuration file in case.
        Output:
            1. gift of waveform
        '''
        
        PMT = data[data['channel']==channel]

        fig = plt.figure(figsize=(10,6))
        plt.style.use('seaborn-pastel')
        ax = plt.axes(xlim=(0, 600), ylim=(-40, 250))
        ax.set_xlabel('Sample [10ns]', fontsize=14)
        ax.set_ylabel('ADC [0.137mV]', fontsize=14)
        plt.tick_params(labelsize=20)
        if window_led != None:
            ax.axvspan(window_led[0], window_led[1], alpha=0.5, color='gold')
        if window_noise != None:
            ax.axvspan(window_noise[0], window_noise[1], alpha=0.5, color='lightblue')
        def init():
            wf.set_data([ ], [ ])
            return wf,
        def animate(i):
            x = np.arange(0, 600, 1)#
            y = PMT['data'][i]
            wf.set_data(x, y)
            return wf,
        wf, = ax.plot([ ],[ ], lw=2)
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=None, interval=len(PMT)/50, 
                                       blit=True, repeat=False)
        
        Config = cp.ConfigParser()
        Config.read(self.config_file)
        path_to_save_date = Config.get('plots_pars','path_plot') + str(self.date) + '/'
        
        print('Hey I am saving the gif here: ', path_to_save_date+str(channel)+'PMT.gif')
        anim.save(path_to_save_date+str(channel)+'PMT.gif', writer='imagemagick')
        anim.event_source.stop()
        del anim
        plt.close()
        
    def ledwindowplot(self, data, lastchannel):
        ''' 
        Function that plot the data from get_ledwindow.
        
        Args:
            1. data: df_led output from get_ledwindwow
            2. channel: last channel
        Note:
            1. the number of PMT is definively wrong.    
        '''
        fig = plt.figure(figsize=(22,10))
        plt.subplot(121)
        plt.hist2d(data['channel'], data['idx_led'], bins=(lastchannel,600), range=((0,lastchannel),(0,600)), cmap=plt.cm.plasma, norm=mpl.colors.LogNorm(), cmin = 1,alpha = 1)
        plt.colorbar(label='Number of events')
        plt.xlabel('PMT', fontsize=14)
        plt.ylabel('Sample [10ns]', fontsize=14)
        plt.tick_params(labelsize=20)
        
        plt.subplot(122)
        plt.hist(data['idx_led'], bins=600, range=(0,600), histtype='step', color='black')
        plt.xlim(0,600)
        median = np.nanmedian(data['idx_led'])
        std = np.nanstd(data['idx_led'])
        window = [int(median-0.5*std),int(median+0.5*std)] 
        plt.axvspan(*window, color='grey', alpha=0.2)
        plt.title('LED window: '+str(median-0.5*std)+' - '+str(median+0.5*std))
        plt.xlabel('Number of events', fontsize=14)
        plt.ylabel('Sample [10ns]', fontsize=14)
        plt.tick_params(labelsize=20)
        
        #Config = cp.ConfigParser()
        #Config.read(self.config_file)
        #path_to_save_date = Config.get('plots_pars','path_plot') + str(self.date) + '/'
        #print('Hey I am saving the figure here: ', path_to_save_date+'led_window.png')
        
        plt.show()
        plt.tight_layout()
        

        

    
    
    