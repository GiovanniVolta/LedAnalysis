import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
import scipy as sp
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
        
        self.run_led = Config.get('runs_pars','run_led')
        self.run_noise = Config.get('runs_pars','run_noise')
        self.light_level = Config.get('runs_pars','light_level')


    def get_amplitude(self, data, window=None):
        '''
        Look for signal amplitude in a given window. 
    
        Args:
            1. data: waveform from a PMT
            2. window: define the sample range where look for the signal amplitude. If None, all the waveform it will be scan.
        Note:
            1. the input data is suppose to have 'channel' and 'data' entries, defined as are in strax.
        Output:
            1. array with PMT channel entry and signal amplitude (in ADC usually) of each waveform.
        '''
        datatype = [('channel', np.int16), ('amplitude', np.float32), ('sample of amplitude', np.float32)]
        amplitude = np.zeros(len(data), dtype =datatype)
        for i, wf in enumerate(data['data']):
            if window != None: 
                amplitude['channel'][i]   = data['channel'][i]
                amplitude['amplitude'][i] = np.max(wf[int(window[0]):int(window[1])])
                amplitude['sample of amplitude'][i] = np.argmax(wf[int(window[0]):int(window[1])])
            else:
                amplitude['channel'][i]   = data['channel'][i]
                amplitude['amplitude'][i] = np.max(wf)
                amplitude['sample of amplitude'][i] = np.argmax(wf)
        amplitude.sort(order='channel')
        return amplitude
    
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
        topnoise,valley, endvalley, spebump,endbin = inputSPE
        idx_1 = endvalley
        idx_2 = spebump + (spebump-endvalley)
        
        if idx_1 < endvalley:
            idx_1 = endvalley
        if idx_2 > endbin:
            idx_2 = endbin
            
        if checkSPE == 1:      
            gauss.set_param_hint('g_height', value=H[spebump], min=H[spebump]-30, max=H[spebump]+30)
            gauss.set_param_hint('g_center', value=B[spebump], min=B[spebump]-30, max=B[spebump]+30)
            gauss.set_param_hint('g_sigma', value=idx_2-idx_1, max=(idx_2-idx_1)+30)
            result_fit = gauss.fit(H[idx_1:idx_2], x=B[idx_1:idx_2], weights=1.0/np.sqrt(H[idx_1:idx_2]))
            
        else:
            #if channel != None:
            #    print('PMT: ', int(channel), '\t :(')
            #print('No SPE')
            gauss  = 0
            result_fit = 0
        return gauss, result_fit
    
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
            _, fit = self.get_sperough(amplitude, ch)
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
            #mask    = (PMT['amplitude'] < mean + sigma) & (PMT['amplitude'] > mean - sigma)

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
        window = [int(median-0.5*std),int(median+0.5*std)]       
        
        length = window[1]-window[0]
        window_noise = [5, 5+length]
        
        Config = cp.ConfigParser()
        Config.read(self.config_file)
        Config.set("window_pars", "led_windows_left", str(window[0]))          # update
        Config.set("window_pars", "led_windows_right", str(window[1]))         # update
        Config.set("window_pars", "noise_windows_left", str(window_noise[0]))  # update
        Config.set("window_pars", "noise_windows_right", str(window_noise[1])) # update
        with open(self.config_file, 'w+') as configfile:
            Config.write(configfile)
            
        return df_fit, df_led, window
        
                
################    
##### Plot #####
################
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
        

        

    
    
    