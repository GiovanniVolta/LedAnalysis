### Python Initialization ###
import numpy as np
import h5py

import scipy as sp
from scipy.optimize import curve_fit

import pandas as pd
pd.options.display.max_colwidth = 100

import numba

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
mpl.use('Agg')

from tqdm import tqdm
import datetime
import resource

import os
import os.path
#############################

### For fitting ###
def gaussian(x,*p) :
    # A gaussian peak with:
    #   Peak height above background : p[0]
    #   Central value                : p[1]
    #   Standard deviation           : p[2]
    return p[0]*np.exp(-1*(x-p[1])**2/(2*p[2]**2))
def gaussian2(x,*p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    return p[0]+p[1]*np.exp(-1*(x-p[2])**2/(2*p[3]**2))
###################

### Useful functions ###
def plot_peak(p, **kwargs):
    '''
    Given waveform, it plots the ADC counts (1 ADC = 0.137mV) as a fucntion of the sampling (dt = 10 ns)
    For example: PMT_25_rr_LED = data_rr_LED[data_rr_LED['channel']==25]
                 plot_peak_2(PMT_25_rr_LED[0], label=pd.to_datetime(PMT_25_rr_LED[0]['time']))
    '''
    start = pd.to_datetime(p['time'])
    lenght = len(p['data'])
    x = np.arange(0, lenght, 1)
    plt.plot(x, p['data'], linestyle='steps-mid',
             **kwargs)
    plt.xlabel("Sample [dt = 10 ns]")
    plt.ylabel("ADC counts")
    print('Start ' + str(p['channel']) + ': ' + str(pd.to_datetime(start)))

def wf_gift_plot(data_rr, n_pmt, window, window_noise):
    for channel in tqdm(range(n_pmt)):
        PMT = data_rr[data_rr['channel']==channel]
        fig = plt.figure(figsize=(10,6))
        plt.style.use('seaborn-pastel')
        ax = plt.axes(xlim=(0, 600), ylim=(-40, 150))
        ax.set_xlabel('Sample (10ns)')
        ax.set_ylabel('ADC (0.137mV)')
        ax.axvspan(window[0], window[1], alpha=0.5, color='gold')
        ax.axvspan(window_noise[0], window_noise[1], alpha=0.5, color='lightblue')
        def init():
            wf.set_data([ ], [ ])
            return wf,
        def animate(i):
            ax.text(10, 120, s=str(i), bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
            x = np.arange(0, 600, 1)#
            y = PMT['data'][i]
            wf.set_data(x, y)
            return wf,
        wf, = ax.plot([ ],[ ], lw=2)
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=None, interval=100, blit=True, repeat=False)
        print('Hey sto provando a salvare: ', path_plot+date+'/'+str(channel)+'PMT.gif')
        anim.save(path_plot+date+'/'+str(channel)+'PMT.gif', writer='imagemagick')
        anim.event_source.stop()
        del anim
        plt.close()

    
###############################  
######## SPE functions ########
###############################

def SPErough(data, n_channel_s = np.arange(0, 248, 1)):
    
    # 1. Identify the rough amplitude range corresponding to a single photoelectron
    # 2. Find the time window in which we have an excess of samples in this amplitude range.
    
    info = {'pmt': [ ], 'LED_mu': [ ], 'LED_sigma': [ ], 'LED_norm': [ ]}
    df2 = pd.DataFrame({'channel': [ ], 'idx_LED': [ ]}) 

    for n_channel in tqdm(n_channel_s):
        wf = data[data['pmt']==n_channel]
        ##################################################################
        # Which binning and range you want for the SPE fit?
        # I am checking the first bin empty and it defines the binning and 
        # the range for the next hist
        ##################################################################
        hist, xbins = np.histogram(wf['Amplitude'], bins=50, range=(0,300))
        xbins_center = np.array([0.5*(xbins[i]+xbins[i+1]) for i in range(len(xbins)-1)])
        y = hist[np.where(hist<1)]
        x = xbins_center[np.where(hist<1)]

        #########################################
        # This is just for data visualization
        # (x, y+1) are the point where there are no
        # data in the histogram
        #########################################
        ###############################################
        # L'estremo superiore del range e' definito
        # come il primo valore con 0 counts dopo 30 ADC
        ###############################################
        idx = x[np.where(x>30)]

        if len(idx)==0:
            hist, xbins = np.histogram(wf['Amplitude'], bins=150, range=(0,300))
        else:
            hist, xbins = np.histogram(wf['Amplitude'], bins=int(idx[0]/2), range=(0, idx[0]))

        xbins_center = np.array([0.5*(xbins[i]+xbins[i+1]) for i in range(len(xbins)-1)])
        #############################
        # ADC del piedistallo
        # usato come input per il fit
        #############################
        idx_0PE_argmax = np.argmax(hist)
        #####################################################
        # temp e' usato per trovare il segnale di 1PE
        # da usare come input nel fit. Viene preso il massimo 
        # dopo ~10 ADC
        #####################################################
        temp = hist[(idx_0PE_argmax+4):]
        idx_1PE_argmax = np.argmax(temp) + (idx_0PE_argmax+4)
        ######################
        # Range in cui fittare
        # il segnale di 1PE
        ######################
        low = np.argmin(hist[idx_0PE_argmax:idx_1PE_argmax]) + (idx_0PE_argmax+4)
        high = int(idx_1PE_argmax*1.1)
        if np.max(temp) > 1e1:
            try:
                init_1PE = [hist[idx_1PE_argmax], idx_1PE_argmax, 5]
                ###########
                #   FIT
                ###########
                popt_1PE, pcov_1PE = curve_fit(gaussian, xbins_center[low:high], hist[low:high], sigma=np.sqrt(hist[low:high]), p0= init_1PE, maxfev=int(1e6))
            except:
                print('THE FIT HAS FAILED')
                ##########
                # NO FIT
                ##########
                popt_1PE = [0,0,0]

        if np.max(temp) < 1e1:
            ##########
            # No 1PE
            ##########
            popt_1PE = [0,0,0]

        N, mean, sig = popt_1PE[0], popt_1PE[1], popt_1PE[2]
        ############################
        # Salvo i parametri in un
        # dizionario. Non si sa mai
        ############################

        info['pmt'].append(n_channel)
        info['LED_mu'].append(mean)
        info['LED_sigma'].append(sig)
        info['LED_norm'].append(N)

        d_temp = pd.DataFrame({'channel': [ ], 'idx_LED': [ ]})
        #######################################
        # Selezione solo gli eventi che cadono
        # dentro al segnale di 1PE e poi prendo
        # il corrispondente sample time
        #######################################

        mask = (wf['Amplitude'] < mean + sig) & (wf['Amplitude'] > mean - sig)
        idx_LED = wf['Sample of Amplitude'][mask]

        if len(idx_LED)==0:
            d_temp['channel'] = n_channel
            d_temp['idx_LED'] = np.nan
        else:
            d_temp['idx_LED'] = idx_LED
            d_temp['channel'] = np.ones_like(idx_LED) * n_channel

        df2 = df2.append(d_temp, ignore_index=True)
        del d_temp, idx_LED
    
    median = np.nanmedian(df2['idx_LED'])
    std = np.nanstd(df2['idx_LED'])
    window = [int(median-0.5*std),int(median+0.5*std)]
    return window, info, df2
    
def ScalingSpectrum(data, n_channel_s = np.arange(0, 249, 1)):
    
    # In order to subtract out the contribution of the noise to the amplitude spectrum, we will assume that 
    # the fraction of SPE signals with amplitude below a threshold of(3-7) ADC counts is very small. 
    # We then scale down the off-time amplitude spectrum such that the total counts below the 
    # (3-7) ADC count threshold is the same as in the LED spectrum.
    # The spectrum also contains contributions of 2 or more photoelectrons. From the scaling down factor 
    # of the noise s, assuming a Poisson distribution of photoelectrons we estimate that the average 
    # number of photoelectrons (occupancy) in the LED run was lambda = -ln(s) = 0.566.
    # The fraction of events with 2 or more photoelectrons is then 1- exp(-lambda)(1+lambda) = 0.111. 
    # The contribution of 2 or more photoelectrons leads to a slight over-estimate in the acceptances calculated.
    
    datatype = [('pmt', np.int16),
                ('spectrumLED', object), ('bins_LED_center', object),
                ('spectrumNOISE', object), ('bins_NOISE_center', object),
                ('spectrumNOISE_scaled_3bin', object), ('occupancy_3bin', np.float32),
                ('spectrumNOISE_scaled_4bin', object), ('occupancy_4bin', np.float32),
                ('spectrumNOISE_scaled_5bin', object), ('occupancy_5bin', np.float32),
                ('spectrumNOISE_scaled_6bin', object), ('occupancy_6bin', np.float32),
                ('spectrumNOISE_scaled_7bin', object), ('occupancy_7bin', np.float32)]

    SPE = np.zeros((len(n_channel_s)), dtype = datatype)

    for n_channel in tqdm(n_channel_s):
        arr = data[data['channel'] == n_channel]

        LED, bins_LED = np.histogram(arr['amplitude_led'], bins=250, range=(0,500))
        bins_LED_center = 0.5 * (bins_LED[1:] + bins_LED[:-1])
        noise, bins_noise = np.histogram(arr['amplitude_noise'], bins=250, range=(0,500))
        bins_noise_center = 0.5 * (bins_noise[1:] + bins_noise[:-1])
        
        SPE[n_channel]['pmt'] = n_channel
        SPE[n_channel]['spectrumLED'] = LED
        SPE[n_channel]['bins_LED_center'] = bins_LED_center
        SPE[n_channel]['spectrumNOISE'] = noise
        SPE[n_channel]['bins_NOISE_center'] = bins_noise_center
        
        for i in range(3,8):
            ADC_correction = i
            scaling_coeff = LED[:i].sum()/noise[:i].sum() 
            SPE[n_channel]['spectrumNOISE_scaled_'+str(i)+'bin'] = noise*scaling_coeff
            SPE[n_channel]['occupancy_'+str(i)+'bin'] = -np.log(scaling_coeff)
    
    return SPE

def SPE_Acceptance(data, n_channel_s = np.arange(0, 249, 1)):
    
    # The acceptance as a function of amplitude (threshold) is defined as the fraction of 
    # noise-subtracted single photoelectron spectrum above that amplitude.
    
    datatype = [('pmt', np.int16),
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

    SPE_acceptance = np.zeros((len(n_channel_s)), dtype = datatype)
    j=0
    for n_channel in tqdm(n_channel_s):
        arr = data[data['pmt'] == n_channel]
        SPE_acceptance[j]['pmt'] = j
        
        for i in range(3,8):
            diff = np.absolute(arr['spectrumLED'][0] - arr['spectrumNOISE_scaled_'+str(i)+'bin'][0])
            sigma_diff = np.sqrt(arr['spectrumLED'][0] + arr['spectrumNOISE_scaled_'+str(i)+'bin'][0])

            res =  1. - np.cumsum(diff)/np.sum(diff)
            x_center = arr['bins_LED_center'][0]
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
    

