import sys
sys.path.append('/home/gvolta/XENONnT/LedAnalysis/Script')
from Initialization import *
import configparser as cp
Config = cp.ConfigParser()
Config.read('/home/gvolta/XENONnT/LedAnalysis/Script/conf.ini')
date = datetime.datetime.today().strftime('%Y%m%d')

print('Inizio SPE.py (GB): ', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000000)

### Strax configuration ###
import strax
import straxen
strax.Mailbox.DEFAULT_MAX_MESSAGES = 2
st = straxen.contexts.strax_SPE()
###########################

n_pmt = Config.getint('pmt_pars','n_pmt')
n_top_pmt = Config.getint('pmt_pars','n_top_pmt')
n_bottom_pmt = Config.getint('pmt_pars','n_bottom_pmt')
window = [Config.getint('window_pars','led_windows_left'), Config.getint('window_pars','led_windows_right')]
window_noise = [Config.getint('window_pars','noise_windows_left'), Config.getint('window_pars','noise_windows_right')]

print('LED window: ', window[0], window[1], type(window))
print('Noise window: ', window_noise[0], window_noise[1], type(window_noise))

runs = st.select_runs(run_mode='LED*')
run_id = Config.get('runs_pars','run_id')

print('Prima di get array (GB): ', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000000)
st = st.new_context(config=dict(led_window=(window[0],window[1]), noise_window=(window_noise[0], window_noise[1])))
data_SPE = st.get_array(run_id, 'led_calibration')
SPE = ScalingSpectrum(data = data_SPE)
SPE_acceptance = SPE_Acceptance(data = SPE)

path_data = Config.get('output_pars','path_data')
save_data = Config.getboolean('output_pars','save_data')
if save_data == True:
    np.save(path_data+date+'/SPE_hist.npy', SPE)
    np.save(path_data+date+'/SPE_acceptance.npy', SPE_acceptance)
        
print('Fine SPE.py (GB): ', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000000)
        

