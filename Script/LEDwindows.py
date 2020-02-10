import sys
sys.path.append('/home/gvolta/XENONnT/LedAnalysis/Script')
from init import *
import configparser as cp
Config = cp.ConfigParser()
Config.read('/home/gvolta/XENONnT/LedAnalysis/Script/conf.ini')
date = datetime.datetime.today().strftime('%Y%m%d')
print('Inizio window.py (GB): ', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000000)

### Strax configuration ###
import strax
import straxen
strax.Mailbox.DEFAULT_MAX_MESSAGES = 2
st = straxen.contexts.strax_SPE()
###########################

n_pmt = Config.getint('pmt_pars','n_pmt')
n_top_pmt = Config.getint('pmt_pars','n_top_pmt')
n_bottom_pmt = Config.getint('pmt_pars','n_bottom_pmt')

runs = st.select_runs(run_mode='LED*')
run_id = Config.get('runs_pars','run_id')
data_rr = st.get_array(run_id, 'raw_records', seconds_range=(0,5))

datatype = [('pmt', np.int16),
            ('Amplitude', np.float32),
            ('Sample of Amplitude', np.float32)]
Data = np.zeros((len(data_rr)), dtype = datatype)
for i in range(len(data_rr)):
    Data[i]['pmt'] = data_rr['channel'][i]
    Data[i]['Amplitude'] = np.max(data_rr['data'][i])
    Data[i]['Sample of Amplitude'] = np.argmax(data_rr['data'][i])
window = SPErough(data = Data)
length = window[1]-window[0]
window_noise = [5, 5+length]

print('LED window: ', window)
print('Nosie window: ', window_noise)

path_plot = Config.get('plots_pars','path_plot')
save_plot = Config.getboolean('plots_pars','save_plot')
if save_plot == True:
    wf_gift_plot(data_rr, n_pmt, window, window_noise)

Config.set("window_pars", "led_windows_left", str(window[0]))          # update
Config.set("window_pars", "led_windows_right", str(window[1]))         # update
Config.set("window_pars", "noise_windows_left", str(window_noise[0]))  # update
Config.set("window_pars", "noise_windows_right", str(window_noise[1])) # update

with open('/home/gvolta/XENONnT/LedAnalysis/Script/conf.ini', 'w+') as configfile:
    Config.write(configfile)
    
del Data, data_rr
print('Fine window.py (GB): ', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000000)