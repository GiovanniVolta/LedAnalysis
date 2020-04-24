import sys
import strax
import straxen
import configparser as cp

sys.path.append('/home/gvolta/XENONnT/LedAnalysis/Commissioning/led_window/')
from PMTanalysis import *
PMT = PMTanalyzer('/home/gvolta/XENONnT/LedAnalysis/Commissioning/led_window/configuration_led_window.ini')
mpl.use('Agg')
### RUNS CONFIG SETTINGS ###

Config = cp.ConfigParser()
Config.read(PMT.config_file)
print(PMT.config_file)

run1 = Config.get('runs_pars','run1')
run2 = Config.get('runs_pars','run2')
run3 = Config.get('runs_pars','run3')
run4 = Config.get('runs_pars','run4')
run5 = Config.get('runs_pars','run5')
run6 = Config.get('runs_pars','run6')
run7 = Config.get('runs_pars','run7')
run8 = Config.get('runs_pars','run8')
run_noise = Config.get('runs_pars', 'run_noise')
print(run1, run2, run3, run4, run5, run6, run7, run8, run_noise)

led_window   = [Config.getint('window_pars','led_windows_left'), Config.getint('window_pars','led_windows_right')]
noise_window = [Config.getint('window_pars','noise_windows_left'), Config.getint('window_pars','noise_windows_right')]
print('led_window: ', led_window, '\tnoise_window: ', noise_window)

strax.Mailbox.DEFAULT_MAX_MESSAGES = 2
st_ = straxen.contexts.xenonnt_led()
st = st_.new_context(storage=[strax.DataDirectory(PMT.straxdata, provide_run_metadata=False)], 
                     config=dict(led_window=(led_window[0],led_window[1]), noise_window=(noise_window[0], noise_window[1]),
                                 channel_list=(0,494)))

print(st.show_config('led_calibration'))

### MAKE LED RUN ###
data = st.get_array([run4, run8, run_noise], 'led_calibration', max_workers=10)
                     
                     
