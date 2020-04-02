import strax
import straxen
import configparser as cp

strax.Mailbox.DEFAULT_MAX_MESSAGES = 2
#st = straxen.contexts.strax_SPE()
strax.mailbox.DEFAULT_TIMEOUT = 3600
st = straxen.contexts.xenon1t_dali()
runs = st.select_runs()

### EDIT BELOW TO CHANGE CONFIG SETTINGS ###
Config = cp.ConfigParser()
Config.read('/home/gvolta/XENONnT/LedAnalysis/Script/Class/configuration.ini')

straxdata_rawrecords = Config.get('runs_pars','straxdata_rawrecords')

straxdata_spe   = Config.get('runs_pars','straxdata_spe')
run_spe_topbulk = Config.get('runs_pars','run_spe_topbulk')
run_spe_topring = Config.get('runs_pars','run_spe_topring')

straxdata_gain  = Config.get('runs_pars','straxdata_gain')
run_gain_step0  = Config.get('runs_pars', 'run_gain_step0')
run_gain_step1  = Config.get('runs_pars', 'run_gain_step1')
run_gain_step2  = Config.get('runs_pars', 'run_gain_step2')
run_gain_step3  = Config.get('runs_pars', 'run_gain_step3')
run_gain_step4  = Config.get('runs_pars', 'run_gain_step4')

light_level     = Config.get('runs_pars','light_level')

led_window   = [Config.getint('window_pars','led_windows_left'), Config.getint('window_pars','led_windows_right')]
noise_window = [Config.getint('window_pars','noise_windows_left'), Config.getint('window_pars','noise_windows_right')]

plugin = 'led_calibration'
print(run_spe_topring, plugin)
############################################

st = st.new_context(storage=[strax.DataDirectory(straxdata_spe,
                                provide_run_metadata=False)],
                    config=dict(led_window=(led_window[0],led_window[1]), 
                                noise_window=(noise_window[0], noise_window[1])))

#data = st.get_array(run_spe_topring, plugin, seconds_range=(0,20))
#print(data)
st.make(run_spe_topring, plugin)