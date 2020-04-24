import sys
print(sys.executable)
import strax
import straxen
print('Straxen: ', straxen.__version__, straxen.__file__)
print('Strax: ', strax.__version__, strax.__file__)
import configparser as cp

sys.path.append('/home/gvolta/XENONnT/LedAnalysis/Commissioning/led_window/')
from PMTanalysis import *
PMT = PMTanalyzer('/home/gvolta/XENONnT/LedAnalysis/Commissioning/led_window/configuration_led_window.ini')
mpl.use('Agg')
### RUNS CONFIG SETTINGS ###

Config = cp.ConfigParser()
Config.read(PMT.config_file)
print(PMT.config_file)

run = Config.get('runs_pars','run_commissioning')
print(run)

strax.Mailbox.DEFAULT_MAX_MESSAGES = 2
st_ = straxen.contexts.xenonnt_online()
st = st_.new_context(storage=[strax.DataDirectory(PMT.straxdata, provide_run_metadata=False)])

data_rr_ = st.get_array(run, 'raw_records', max_workers=20, seconds_range=(0, 60))
data_rr  = PMT.get_signal(data_rr_)
amplitude = PMT.get_amplitude(data_rr, window=[0,160])

path_to_save_date = Config.get('output_pars','path_data_commissioning') + str(PMT.date) + '/'
name = 'Amplitude_'+run
print('Salvo amplitude qui: '+path_to_save_date+name)
np.savez(path_to_save_date+name, x=amplitude)


channels = np.arange(0, 494, 1)

## Plot of Amplitude spectrum

path_plot_commissioning = Config.get('plots_pars', 'path_plot_commissioning') + str(PMT.date) + '/'
print('Salvo lo spettro di qui: '+path_plot_commissioning+'AmpSpec/run'+run+'_PMT...png')
for ch in channels:
    try:
        fig = plt.figure(figsize=(10, 5))
        ADC = np.arange(-10, 500, 1)
        plt.hist(amplitude[amplitude['channel']==ch]['amplitude'], bins=ADC, label='PMT %s'%(str(ch)))
        plt.yscale('log')
        plt.grid(color='grey', linestyle='-', linewidth=2, alpha=0.25)
        plt.xlim(0,200)
        plt.legend(loc=0)
        plt.savefig(path_plot_commissioning+'AmpSpec/run'+run+'_PMT'+str(ch)+'.png')
        plt.close(fig)
    except:
        print('Something went wrong in PMT ', ch)
    
## SPE rough spectrum fit
print('Salvo il fit qui: '+path_plot_commissioning+'SPEfit/run'+run+'_PMT...png')
bad_ch = [ ]
for ch in channels:
    try:
        gauss, result_fit, fit_interval = PMT.get_sperough(amplitude=amplitude, channel=ch)
        def gaussian(x,*p) :
            # A gaussian peak with:
            #   Peak height above background : p[0]
            #   Central value                : p[1]
            #   Standard deviation           : p[2]
            return p[0]*np.exp(-1*(x-p[1])**2/(2*p[2]**2))

        plt.figure(figsize=(10,5))
        fit_input, binning, check = PMT.get_speinput(amplitude=amplitude, channel=ch)
        H, B, _  = plt.hist(amplitude[amplitude['channel']==ch]['amplitude'], 
                            bins=binning, label='PMT %s'%(str(ch)))

        plt.plot(B[fit_input], H[fit_input], '*')
        result_fit.plot_fit(show_init=True)

        popt  = result_fit.best_values
        N     = popt['g_amplitude']/(np.sqrt(2*np.pi)*popt['g_sigma'])
        mu    = popt['g_center']
        sigma = popt['g_sigma']

        x = np.linspace(0,100, 100)
        y = gaussian(x, N, mu, sigma)
        plt.plot(x, y, 'k--')

        plt.yscale('log')
        plt.ylim(1e-1)
        plt.savefig(path_plot_commissioning+'SPEfit/run'+run+'_PMT'+str(ch)+'.png')
        plt.close(fig)
    except:
        print('Something went wrong in PMT ', ch)
        
### Good channel
good_ch = [ ]
for ch in channels:
    if ch not in bad_ch:
        good_ch.append(ch)
print('I channels buoni sono: ', good_ch)

### LED window
path_to_save_date = Config.get('output_pars','path_data_commissioning') + str(PMT.date) + '/'
name = 'LED_window_run'+run
print('Salvo il fit qui: '+path_to_save_date+name)
try:
    df_fit, df_led, window = PMT.get_ledwindow(amplitude , good_ch)
    print(window)
    np.savez(path_to_save_date+name, x=df_fit, y=df_led)
except:
    print('Somethin went wrong')
  
