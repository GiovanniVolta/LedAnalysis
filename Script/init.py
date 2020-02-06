### Strax configuration ###
import strax
import straxen
st = straxen.contexts.strax_workshop_dali()
st.register(straxen.plugins.led_calibration.LEDCalibration)
print(st.show_config('led_calibration'))
###########################

### Python Initialization ###
import numpy as np

import scipy as sp
from scipy.optimize import curve_fit

import pandas as pd
pd.options.display.max_colwidth = 100

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 24
plt.rc('font', size=MEDIUM_SIZE)                                   # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)                              # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)                              # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)                             # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)                             # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE, loc = 'best')               # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE, figsize=(15,8))            # fontsize of the figure title
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

from tqdm import tqdm
#############################

### OS ###
import os
import os.path
##########

### For fitting ###
def gaussian(x,*p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    return p[0]+p[1]*np.exp(-1*(x-p[2])**2/(2*p[3]**2))
def gaussian2(x,*p) :
    # A gaussian peak with:
    #   Peak height above background : p[0]
    #   Central value                : p[1]
    #   Standard deviation           : p[2]
    return p[0]*np.exp(-1*(x-p[1])**2/(2*p[2]**2))
def gaus(x, a, mu, sig):
    return a*np.exp(-0.5*(x-mu)**2/sig**2)
# TODO: elimina gaus e modifica in Windows_Identification
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
    
#def plot_wf(ch, df):
#    '''
#    Given the dataframe (data_rr) and selecting the channel
#    the waveforms are plotted as animation.
#    TODO: add **kwargs for customization.
#    '''
#    channel = ch
#    PMT = df[df['channel']==0]
#
#    # First set up the figure, the axis, and the plot element we want to animate
#    fig = plt.figure(figsize=(10,5))
#    ax = plt.axes(xlim=(0, 600), ylim=(-10, 200))
#    wf, = ax.plot([ ],[ ], lw=2)
#    # initialization function: plot the background of each frame
#    def init():
#        wf.set_data([ ], [ ])
#        return wf,
#    # animation function.  This is called sequentially
#    def animate(i):
#        x = np.arange(0, 600, 1)
#        y = PMT['data'][i]
#        wf.set_data(x, y)
#        return wf,
#    # call the animator.  blit=True means only re-draw the parts that have changed.
#    #anim = animation.FuncAnimation(fig, animate, init_func=init,
#    #                               frames=200, interval=20, blit=True)
#
#    #anim.save('basic_animation.gif', fps=30, extra_args=['-vcodec', 'libx264'])
#    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, interval=20, blit=True)
#    #return anim

########################

##############################
#### XENON1T configuration ###
##############################

# ROOT key alias
MC_branch_alias = [['rrp_pri','(xp_pri*xp_pri + yp_pri*yp_pri)/10.']]

# variable ranges [cm]
LXe_minZ  = -96.7
LXe_maxZ  = 0.
LXe_minR  = -47.9
LXe_maxR  = 47.9
LXe_minRR = 0.
LXe_maxRR = 2294.41
LXe_Cut = "zp_pri/10.<={0} && zp_pri/10.>={1} && rrp_pri/10.>={2} && rrp_pri/10.<={3}".format(LXe_maxZ,LXe_minZ,LXe_minRR,LXe_maxRR)

GXe_minZ  = 0.
GXe_maxZ  = 0.6
GXe_minR  = -47.9
GXe_maxR  = 47.9
GXe_minRR = 0.
GXe_maxRR = 2294.41
GXe_Cut = "zp_pri/10.<={0} && zp_pri/10.>={1} && rrp_pri/10.>={2} && rrp_pri/10.<={3}".format(GXe_maxZ,GXe_minZ,GXe_minRR,GXe_maxRR)

TPC_minZ  = -140.
TPC_maxZ  = 6.
TPC_minR  = -60.
TPC_maxR  = 60.
TPC_minRR = 0.
TPC_maxRR = 3600.
TPC_Cut = "zp_pri/10.<={0} && zp_pri/10.>={1} && rrp_pri/10.>={2} && rrp_pri/10.<={3}".format(TPC_maxZ,TPC_minZ,TPC_minRR,TPC_maxRR)

QE_top = 0.314
QE_bottom = 0.366
PMTs_top = 127
PMTs_bottom = 121

nbinsZ  = 100
nbinsRR = 100
nbinsR  = 100
LCE_min = 0
LCE_max = 100

# PMT values as in PAX
# (Hamamatsu datasheets)
PMT_QEs = [0.281,0.283,0.284,0.284,0.286,0.287,0.290,0.293,0.293,0.294,0.295,0.296,0.296,0.297,0.329,0.298,0.299,0.301,0.302,0.304,0.305,0.308,0.308,0.308,0.309,0.309,0.310,0.310,0.310,0.311,0.312,0.314,0.330,0.315,0.315,0.315,0.315,0.316,0.316,0.317,0.319,0.327,0.320,0.320,0.321,0.342,0.322,0.322,0.322,0.322,0.322,0.321,0.323,0.323,0.323,0.323,0.324,0.325,0.325,0.325,0.325,0.325,0.325,0.325,0.325,0.326,0.326,0.326,0.326,0.326,0.319,0.327,0.328,0.328,0.328,0.328,0.328,0.297,0.329,0.329,0.329,0.329,0.329,0.329,0.330,0.330,0.331,0.331,0.331,0.332,0.332,0.332,0.332,0.333,0.334,0.334,0.335,0.336,0.336,0.337,0.338,0.339,0.339,0.340,0.340,0.340,0.350,0.341,0.341,0.341,0.341,0.341,0.341,0.341,0.342,0.342,0.342,0.321,0.342,0.342,0.342,0.342,0.343,0.343,0.345,0.345,0.345,0.404,0.405,0.413,0.352,0.350,0.316,0.360,0.346,
           0.346,0.347,0.347,0.348,0.347,0.329,0.359,0.374,0.360,0.361,0.361,0.361,0.349,0.394,0.347,0.358,0.380,0.385,0.374,0.376,0.376,0.363,0.349,0.389,0.342,0.357,0.372,0.385,0.396,0.386,0.387,0.376,0.363,0.351,0.389,0.359,0.357,0.371,0.385,0.395,0.400,0.397,0.390,0.376,0.363,0.351,0.359,0.356,0.371,0.385,0.393,0.399,0.401,0.397,0.390,0.378,0.363,0.352,0.343,0.356,0.371,0.385,0.393,0.399,0.397,0.391,0.380,0.363,0.353,0.334,0.315,0.336,0.370,0.384,0.393,0.391,0.391,0.373,0.364,0.353,0.400,0.339,0.356,0.369,0.383,0.382,0.381,0.381,0.364,0.353,0.387,0.357,0.356,0.368,0.367,0.367,0.367,0.366,0.354,0.322,0.334,0.356,0.355,0.354,0.354,0.354,0.354,0.324,0.320,0.383,0.374,0.326,0.324]
PMT_positions = [{'x': -12.345668451390225, 'y': 46.074661913988564},{'x': -4.1573289290632864, 'y': 47.518487098976266},{'x': 4.1573289290632935, 'y': 47.518487098976259},{'x': 12.345668451390232, 'y': 46.074661913988557},{'x': 20.158891085031385, 'y': 43.230881441648194},{'x': 27.359596013944909, 'y': 39.073552512584904},{'x': 33.728993462598353, 'y': 33.728993462598289},{'x': 39.073552512584904, 'y': 27.359596013944902},{'x': 43.230881441648222, 'y': 20.158891085031343},{'x': 46.074661913988564, 'y': 12.345668451390226},{'x': 47.518487098976266, 'y': 4.1573289290632882},{'x': 47.518487098976259, 'y': -4.1573289290632918},{'x': 46.07466191398855, 'y': -12.345668451390271},{'x': 43.230881441648194, 'y': -20.158891085031378},{'x': 39.073552512584889, 'y': -27.359596013944923},{'x': 33.728993462598318, 'y': -33.728993462598318},{'x': 27.359596013944891, 'y': -39.073552512584911},{'x': 20.158891085031346, 'y': -43.230881441648222},{'x': 12.34566845139023, 'y': -46.074661913988564},{'x': 4.1573289290632918, 'y': -47.518487098976266},{'x': -4.1573289290633095, 'y': -47.518487098976259},{'x': -12.345668451390264, 'y': -46.074661913988557},{'x': -20.158891085031378, 'y': -43.230881441648194},{'x': -27.359596013944902, 'y': -39.073552512584911},
                 {'x': -33.728993462598325, 'y': -33.728993462598304},{'x': -39.073552512584918, 'y': -27.359596013944888},{'x': -43.230881441648208, 'y': -20.158891085031357},{'x': -46.074661913988564, 'y': -12.345668451390232},{'x': -47.518487098976266, 'y': -4.1573289290632864},{'x': -47.518487098976259, 'y': 4.157328929063306},{'x': -46.074661913988557, 'y': 12.345668451390251},{'x': -43.230881441648201, 'y': 20.158891085031371},{'x': -39.073552512584904, 'y': 27.359596013944902},{'x': -33.728993462598311, 'y': 33.728993462598318},{'x': -27.359596013944895, 'y': 39.073552512584911},{'x': -20.158891085031357, 'y': 43.230881441648208},{'x': -10.288057042825152, 'y': 38.395551594990479},{'x': -2.0803542606570264, 'y': 39.695524006494317},{'x': 6.2182699853491794, 'y': 39.260611538656725},{'x': 14.245125994425699, 'y': 37.109821953263769},{'x': 21.649401641847319, 'y': 33.337155075830609},{'x': 28.107494552165267, 'y': 28.107494552165264},{'x': 33.337155075830609, 'y': 21.649401641847316},{'x': 37.109821953263783, 'y': 14.245125994425663},{'x': 39.260611538656732, 'y': 6.2182699853491421},{'x': 39.695524006494303, 'y': -2.0803542606570282},{'x': 38.395551594990451, 'y': -10.288057042825226},{'x': 35.417509336487612, 'y': -18.046122364647005},
                 {'x': 30.891551967914587, 'y': -25.015485544231051},{'x': 25.015485544231034, 'y': -30.891551967914602},{'x': 18.046122364646969, 'y': -35.417509336487633},{'x': 10.288057042825173, 'y': -38.395551594990472},{'x': 2.080354260657014, 'y': -39.695524006494317},{'x': -6.2182699853491759, 'y': -39.260611538656725},
                 {'x': -14.245125994425695, 'y': -37.109821953263769},{'x': -21.649401641847341, 'y': -33.337155075830594},{'x': -28.107494552165271, 'y': -28.107494552165253},{'x': -33.337155075830609, 'y': -21.649401641847319},{'x': -37.109821953263776, 'y': -14.245125994425678},{'x': -39.260611538656725, 'y': -6.2182699853491723},{'x': -39.695524006494303, 'y': 2.0803542606570229},{'x': -38.395551594990465, 'y': 10.288057042825208},{'x': -35.417509336487619, 'y': 18.04612236464699},{'x': -30.891551967914587, 'y': 25.015485544231041},{'x': -25.015485544231034, 'y': 30.891551967914602},{'x': -18.04612236464698, 'y': 35.417509336487633},{'x': -8.2304456342601497, 'y': 30.716441275992377},{'x': 3.1086244689504383e-14, 'y': 31.800000000000001},{'x': 8.2304456342601817, 'y': 30.716441275992366},{'x': 15.900000000000009, 'y': 27.539607840345141},{'x': 22.485995641732213, 'y': 22.485995641732213},{'x': 27.539607840345159, 'y': 15.899999999999984},{'x': 30.716441275992374, 'y': 8.2304456342601515},{'x': 31.800000000000001, 'y': 0.0},{'x': 30.716441275992366, 'y': -8.2304456342601817},{'x': 27.539607840345138, 'y': -15.90000000000002},{'x': 22.485995641732202, 'y': -22.485995641732224},{'x': 15.899999999999999, 'y': -27.539607840345152},
                 {'x': 8.2304456342601533, 'y': -30.716441275992374},{'x': -1.3322676295501878e-14, 'y': -31.800000000000001},{'x': -8.2304456342601622, 'y': -30.716441275992374},{'x': -15.900000000000006, 'y': -27.539607840345145},{'x': -22.485995641732217, 'y': -22.485995641732202},{'x': -27.539607840345152, 'y': -15.899999999999993},{'x': -30.716441275992374, 'y': -8.2304456342601551},{'x': -31.800000000000001, 'y': 3.5527136788005009e-15},{'x': -30.71644127599237, 'y': 8.2304456342601675},{'x': -27.539607840345145, 'y': 15.900000000000006},{'x': -22.48599564173221, 'y': 22.485995641732217},{'x': -15.899999999999997, 'y': 27.539607840345152},{'x': -6.1728342256951123, 'y': 23.037330956994282},{'x': 2.0786644645316468, 'y': 23.759243549488129},{'x': 10.079445542515693, 'y': 21.615440720824097},{'x': 16.864496731299177, 'y': 16.864496731299145},{'x': 21.615440720824111, 'y': 10.079445542515671},{'x': 23.759243549488133, 'y': 2.0786644645316441},{'x': 23.037330956994275, 'y': -6.1728342256951354},{'x': 19.536776256292445, 'y': -13.679798006972462},{'x': 13.679798006972446, 'y': -19.536776256292455},{'x': 6.172834225695115, 'y': -23.037330956994282},{'x': -2.0786644645316548, 'y': -23.759243549488129},
                 {'x': -10.079445542515689, 'y': -21.615440720824097},{'x': -16.864496731299162, 'y': -16.864496731299152},{'x': -21.615440720824104, 'y': -10.079445542515678},{'x': -23.759243549488133, 'y': -2.0786644645316432},{'x': -23.037330956994278, 'y': 6.1728342256951256},{'x': -19.536776256292452, 'y': 13.679798006972451},{'x': -13.679798006972447, 'y': 19.536776256292455},
                 {'x': -4.1152228171300749, 'y': 15.358220637996189},{'x': 4.1152228171300909, 'y': 15.358220637996183},{'x': 11.242997820866107, 'y': 11.242997820866107},{'x': 15.358220637996187, 'y': 4.1152228171300758},{'x': 15.358220637996183, 'y': -4.1152228171300909},{'x': 11.242997820866101, 'y': -11.242997820866112},{'x': 4.1152228171300766, 'y': -15.358220637996187},{'x': -4.1152228171300811, 'y': -15.358220637996187},{'x': -11.242997820866108, 'y': -11.242997820866101},{'x': -15.358220637996187, 'y': -4.1152228171300775},{'x': -15.358220637996185, 'y': 4.1152228171300838},{'x': -11.242997820866105, 'y': 11.242997820866108},{'x': -2.0576114085650374, 'y': 7.6791103189980943},{'x': 5.6214989104330533, 'y': 5.6214989104330533},{'x': 7.6791103189980916, 'y': -2.0576114085650454},{'x': 2.0576114085650383, 'y': -7.6791103189980934},{'x': -5.6214989104330542, 'y': -5.6214989104330506},{'x': -7.6791103189980925, 'y': 2.0576114085650419},{'x': 0.0, 'y': 0.0},{'x': -30.689924073888996, 'y': 32.282016051339312},{'x': -23.2988878137987, 'y': 35.343483510260029},{'x': -15.907851553708406, 'y': 38.404950969180746},{'x': -8.5168152936181123, 'y': 41.466418428101463},{'x': -1.1257790335278184, 'y': 44.52788588702218},{'x': -39.125169871739701, 'y': 21.2889897014281},
                 {'x': -31.734133611649405, 'y': 24.350457160348821},{'x': -24.343097351559109, 'y': 27.411924619269541},{'x': -16.952061091468817, 'y': 30.473392078190258},{'x': -9.5610248313785249, 'y': 33.534859537110975},{'x': -2.169988571288231, 'y': 36.596326996031692},{'x': 5.2210476888020647, 'y': 39.657794454952409},{'x': 12.612083948892357, 'y': 42.719261913873133},{'x': -40.16937940950011, 'y': 13.357430810437622},{'x': -32.778343149409821, 'y': 16.418898269358341},{'x': -25.387306889319525, 'y': 19.480365728279057},{'x': -17.99627062922923, 'y': 22.541833187199778},{'x': -10.605234369138937, 'y': 25.603300646120495},{'x': -3.2141981090486436, 'y': 28.664768105041212},{'x': 4.1768381510416503, 'y': 31.726235563961932},{'x': 11.567874411131942, 'y': 34.787703022882653},{'x': 18.958910671222238, 'y': 37.84917048180337},{'x': -41.213588947260526, 'y': 5.4258719194471414},{'x': -33.82255268717023, 'y': 8.4873393783678583},{'x': -26.431516427079938, 'y': 11.548806837288577},{'x': -19.040480166989642, 'y': 14.610274296209296},{'x': -11.64944390689935, 'y': 17.671741755130014},{'x': -4.2584076468090561, 'y': 20.733209214050731},{'x': 3.1326286132812369, 'y': 23.794676672971448},{'x': 10.523664873371533, 'y': 26.856144131892169},
                 {'x': 17.914701133461826, 'y': 29.917611590812889},{'x': 25.305737393552118, 'y': 32.979079049733606},{'x': -42.257798485020942, 'y': -2.5056869715433443},{'x': -34.866762224930646, 'y': 0.5557804873773744},{'x': -27.47572596484035, 'y': 3.6172479462980931},{'x': -20.084689704750055, 'y': 6.6787154052188109},
                 {'x': -12.693653444659763, 'y': 9.7401828641395287},{'x': -5.3026171845694687, 'y': 12.801650323060247},{'x': 2.0884190755208252, 'y': 15.863117781980966},{'x': 9.4794553356111191, 'y': 18.924585240901685},{'x': 16.870491595701409, 'y': 21.986052699822402},{'x': 24.261527855791705, 'y': 25.047520158743119},{'x': 31.652564115882001, 'y': 28.108987617663839},{'x': -43.302008022781351, 'y': -10.437245862533826},{'x': -35.910971762691055, 'y': -7.3757784036131078},{'x': -28.519935502600763, 'y': -4.3143109446923908},{'x': -21.128899242510471, 'y': -1.2528434857716721},{'x': -13.737862982420175, 'y': 1.8086239731490465},{'x': -6.3468267223298813, 'y': 4.8700914320697644},{'x': 1.0442095377604126, 'y': 7.9315588909904831},{'x': 8.4352457978507047, 'y': 10.993026349911201},{'x': 15.826282057941, 'y': 14.05449380883192},{'x': 23.217318318031293, 'y': 17.115961267752638},{'x': 30.608354578121585, 'y': 20.177428726673355},{'x': 37.99939083821188, 'y': 23.238896185594072},{'x': -36.955181300451471, 'y': -15.307337294603592},{'x': -29.564145040361176, 'y': -12.245869835682873},{'x': -22.17310878027088, 'y': -9.1844023767621543},{'x': -14.782072520180588, 'y': -6.1229349178414365},{'x': -7.3910362600902939, 'y': -3.0614674589207183},{'x': 0.0, 'y': 0.0},
                 {'x': 7.3910362600902939, 'y': 3.0614674589207183},{'x': 14.782072520180588, 'y': 6.1229349178414365},{'x': 22.17310878027088, 'y': 9.1844023767621543},{'x': 29.564145040361176, 'y': 12.245869835682873},{'x': 36.955181300451471, 'y': 15.307337294603592},{'x': -37.99939083821188, 'y': -23.238896185594072}, 
                 {'x': -30.608354578121585, 'y': -20.177428726673355},{'x': -23.217318318031293, 'y': -17.115961267752638},{'x': -15.826282057941, 'y': -14.05449380883192},{'x': -8.4352457978507047, 'y': -10.993026349911201},{'x': -1.0442095377604126, 'y': -7.9315588909904831},{'x': 6.3468267223298813, 'y': -4.8700914320697644},{'x': 13.737862982420175, 'y': -1.8086239731490465},{'x': 21.128899242510471, 'y': 1.2528434857716721},{'x': 28.519935502600763, 'y': 4.3143109446923908},{'x': 35.910971762691055, 'y': 7.3757784036131078},{'x': 43.302008022781351, 'y': 10.437245862533826},{'x': -31.652564115882001, 'y': -28.108987617663839},{'x': -24.261527855791705, 'y': -25.047520158743119},{'x': -16.870491595701409, 'y': -21.986052699822402},{'x': -9.4794553356111191, 'y': -18.924585240901685},{'x': -2.0884190755208252, 'y': -15.863117781980966},{'x': 5.3026171845694687, 'y': -12.801650323060247},{'x': 12.693653444659763, 'y': -9.7401828641395287},{'x': 20.084689704750055, 'y': -6.6787154052188109},{'x': 27.47572596484035, 'y': -3.6172479462980931},{'x': 34.866762224930646, 'y': -0.5557804873773744},{'x': 42.257798485020942, 'y': 2.5056869715433443},{'x': -25.305737393552118, 'y': -32.979079049733606},{'x': -17.914701133461826, 'y': -29.917611590812889},
                 {'x': -10.523664873371533, 'y': -26.856144131892169},{'x': -3.1326286132812369, 'y': -23.794676672971448},{'x': 4.2584076468090561, 'y': -20.733209214050731},{'x': 11.64944390689935, 'y': -17.671741755130014},{'x': 19.040480166989642, 'y': -14.610274296209296},{'x': 26.431516427079938, 'y': -11.548806837288577},{'x': 33.82255268717023, 'y': -8.4873393783678583},{'x': 41.213588947260526, 'y': -5.4258719194471414},{'x': -18.958910671222238, 'y': -37.84917048180337},{'x': -11.567874411131942, 'y': -34.787703022882653},{'x': -4.1768381510416503, 'y': -31.726235563961932},{'x': 3.2141981090486436, 'y': -28.664768105041212},{'x': 10.605234369138937, 'y': -25.603300646120495},{'x': 17.99627062922923, 'y': -22.541833187199778},{'x': 25.387306889319525, 'y': -19.480365728279057},{'x': 32.778343149409821, 'y': -16.418898269358341},{'x': 40.16937940950011, 'y': -13.357430810437622},{'x': -12.612083948892357, 'y': -42.719261913873133},{'x': -5.2210476888020647, 'y': -39.657794454952409},{'x': 2.169988571288231, 'y': -36.596326996031692},{'x': 9.5610248313785249, 'y': -33.534859537110975},{'x': 16.952061091468817, 'y': -30.473392078190258},{'x': 24.343097351559109, 'y': -27.411924619269541},{'x': 31.734133611649405, 'y': -24.350457160348821},
                 {'x': 39.125169871739701, 'y': -21.2889897014281},{'x': 1.1257790335278184, 'y': -44.52788588702218},{'x': 8.5168152936181123, 'y': -41.466418428101463},{'x': 15.907851553708406, 'y': -38.404950969180746},{'x': 23.2988878137987, 'y': -35.343483510260029},{'x': 30.689924073888996, 'y': -32.282016051339312}]
top_channels = list(range(0, 127))
bottom_channels = list(range(127, 247+1))
PMT_distance_top = 7.95  # cm
PMT_distance_bottom = 8.0  # cm
PMTOuterRingRadius = 3.875  # cm

# Values for the ScienceRuns
Excluded_PMTs_SR1 = [1, 12, 26, 34, 62, 65, 79, 86, 88, 102, 118, 130, 134, 135, 148, 150, 152, 162, 178, 183, 198, 206, 213, 214, 234, 239, 244, 51, 73, 137, 139, 27, 91, 167, 203] #https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:data_quality_pmt#list_of_pmts_excluded_from_the_first_science_result_shingo_oliver

# https://indico.in2p3.fr/event/9408/session/8/contribution/30/material/slides/0.pdf
# https://arxiv.org/pdf/1202.2628.pdf
# https://arxiv.org/pdf/1509.04055.pdf
# https://arxiv.org/pdf/1502.01000.pdf QE increase
PMT_CE = 0.90 # as reported by the PMT group
PMT_QE_Inc = 1.10 # increase at cryogenic temperatures

############################################################################################################
    
###############################  
######## SPE functions ########
###############################

def SPErough(data, n_channel_s = np.arange(0, 248, 1)):
    
    # 1. Identify the rough amplitude range corresponding to a single photoelectron
    # 2. Find the time window in which we have an excess of samples in this amplitude range.
    
    info = {'pmt': [ ], 'noise_mu': [], 'noise_sigma': [], 'noise_norm': [], 'LED_mu': [], 'LED_sigma': [], 'LED_norm': []}
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
        # L'estremo superiore del range è definito
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
        # temp è usato per trovare il segnale di 1PE
        # da usare come input nel fit. Viene preso il massimo 
        # dopo ~10 ADC
        #####################################################
        temp = hist[(idx_0PE_argmax+4):]
        idx_1PE_argmax = np.argmax(temp) + (idx_0PE_argmax+4)
        ######################
        # Range in cui fittare
        # il segnale di 1PE
        ######################
        low = 15 
        high = idx_1PE_argmax + 10
        if np.max(temp) > 1e1:
            try:
                init_1PE = [100, idx_1PE_argmax, 5]
                ###########
                #   FIT
                ###########
                popt_1PE, pcov_1PE = curve_fit(gaus, xbins_center[low:high], hist[low:high], sigma=np.sqrt(hist[low:high]),
                                       p0= init_1PE, maxfev=int(1e6))
            except:
                ##########
                # NO FIT
                ##########
                popt_1PE = [0,0,0]

        if np.max(temp) < 1e1:
            ##########
            # No 1PE
            ##########
            popt_1PE = [0,0,0]

        #######################
        # Per gioco fitto anche
        # il piedistallo
        #######################
        #hist_0PE = hist[:(idx_0PE_argmax+4)]
        #xbins_center_0PE = xbins_center[:(idx_0PE_argmax+4)]
        #try:
        #    init_0PE = [1000, idx_0PE_argmax, 1]
        #    popt_0PE, pcov_0PE = curve_fit(gaus, xbins_center_0PE, hist_0PE, sigma=np.sqrt(hist_0PE),
        #                                   p0= init_0PE, maxfev=int(1e6))
        #except:
        #    popt_0PE = [0,0,0]
            
        #########################################################
        # N : number of events to simulate
        # occ: lambda, aka occupancy, aka the mean of the poisson
        #########################################################
        #N = 50000 
        #occ = 0.2 
        
        #noise_norm = popt_0PE[0]
        #noise_mu = popt_0PE[1]
        #noise_sigma = np.abs(popt_0PE[2])
        #LED_norm = popt_1PE[0]
        #LED_mu = popt_1PE[1]
        #LED_sigma = np.abs(popt_1PE[2])

        ##########################################################################################
        # LED signal looks like noise (gaussian) + LED component (gaussian convolved with poisson)
        # noise signal is just the noise part with same mean, sigma
        # noise has mean 0, arbitrary sigma
        # LED has arbitrary mean, sigma. Just played with numbers til it looked about right
        ##########################################################################################
        #if (LED_mu > 0):
        #    LED_vals = (sp.stats.norm.rvs(loc=noise_mu, scale=noise_sigma, size=N) + 
        #                sp.stats.norm.rvs(loc=LED_mu, scale=LED_sigma, size=N) * sp.stats.poisson.rvs(occ, size=N))
        #    LED, bins = np.histogram(LED_vals, bins=400, range=(-100.5, 299.5))
        #    bins = 0.5 * (bins[1:] + bins[:-1])
        #noise_vals = sp.stats.norm.rvs(loc=noise_mu, scale=noise_sigma, size=N)
        #noise, bins = np.histogram(noise_vals, bins=400, range=(-100.5, 299.5))
        #bins = 0.5 * (bins[1:] + bins[:-1])

        N, mean, sig = popt_1PE[0], popt_1PE[1], popt_1PE[2]
        ############################
        # Salvo i parametri in un
        # dizionario. Non si sa mai
        ############################

        #info['pmt'].append(n_channel)
        #info['noise_mu'].append(noise_mu)
        #info['noise_sigma'].append(noise_sigma)
        #info['noise_norm'].append(noise_norm)
        #info['LED_mu'].append(LED_mu)
        #info['LED_sigma'].append(LED_sigma)
        #info['LED_norm'].append(LED_norm)

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

    median = df2['idx_LED'].median()
    std = df2['idx_LED'].std()
    window = [int(median-0.5*std),int(median+0.5*std)]
    #print('mean: ', mean, 'sigma: ', std, 'window LED: ', window)
    return window#, info, df2
    

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

        LED, bins_LED = np.histogram(arr['amplitudeLED'], bins=250, range=(0,500))
        bins_LED_center = 0.5 * (bins_LED[1:] + bins_LED[:-1])
        noise, bins_noise = np.histogram(arr['amplitudeNOISE'], bins=250, range=(0,500))
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
    

    