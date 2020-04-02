import strax
import straxen
import numpy as np

import spe_class
strax.mailbox.DEFAULT_TIMEOUT = 3600


common_opts = dict(
    register_all=[
        straxen.daqreader,
        straxen.pulse_processing,
        straxen.peaklet_processing,
        straxen.peak_processing,
        straxen.event_processing,
        straxen.cuts],
    store_run_fields=(
        'name', 'number',
        'reader.ini.name', 'tags.name',
        'start', 'end', 'livetime',
        'trigger.events_built'),
    check_available=('raw_records', 'records', 'peaklets',
                     'events', 'event_info'))

def strax_nofilter():
    return strax.Context(
        storage=[
            strax.DataDirectory(
                '/dali/lgrandi/aalbers/strax_data_raw',
                take_only='raw_records',
                deep_scan=False,
                readonly=True),
            strax.DataDirectory(
                '/dali/lgrandi/aalbers/strax_data',
                readonly=True,
                provide_run_metadata=False),
            strax.DataDirectory(
                '/dali/lgrandi/pgaemers/strax_data',
                readonly=False,
                provide_run_metadata=False),
#             strax.DataDirectory('',
#                                 provide_run_metadata=False)
        ],
        register=straxen.plugins.pax_interface.RecordsFromPax,
        # When asking for runs that don't exist, throw an error rather than
        # starting the pax converter
        forbid_creation_of=('raw_records',),
        **common_opts)

st = strax_nofilter()
st.set_config(dict(pmt_pulse_filter=None))

# runs = st.select_runs()
run = '170204_1410'

st.make(run, 'records')
st.make(run, 'peaks')