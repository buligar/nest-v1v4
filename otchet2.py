import scipy
import locale
import nest
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity
from scipy.signal import savgol_filter


nest.ResetKernel()
nest.SetKernelStatus({
"local_num_threads": 1,
"resolution": 0.1,
"rng_seed": 1
})

# create inputs
ac1 = nest.Create("ac_generator", 1, params={
"amplitude": 1000,
"frequency": 40,
"phase": 0,
"start": 0,
"stop": 1000,
})
ac2 = nest.Create("ac_generator", 1, params={
"amplitude": 500,
"frequency": 12,
"offset": 0,
"phase": 0,
"start": 0,
"stop": 1000,
})
ng1 = nest.Create("noise_generator", 1, params={
"mean": 0,
"std": 450,
"start": 0,
"stop": 1000,
})
ng2 = nest.Create("noise_generator", 1, params={
"mean": 0,
"std": 300,
"start": 0,
"stop": 1000,
})

# create recorders
vm1 = nest.Create("voltmeter", 1)
vm2 = nest.Create("voltmeter", 1)
sr1 = nest.Create("spike_recorder", 1)
sr2 = nest.Create("spike_recorder", 1)
sr3 = nest.Create("spike_recorder", 1)
sr4 = nest.Create("spike_recorder", 1)

# create populations
eneurons = 40
ineurons = 10
n1 = nest.Create("iaf_psc_alpha", eneurons, params={
"tau_m": 6,
})
n2 = nest.Create("iaf_psc_alpha", ineurons, params={
"tau_m": 15,
})
n3 = nest.Create("iaf_psc_alpha", eneurons, params={
"tau_m": 30,
})
n4 = nest.Create("iaf_psc_alpha", ineurons, params={
"tau_m": 75,
})
n5 = nest.Create("iaf_psc_alpha", eneurons, params={
"tau_m": 6,
})
n6 = nest.Create("iaf_psc_alpha", ineurons, params={
"tau_m": 15,
})
n7 = nest.Create("iaf_psc_alpha", eneurons, params={
"tau_m": 30,
})
n8 = nest.Create("iaf_psc_alpha", ineurons, params={
"tau_m": 75,
})

# node connection
# локальное соединение надгранулярного уровня v1 
nest.Connect(n1, n2, syn_spec={
"weight": 3.5,
})
nest.Connect(n1, n1, syn_spec={
"weight": 1.5,
})
nest.Connect(n2, n2, syn_spec={
"weight": -2.5,
})
nest.Connect(n2, n1, syn_spec={
"weight": -3.25,
})
# локальное соединение подгранулярного уровня v1 
nest.Connect(n3, n3, syn_spec={
"weight": 1.5,
})
nest.Connect(n3, n4, syn_spec={
"weight": 3.5,
})
nest.Connect(n4, n4, syn_spec={
"weight": -2.5,
})
nest.Connect(n4, n3, syn_spec={
"weight": -3.25,
})

# контурное соединение v1 
nest.Connect(n1, n3, syn_spec={
"weight": 0.75,
})
nest.Connect(n3, n2, syn_spec={
"weight": 1,
})

# локальное соединение надгранулярного уровня v4
nest.Connect(n5, n5, syn_spec={
"weight": 1.5,
})
nest.Connect(n5, n6, syn_spec={
"weight": 3.5,
})
nest.Connect(n6, n6, syn_spec={
"weight": -2.5,
})
nest.Connect(n6, n5, syn_spec={
"weight": -3.25,
})
# локальное соединение подгранулярного уровня v4
nest.Connect(n7, n7, syn_spec={
"weight": 1.5,
})
nest.Connect(n7, n8, syn_spec={
"weight": 3.5,
})
nest.Connect(n8, n7, syn_spec={
"weight": -3.25,
})
nest.Connect(n8, n8, syn_spec={
"weight": -2.5,
})
# контурное соединение v4
nest.Connect(n5, n7, syn_spec={
"weight": 0.75,
})
nest.Connect(n7, n6, syn_spec={
"weight": 1,
})

# модульное соединение v1,v4
nest.Connect(n1, n5)
nest.Connect(n7, n1, syn_spec={
"weight": 0.1,
})
nest.Connect(n7, n2, syn_spec={
"weight": 0.5,
})
nest.Connect(n7, n3, syn_spec={
"weight": 0.95,
})
nest.Connect(n7, n4, syn_spec={
"weight": 0.5,
})

# connect inputs
nest.Connect(ac1, n1)
nest.Connect(ac2, n3)
nest.Connect(ac1, n5)
nest.Connect(ac2, n7)

nest.Connect(ng1, n1)
nest.Connect(ng2, n3)
nest.Connect(ng1, n5)
nest.Connect(ng2, n7)

# connect outputs
nest.Connect(vm1, n1)
nest.Connect(vm2, n3)
nest.Connect(n1, sr1)
nest.Connect(n3, sr2)

# run simulation
nest.Simulate(1000)

vm1 = np.array(vm1.events['V_m'])
vm2 = np.array(vm2.events['V_m'])

time = np.arange(1,999)
vm1all = vm1.reshape(999, eneurons)
vm2all = vm2.reshape(999, eneurons)


X = vm1[::eneurons]
Y = vm2[::eneurons]
resulting_array = np.column_stack((X, Y))
sampling_frequency = 1000
time = np.arange(0,999)
fs = 1000

# plot raw's, psd's, spectrogram's
plt.figure()
plt.subplot(231)
plt.plot(time, X, color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title('Raw (gamma)')

plt.subplot(232)
plt.title("PSD (gamma)")
(f, S) = scipy.signal.welch(vm1[::eneurons], fs)
plt.semilogy(f, S)
plt.xlim([0, 100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (mV**2/Hz)')
plt.legend(['vm1'])

plt.subplot(233)
plt.title("Spectrogram (gamma)")
f, t, Sxx = scipy.signal.spectrogram(vm1[::eneurons], fs)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.ylim([0, 100])
plt.xlabel('Time (s)')

plt.subplot(234)
plt.plot(time, Y, color="green")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title('Raw (alpha)')

plt.subplot(235)
plt.title("Power spectral density (alpha)")
(f, S) = scipy.signal.welch(vm2[::eneurons], fs)
plt.semilogy(f, S)
plt.xlim([0, 100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (mV**2/Hz)')
plt.legend(['vm2'])

plt.subplot(236)
plt.title("Spectrogram (alpha)")
f, t, Sxx = scipy.signal.spectrogram(vm2[::eneurons], fs)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.ylim([0, 100])
plt.xlabel('Time (s)')

plt.tight_layout()

# plot spike activity's
nest.raster_plot.from_device(sr1, hist=True)
plt.title('Population dynamics (gamma)')
nest.raster_plot.from_device(sr2, hist=True)
plt.title('Population dynamics (alpha)')

# compute connectivity
multitaper = Multitaper(
    resulting_array,
    sampling_frequency=sampling_frequency,
)
connectivity = Connectivity.from_multitaper(
    multitaper, expectation_type="time_trials_tapers",
)

# locale.setlocale(locale.LC_NUMERIC, 'ru_RU.UTF8')
matplotlib.rcParams['axes.formatter.use_locale'] = True

# plot csd, granger causality, coherence
fig, axis_handles = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

axis_handles[0].semilogy(
    connectivity.frequencies, connectivity.power()[..., 0],
    label='V1->V4 2/3', c='black'
)
axis_handles[0].semilogy(
    connectivity.frequencies, connectivity.power()[..., 1],
    label='V4->V1 5/6', c='black', linestyle='--'
)
axis_handles[0].set_ylabel('Cross spectral density\n (mV**2/Hz)', fontsize=14)
axis_handles[0].set_xlabel("Frequency (Hz)", fontsize=14)
axis_handles[0].set_title('Cross spectral density', fontsize=16)
axis_handles[0].set_xlim((0, 50))
axis_handles[0].legend(fontsize=12)

smoothed_y = savgol_filter(connectivity.pairwise_spectral_granger_prediction()[..., 1, 0].T, window_length=20, polyorder=2)
axis_handles[1].plot(
    connectivity.frequencies,
    connectivity.pairwise_spectral_granger_prediction()[..., 1, 0].T,
    label='V1->V4 2/3', c='black'
)
smoothed_y = savgol_filter(connectivity.pairwise_spectral_granger_prediction()[..., 0, 1].T, window_length=20, polyorder=2)
axis_handles[1].set_title("Pairwise Granger causality", fontsize=16)
axis_handles[1].plot(
    connectivity.frequencies,
    connectivity.pairwise_spectral_granger_prediction()[..., 0, 1].T,
    label='V4->V1 5/6', c='black', linestyle='--'
)

axis_handles[1].set_xlim((0, 50))
axis_handles[1].set_xlabel("Frequency (Hz)", fontsize=14)
axis_handles[1].legend(fontsize=12)

axis_handles[2].set_title("Partial directional coherence", fontsize=16)
axis_handles[2].plot(
    connectivity.frequencies,
    connectivity.partial_directed_coherence()[..., 1, 0].T,
    label='V1->V4 2/3', c='black'
)
axis_handles[2].plot(
    connectivity.frequencies,
    connectivity.partial_directed_coherence()[..., 0, 1].T,
    label='V4->V1 5/6', c='black', linestyle='--'
)
axis_handles[2].set_xlim((0, 50))
axis_handles[2].set_xlabel("Frequency (Hz)", fontsize=14)
axis_handles[2].legend(fontsize=12)
plt.show()