import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import butter, lfilter
from scipy.signal import freqs


def exponential_smooth(data, smooth_fac):
    """
    :param data(np.array)
    :param smooth_fac(int): span_interval
    :return:
    """
    ser = pd.Series(data)
    return ser.ewm(span=smooth_fac).mean()


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=True)
    return b, a


def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def draw_frequency_response(b, a, fs):
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--file', nargs=2, help='filename of the rppg and biopac data')
parser.add_argument('-se', '--smooth_exp', default=False, action='store_true', help='True if the rppg data is smoothed')
parser.add_argument('-sb', '--smooth_butter', default=False, action='store_true')
parser.add_argument('--smooth_factor', default=None, type=int, help='Smoooth Factor (span of ewm)')
parser.add_argument('--savefig', default=False, action='store_true', help="True to save figure")
args = parser.parse_args()

rppg_pulse = pd.read_csv(args.file[0])
biopac_pulse = pd.read_csv(args.file[1])
plt.plot(biopac_pulse['Time'] + 5, biopac_pulse['BPM'], label='biopac')
if args.smooth_exp:
    smoothed = exponential_smooth(rppg_pulse['BPM'], smooth_fac=args.smooth_factor)
    plt.plot(rppg_pulse['Time'], smoothed, label='rPPG_smoothed')
elif args.smooth_butter:
    dt = np.diff(rppg_pulse['Time'].values)
    average_dt = np.mean(dt)
    fs = 1 / average_dt
    print(average_dt)
    smoothed = butter_lowpass_filter(rppg_pulse['BPM'].values, cutOff=20, fs=fs, order=8)
    plt.plot(rppg_pulse['Time'], smoothed, label='rPPG_smoothed_butter')
else:
    plt.plot(rppg_pulse['Time'], rppg_pulse['BPM'], label='rPPG')

plt.ylim(20, 100)
plt.xlim(10, 80)
# plt.title('BPM After Exercise')
plt.xlabel('Time (s)')
plt.ylabel('BPM')
plt.legend()
if args.savefig:
    if args.smooth:
        plt.savefig('compare_{}_sm_{}.png'.format(args.file[0].split('_')[0], args.smooth_factor))
    else:
        plt.savefig('compare_{}_original.png'.format(args.file[0].split('_')[0]))
else:
    plt.show()
