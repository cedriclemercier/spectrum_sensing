# currently the program to get the datasets
# check the dimension of the splitted vector
# for 16qam, the scaled transmit power is not exactly equal to what it should be
# check the formula for ht again, we have divided by math.sqrt(nTap)
# we have normalized the power = 1
# math.sqrt(80/64)

import numpy as np
import math
import random
from scipy.stats import poisson
from scipy.stats import skew
from scipy.stats import kurtosis


def bandwidth_count(a):   # returns the largest available bandwidth block size
    count = 0
    prev = 0
    for k in range(0, len(a)):
        if a[k] == -1:  # minus 1 means bandwidth is not used
            count += 1
        else:
            if count > prev:
                prev = count
            count = 0

    if count > prev:
        prev = count

    return prev


class user:     # class to set up all devices and their properties

    def __init__(self, uid, usage):
        self.userid = uid
        self.modulation = 0  # needs to be tested if the string is accepted correctly
        self.bandwidth = 0  # will be changed later on
        self.datarate = 0  # will be changed later on
        self.symbols = 0  # will be changed later on
        self.usage = usage  # will be changed later on
        self.calltime = 0  # will be allocated later on
        self.bandwidth_marker = -2  # if bandwidth marker = -2, the particular bandwidth is not used for transmission

    def setcallTime(self, cT):  # function to assign call time to the devices
        self.calltime = cT

    def deccallTime(self):  # function to decrease the call time
        self.calltime += -1

    def bandwidth_allocation(self, b_array):  # we pass bandwidth halow np array here
        check_bw = np.empty(self.bandwidth)
        check_bw.fill(-1)
        m = len(b_array) - len(check_bw) + 1
        index = []
        for position in range(m):
            if (b_array[position:position + self.bandwidth] == check_bw).all():
                index.append(position)
                break

        b_array[index[0]:index[0] + self.bandwidth] = self.userid
        self.bandwidth_marker = index[0]

    def bandwidth_release(self, bandwidth_array):  # to release the bandwidth once the call has ended
        bandwidth_array[self.bandwidth_marker: self.bandwidth_marker + self.bandwidth] = -1
        self.bandwidth_marker = -2


user_list = []

# classifying the devices as iot and assigning device id
for i in range(50000):
    t = 'IoT'
    user_i = user(i, t)
    user_list.append(user_i)

# assigning call time for iot as 1
#for q in user_list:
#    if q.usage == 'IoT':
#        q.setcallTime(1)

# assigning bandwidth, data rate and number of symbols for each type of devices
count_iot = 0
count_sph = 0  # pls ignore
for i in user_list:
    if i.usage == 'smartphone':
        count_sph += 1
        if count_sph % 4 == 0:
            i.bandwidth = 1
            i.datarate = 8000
            i.symbols = 38
            i.modulation = '16QAM'
        elif count_sph % 4 == 1:
            i.bandwidth = 4
            i.datarate = 8000
            i.symbols = 152
            i.modulation = 'BPSK'
        elif count_sph % 4 == 2:
            i.bandwidth = 8
            i.datarate = 64000
            i.symbols = 304
            i.modulation = '16QAM'
        else:
            i.bandwidth = 32
            i.datarate = 64000
            i.symbols = 1216
            i.modulation = 'BPSK'
    elif i.usage == 'IoT':
        count_iot += 1
        if count_iot % 6 == 0:
            i.bandwidth = 9  # NB IoT
            i.datarate = 72000
            i.symbols = 342
            i.modulation = '16QAM'
        elif count_iot % 6 == 1:
            i.bandwidth = 20  # EC GSM
            i.datarate = 160000
            i.symbols = 760
            i.modulation = '16QAM'
        elif count_iot % 6 == 2:  # NB IoT
            i.bandwidth = 36
            i.datarate = 72000
            i.symbols = 1368
            i.modulation = 'BPSK'
        elif count_iot % 6 == 3:
            i.bandwidth = 80  # EC GSM
            i.datarate = 160000
            i.symbols = 3040
            i.modulation = 'BPSK'
        elif count_iot % 6 == 4:  # LTE Cat1
            i.bandwidth = 125
            i.datarate = 1000000
            i.symbols = 4750
            i.modulation = '16QAM'
        else:
            i.bandwidth = 500  # LTE Cat1
            i.datarate = 1000000
            i.symbols = 19000
            i.modulation = 'BPSK'

# python lists to keep track of devices
freeUsers = user_list
proposedNewCallers = []
newCallers = []
rejectedCalls = []
busyUsers = []
callEnded = []

bandwidhtHalow = np.empty(26000)  # the actual bandwidth array used here; not related to the dataset;
bandwidhtHalow.fill(-1)  # the default value; otherwise carries the value of the user id using the bandwidth
wavelenght = 1/3  # checked
SNR = -5  # check how snr is used in the ofdm function
nSymPerB = 38  # number of symbols per 1kHz bandwidth
total_bandwidth = 0


def ofdm(mod, s, SNR, bandwidth, bm, userid):  # Adaptive OFDM modulation, number of symbols, SNR,
    """
    This function is to decide which signal to transmit among BPSK, 16-QAM
    """
    if mod == 'BPSK':
        nFFT = 64  # fft size
        nDSC = 52  # number of data subcarriers
        nBitPerSym = 52  # number of bits per OFDM symbol (same as the number of subcarriers for BPSK)
        nSym = s  # number of OFDM symbols to be transmitted

        # Transmitter
        ipBit = np.random.binomial(n=1, p=0.5, size=(nBitPerSym * nSym))
        ipMod = 2 * ipBit - 1  # BPSK modulation 0 --> -1, 1 --> +1
        ipSP = ipMod.reshape((nSym, nBitPerSym))  # nSym*52

        # Assigning modulated symbols to subcarriers from [-26 to - 1, +1 to + 26]
        idx_IN_columns1 = np.arange(0, nBitPerSym / 2, dtype=int)
        extractedData1 = ipSP[:, idx_IN_columns1]

        idx_IN_columns2 = np.arange(nBitPerSym / 2, nBitPerSym, dtype=int)
        extractedData2 = ipSP[:, idx_IN_columns2]

        xF1 = np.zeros([nSym, 6], dtype=int)
        xF2 = np.zeros([nSym, 5], dtype=int)
        xF3 = np.zeros([nSym, 1], dtype=int)

        OFDM_Data = np.concatenate((xF1, extractedData1, xF3, extractedData2, xF2), axis=1)  # 154*64

        # Taking FFT, the term (nFFT/sqrt(nDSC)) is for normalizing the power of transmit symbol to 1
        shift = np.fft.fftshift(OFDM_Data)
        inverse = np.fft.ifft(shift)
        OFDM_time = (nFFT / math.sqrt(nDSC)) * inverse

        # changing the transmitted power
        factor = pow(10, SNR/10)
        sq_root_factor = math.sqrt(factor)
        OFDM_time_boosted = OFDM_time * sq_root_factor

        # Appending cyclic prefix
        CP_ColumnNumbers = np.arange(48, 64, dtype=int)
        CP = OFDM_time_boosted[:, CP_ColumnNumbers]
        OFDM_withCP = np.concatenate((CP, OFDM_time_boosted), axis=1)  # 154*80

        #  Concatenating multiple symbols to form a long vector
        OFDM_RX = OFDM_withCP.reshape(1, nSym * 80)  # we are readying the signal to be divided into number of bandwidths

        # Changing dimension for suitability
        OFDM_RX_Dim = np.transpose(OFDM_RX)  # to get the array ready for squeeze
        OFDM_RX_Dim_Changed = np.squeeze(OFDM_RX_Dim)  # removing one dimension of the array

        # Splitting the received signal into same number of parts as the bandwidth allocated
        OFDM_RX_Splitted = np.array_split(OFDM_RX_Dim_Changed, bandwidth)
        OFDM_RX_Splitted = np.transpose(OFDM_RX_Splitted)  # these two transposes are needed to convert array into numpy array
        OFDM_RX_Splitted = np.transpose(OFDM_RX_Splitted)

        for w in range(bandwidth):
            # adding rayleigh fading: number of taps and the channel
            # channel model channel gain is 1 confirmed
            nTap = 4
            ht = (1 / math.sqrt(2)) * (1 / math.sqrt(nTap)) * (np.random.randn(nSymPerB, nTap) + 1j * np.random.randn(nSymPerB, nTap))

            # taking out the data for the first sub band
            signal = OFDM_RX_Splitted[w]  # shape is (3040,0)

            # splitting signal into symbols
            signal_split = np.array_split(signal, nSymPerB)
            signal_split = np.transpose(signal_split)
            signal_split = np.transpose(signal_split)  # shape of signal split is (38, 80)

            # convolution
            convl = np.zeros([nSymPerB, signal_split.shape[1] + nTap - 1], dtype=complex)  # shape is (38,83)
            for jj in range(0, nSymPerB):
                convl[jj, :] = np.convolve(ht[jj, :], signal_split[jj, :])

            # adding awgn
            nt00 = np.random.randn(1, convl.shape[0] * convl.shape[1]) + 1j * np.random.randn(1, convl.shape[0] * convl.shape[1])
            nt01 = (1 / math.sqrt(2)) * nt00
            convl_reshape = convl.reshape(1, convl.shape[0] * convl.shape[1])  # shape is (1, 3154)
            OFDM_RX_WithNoise = convl_reshape + nt01

            # discrete power calculation (delete after testing)
            OFDM_RX_WithNoise_abs = abs(OFDM_RX_WithNoise)
            OFDM_RX_WithNoise_abs_square = np.square(OFDM_RX_WithNoise_abs)
            power01 = np.sum(OFDM_RX_WithNoise_abs_square, axis=1, keepdims=True)
            power = power01 * (1 / 3154)

            # Calculating the real part and imaginary sum
            OFDM_RX_WithNoise_sum = np.sum(OFDM_RX_WithNoise, axis=1, keepdims=True)
            OFDM_RX_WithNoise_sum_Real = OFDM_RX_WithNoise_sum.real  # working fine, tested
            OFDM_RX_WithNoise_sum_Imag = OFDM_RX_WithNoise_sum.imag  # working fine, tested

            # calculating the variance of the real part
            OFDM_RX_WithNoise_real = OFDM_RX_WithNoise.real  # working fine, tested
            OFDM_RX_WithNoise_real_variance = np.var(OFDM_RX_WithNoise_real, axis=1, keepdims=True)  # working fine, tested

            # calculating the variance of the imaginary part
            OFDM_RX_WithNoise_imag = OFDM_RX_WithNoise.imag  # working fine, tested
            OFDM_RX_WithNoise_imag_variance = np.var(OFDM_RX_WithNoise_imag, axis=1, keepdims=True)  # working fine, tested

            # calculating the range of the real part
            OFDM_RX_WithNoise_real_range = np.ptp(OFDM_RX_WithNoise_real, axis=1, keepdims=True)

            # calculating the range of the imaginary part
            OFDM_RX_WithNoise_imag_range = np.ptp(OFDM_RX_WithNoise_imag, axis=1, keepdims=True)

            # preping for skewness
            OFDM_RX_WithNoise_real_transpose = np.transpose(OFDM_RX_WithNoise_real)
            OFDM_RX_WithNoise_real_transpose_sq = np.squeeze(OFDM_RX_WithNoise_real_transpose)

            OFDM_RX_WithNoise_imag_transpose = np.transpose(OFDM_RX_WithNoise_imag)
            OFDM_RX_WithNoise_imag_transpose_sq = np.squeeze(OFDM_RX_WithNoise_imag_transpose)

            # calculating the skewness of the real part
            skew_real = skew(OFDM_RX_WithNoise_real_transpose_sq)
            OFDM_RX_WithNoise_real_skew = np.full(shape=(1, 1), fill_value=skew_real)

            # calculating the skewness of the imag part
            skew_imag = skew(OFDM_RX_WithNoise_imag_transpose_sq)
            OFDM_RX_WithNoise_imag_skew = np.full(shape=(1, 1), fill_value=skew_imag)

            # calculating the kurtosis of the real part
            real_kurt = kurtosis(OFDM_RX_WithNoise_real, axis=1)
            OFDM_RX_WithNoise_real_kurt = np.full(shape=(1, 1), fill_value=real_kurt)

            # calculating the kurtosis of the imaginary part
            imag_kurt = kurtosis(OFDM_RX_WithNoise_imag, axis=1)
            OFDM_RX_WithNoise_imag_kurt = np.full(shape=(1, 1), fill_value=imag_kurt)

            # device id column
            id_array = np.full(shape=(1, 1), fill_value=1)

            # readying for dataset  # working fine, tested
            final_dataset = np.hstack((id_array, power, OFDM_RX_WithNoise_sum_Real, OFDM_RX_WithNoise_sum_Imag, OFDM_RX_WithNoise_real_variance, OFDM_RX_WithNoise_imag_variance, OFDM_RX_WithNoise_real_range, OFDM_RX_WithNoise_imag_range, OFDM_RX_WithNoise_real_skew, OFDM_RX_WithNoise_imag_skew, OFDM_RX_WithNoise_real_kurt, OFDM_RX_WithNoise_imag_kurt))

            # inserting into dataset matrix via marker
            dataset_this_second[bm + w,] = final_dataset  # working fine, tested

    elif mod == '16QAM':
        nFFT = 64  # fft size
        nDSC = 52  # number of data subcarriers
        mu = 4  # bits per symbol (i.e. 16QAM)
        nBitPerSym = nDSC * mu  # number of bits per OFDM symbol
        nSym = s  # number of symbols

        mapping_table = {
            (0, 0, 0, 0): (1 / math.sqrt(10)) * (-3 - 3j),
            (0, 0, 0, 1): (1 / math.sqrt(10)) * (-3 - 1j),
            (0, 0, 1, 0): (1 / math.sqrt(10)) * (-3 + 3j),
            (0, 0, 1, 1): (1 / math.sqrt(10)) * (-3 + 1j),
            (0, 1, 0, 0): (1 / math.sqrt(10)) * (-1 - 3j),
            (0, 1, 0, 1): (1 / math.sqrt(10)) * (-1 - 1j),
            (0, 1, 1, 0): (1 / math.sqrt(10)) * (-1 + 3j),
            (0, 1, 1, 1): (1 / math.sqrt(10)) * (-1 + 1j),
            (1, 0, 0, 0): (1 / math.sqrt(10)) * (3 - 3j),
            (1, 0, 0, 1): (1 / math.sqrt(10)) * (3 - 1j),
            (1, 0, 1, 0): (1 / math.sqrt(10)) * (3 + 3j),
            (1, 0, 1, 1): (1 / math.sqrt(10)) * (3 + 1j),
            (1, 1, 0, 0): (1 / math.sqrt(10)) * (1 - 3j),
            (1, 1, 0, 1): (1 / math.sqrt(10)) * (1 - 1j),
            (1, 1, 1, 0): (1 / math.sqrt(10)) * (1 + 3j),
            (1, 1, 1, 1): (1 / math.sqrt(10)) * (1 + 1j)
        }

        def Mapping(bits):
            return np.array([mapping_table[tuple(b)] for b in bits])

        # Transmitter
        ipBit = np.random.binomial(n=1, p=0.5, size=(nBitPerSym * nSym))
        ipMod0 = ipBit.reshape((nSym * nDSC, mu))
        ipMod = Mapping(ipMod0)
        ipSP = ipMod.reshape(nSym, nDSC)  # nSym*52

        # Assigning modulated symbols to subcarriers from [-26 to - 1, +1 to + 26]
        idx_IN_columns1 = np.arange(0, nDSC / 2, dtype=int)
        extractedData1 = ipSP[:, idx_IN_columns1]

        idx_IN_columns2 = np.arange(nDSC / 2, nDSC, dtype=int)
        extractedData2 = ipSP[:, idx_IN_columns2]

        xF1 = np.zeros([nSym, 6], dtype=int)
        xF2 = np.zeros([nSym, 5], dtype=int)
        xF3 = np.zeros([nSym, 1], dtype=int)

        OFDM_Data = np.concatenate((xF1, extractedData1, xF3, extractedData2, xF2), axis=1)

        # Taking FFT, the term (nFFT/sqrt(nDSC)) is for normalizing the power of transmit symbol to 1
        # conflict between the two sources regarding fftshift
        shift = np.fft.fftshift(OFDM_Data)
        inverse = np.fft.ifft(shift)
        OFDM_time = (nFFT / math.sqrt(nDSC)) * inverse

        # changing the transmitted power
        factor = pow(10, SNR/10)
        sq_root_factor = math.sqrt(factor)
        OFDM_time_boosted = OFDM_time * sq_root_factor

        # Appending cyclic prefix
        CP_ColumnNumbers = np.arange(48, 64, dtype=int)
        CP = OFDM_time_boosted[:, CP_ColumnNumbers]
        OFDM_withCP = np.concatenate((CP, OFDM_time_boosted), axis=1)

        #  Concatenating multiple symbols to form a long vector
        OFDM_RX = OFDM_withCP.reshape(1, nSym * 80)  # we are readying the signal to be divided into number of bandwidths

        # changing dimension for suitability
        OFDM_RX_Dim = np.transpose(OFDM_RX)  # to get the array ready for squeeze
        OFDM_RX_Dim_Changed = np.squeeze(OFDM_RX_Dim)  # removing one dimension of the array, needed for splitting

        # Splitting the received signal into same number of parts as the bandwidth allocated
        OFDM_RX_Splitted = np.array_split(OFDM_RX_Dim_Changed, bandwidth)  # returns a python array
        OFDM_RX_Splitted = np.transpose(OFDM_RX_Splitted)  # these two transposes are needed to convert array into numpy array
        OFDM_RX_Splitted = np.transpose(OFDM_RX_Splitted)

        for w in range(bandwidth):
            # adding rayleigh fading: number of taps and the channel
            # channel model: channel gain is 1 confirmed
            nTap = 4
            ht = (1 / math.sqrt(2)) * (1 / math.sqrt(nTap)) * (np.random.randn(nSymPerB, nTap) + 1j * np.random.randn(nSymPerB, nTap))

            # taking out the data for the first sub band
            signal = OFDM_RX_Splitted[w]  # shape is (3040,0)

            # splitting signal into symbols
            signal_split = np.array_split(signal, nSymPerB)
            signal_split = np.transpose(signal_split)
            signal_split = np.transpose(signal_split)  # shape of signal split is (38, 80)

            # convolution
            convl = np.zeros([nSymPerB, signal_split.shape[1] + nTap - 1], dtype=complex)  # shape is (38,83)
            for jj in range(0, nSymPerB):
                convl[jj, :] = np.convolve(ht[jj, :], signal_split[jj, :])

            # adding awgn
            nt00 = np.random.randn(1, convl.shape[0] * convl.shape[1]) + 1j * np.random.randn(1, convl.shape[0] * convl.shape[1])
            nt01 = (1 / math.sqrt(2)) * nt00
            # the variance after the first step is around 2, after the second step is around 1
            # the mean is around 0 before and after the second step
            # shape is (1, 3154)
            convl_reshape = convl.reshape(1, convl.shape[0] * convl.shape[1])  # shape is (1, 3154)
            OFDM_RX_WithNoise = convl_reshape + nt01

            # power
            OFDM_RX_WithNoise_abs = abs(OFDM_RX_WithNoise)
            OFDM_RX_WithNoise_abs_square = np.square(OFDM_RX_WithNoise_abs)
            power01 = np.sum(OFDM_RX_WithNoise_abs_square, axis=1, keepdims=True)
            power = power01 * (1 / 3154)

            # Calculating the real part and imaginary sum
            OFDM_RX_WithNoise_sum = np.sum(OFDM_RX_WithNoise, axis=1, keepdims=True)
            OFDM_RX_WithNoise_sum_Real = OFDM_RX_WithNoise_sum.real  # working fine, tested
            OFDM_RX_WithNoise_sum_Imag = OFDM_RX_WithNoise_sum.imag  # working fine, tested

            # calculating the variance of the real part
            OFDM_RX_WithNoise_real = OFDM_RX_WithNoise.real  # working fine, tested
            OFDM_RX_WithNoise_real_variance = np.var(OFDM_RX_WithNoise_real, axis=1, keepdims=True)  # working fine, tested

            # calculating the variance of the imaginary part
            OFDM_RX_WithNoise_imag = OFDM_RX_WithNoise.imag  # working fine, tested
            OFDM_RX_WithNoise_imag_variance = np.var(OFDM_RX_WithNoise_imag, axis=1, keepdims=True)  # working fine, tested

            # calculating the range of the real part
            OFDM_RX_WithNoise_real_range = np.ptp(OFDM_RX_WithNoise_real, axis=1, keepdims=True)

            # calculating the range of the imaginary part
            OFDM_RX_WithNoise_imag_range = np.ptp(OFDM_RX_WithNoise_imag, axis=1, keepdims=True)

            # preping for skewness
            OFDM_RX_WithNoise_real_transpose = np.transpose(OFDM_RX_WithNoise_real)
            OFDM_RX_WithNoise_real_transpose_sq = np.squeeze(OFDM_RX_WithNoise_real_transpose)

            OFDM_RX_WithNoise_imag_transpose = np.transpose(OFDM_RX_WithNoise_imag)
            OFDM_RX_WithNoise_imag_transpose_sq = np.squeeze(OFDM_RX_WithNoise_imag_transpose)

            # calculating the skewness of the real part
            skew_real = skew(OFDM_RX_WithNoise_real_transpose_sq)
            OFDM_RX_WithNoise_real_skew = np.full(shape=(1, 1), fill_value=skew_real)

            # calculating the skewness of the imag part
            skew_imag = skew(OFDM_RX_WithNoise_imag_transpose_sq)
            OFDM_RX_WithNoise_imag_skew = np.full(shape=(1, 1), fill_value=skew_imag)

            # calculating the kurtosis of the real part
            real_kurt = kurtosis(OFDM_RX_WithNoise_real, axis=1)
            OFDM_RX_WithNoise_real_kurt = np.full(shape=(1, 1), fill_value=real_kurt)

            # calculating the kurtosis of the imaginary part
            imag_kurt = kurtosis(OFDM_RX_WithNoise_imag, axis=1)
            OFDM_RX_WithNoise_imag_kurt = np.full(shape=(1, 1), fill_value=imag_kurt)

            # device id column
            id_array = np.full(shape=(1, 1), fill_value=1)

            # readying for dataset  # working fine, tested
            final_dataset = np.hstack((id_array, power, OFDM_RX_WithNoise_sum_Real, OFDM_RX_WithNoise_sum_Imag, OFDM_RX_WithNoise_real_variance, OFDM_RX_WithNoise_imag_variance, OFDM_RX_WithNoise_real_range, OFDM_RX_WithNoise_imag_range, OFDM_RX_WithNoise_real_skew, OFDM_RX_WithNoise_imag_skew, OFDM_RX_WithNoise_real_kurt, OFDM_RX_WithNoise_imag_kurt))

            # inserting into dataset matrix via marker
            dataset_this_second[bm + w,] = final_dataset  # working fine, tested


def awg_noise(p):  # in the subbands where there is no transmit signals, we only have awgn
    #  Gaussian noise of unit variance, 0 mean
    nt1 = np.random.randn(1, nSymPerB * 83) + 1j * np.random.randn(1, nSymPerB * 83)
    nt = (1 / math.sqrt(2)) * nt1  # shape is alright

    # noise power
    nt_abs = abs(nt)
    nt_abs_square = np.square(nt_abs)
    noise_power01 = np.sum(nt_abs_square, axis=1, keepdims=True)
    noise_power = noise_power01*(1/3154)

    # sum of real and imaginary parts
    nt_sum = np.sum(nt, axis=1, keepdims=True)
    nt_sum_real = nt_sum.real
    nt_sum_imag = nt_sum.imag

    # calculating the variance of the real part
    nt_real = nt.real
    nt_real_variance = np.var(nt_real, axis=1, keepdims=True)

    # calculating the variance of the imaginary part
    nt_imag = nt.imag
    nt_imag_variance = np.var(nt_imag, axis=1, keepdims=True)

    # calculating the range of the real part
    nt_real_range = np.ptp(nt_real, axis=1, keepdims=True)

    # calculating the range of the imaginary part
    nt_imag_range = np.ptp(nt_imag, axis=1, keepdims=True)

    # preping for skewness
    nt_real_transpose = np.transpose(nt_real)
    nt_real_transpose_sq = np.squeeze(nt_real_transpose)

    nt_imag_transpose = np.transpose(nt_imag)
    nt_imag_transpose_sq = np.squeeze(nt_imag_transpose)

    # calculating the skewness of the real part
    skew_real = skew(nt_real_transpose_sq)
    nt_real_skew = np.full(shape=(1, 1), fill_value=skew_real)

    # calculating the skewness of the real part
    skew_imag = skew(nt_imag_transpose_sq)
    nt_imag_skew = np.full(shape=(1, 1), fill_value=skew_imag)

    # calculating the kurtosis of the real part
    real_kurt = kurtosis(nt_real, axis=1)
    nt_real_kurt = np.full(shape=(1, 1), fill_value=real_kurt)

    # calculating the kurtosis of the imaginary part
    imag_kurt = kurtosis(nt_imag, axis=1)
    nt_imag_kurt = np.full(shape=(1, 1), fill_value=imag_kurt)

    # device id column
    id_array = np.full(shape=(1, 1), fill_value=00000)

    # hstacking them all
    noise_final = np.hstack((id_array, noise_power, nt_sum_real, nt_sum_imag, nt_real_variance, nt_imag_variance, nt_real_range, nt_imag_range, nt_real_skew, nt_imag_skew, nt_real_kurt, nt_imag_kurt))

    # dataset_this_second has already been defined as
    dataset_this_second[p: p + 1, ] = noise_final


dataset = np.zeros((1, 12))  # will be used to concatenate all datasets; we need to have an initial row of zero

for x in range(1, 13):  # 12 iterations are enough to generate 5k instances of each class

    freeBandwidth = bandwidth_count(bandwidhtHalow)  # just for validation purposes
    print(x)

    # no of new calls to be set up
    C = int(poisson.rvs(mu=7, size=1))  # poisson distribution

    if C <= 0 or C >= 50000:  # upper limit is the total number of devices
        pass

    else:
        proposedNewCallers = random.sample(list(freeUsers), C)  # users who wants new call set up
        minimumBandwidthRequested = min(y.bandwidth for y in proposedNewCallers)  # minimum bandwidth of the devices requesting connection

        if bandwidth_count(bandwidhtHalow) >= minimumBandwidthRequested:  # calls can be made as sufficient bandwidht is available
            for y in proposedNewCallers:
                if y.bandwidth <= bandwidth_count(bandwidhtHalow):
                    y.bandwidth_allocation(bandwidhtHalow)
                    newCallers.append(y)
                else:
                    rejectedCalls.append(y)
        else:
            rejectedCalls = [y for y in proposedNewCallers]  # all calls to be rejected as not enough bandwidth is available

        for y in newCallers:  # setting the time of each new caller
            random_time = int(poisson.rvs(mu=1, size=1))  # poisson distribution
            if random_time <= 0:
                random_time = 1

            y.setcallTime(random_time)

        busyUsers = busyUsers + newCallers  # adding the newly allowed callers to busy callers
        freeUsers = [i for i in freeUsers if i not in newCallers]  # removing the newcallers from free users

    bandwidth_used = 0  # calculating the bandwidth used in this second
    for y in busyUsers:
        bandwidth_used += y.bandwidth

    print('Bandwidth used in this second is {:6}'.format(bandwidth_used))
    total_bandwidth = total_bandwidth + bandwidth_used
    print(total_bandwidth)

    # here we are going to store the data for one second
    dataset_this_second = np.empty((26000, 12))
    dataset_this_second.fill(0)

    # here we are adding the signal values, the values will be added via the ofdm function
    for y in busyUsers:
        ofdm(y.modulation, y.symbols, SNR, y.bandwidth, y.bandwidth_marker, y.userid)

    # here we are adding the awgn only
    np.savetxt("bandwidthhalow.csv", bandwidhtHalow, delimiter=",")  # correct
    awgn = np.where(bandwidhtHalow == -1)
    awgn = awgn[0]
    awgn = list(awgn)  # correct

    for i in awgn:
        awg_noise(i)

    # here we are extracting the dataset for one sec and adding to the accumulator dataset
    np.savetxt("dataset_one_second.csv", dataset_this_second, delimiter=",")
    dataset = np.concatenate((dataset, dataset_this_second))

    for y in busyUsers:  # decreasing the call time for each buy users
        y.deccallTime()

    for y in busyUsers:  # removing the users whose call time has decreased to zero
        if y.calltime == 0:
            callEnded.insert(1, y)
            #  busyUsers.remove(y)

    if len(callEnded) != 0:
        freeUsers = freeUsers + callEnded

    for y in callEnded:
        y.bandwidth_release(bandwidhtHalow)

    freeBandwidth = bandwidth_count(bandwidhtHalow)

    busyUsers = [i for i in busyUsers if i not in callEnded]

    callEnded.clear()
    newCallers.clear()
    proposedNewCallers.clear()
    rejectedCalls.clear()

np.savetxt("ML_Project_SNR_Minus5.csv", dataset, delimiter=",")
