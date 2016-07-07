#!/usr/bin/env python
# encoding: utf-8
"""
Author: Che-Yuan Liang mister2dot4@gmail.com July. 2016

dependencies - python 2.7, numpy1.92, scipy15.1.
(optional: matplotlib1.43, mir_eval for plot figure)

OnsetDetectorCLR.py - onset detector based on Constrained Linear Reconstruction
(BPDN, BPDN+, NNLS, OLS)

usage - command line - “python OnsetDetectorCLR.py inputfile.wav <arguments>
        to check the arguments use -h
        i.e., to plot, parse -p
		
For more details of this program, please refer to:
“Musical Onset Detection Using Constrained Linear Reconstruction”, 
IEEE Signal Process. Lett., Vol. 22, Issue: 11, 2015.

This program includes SuperFlux. Software credit:
Copyright (c) 2012 - 2014 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

"""
Please note that SuperFlux released together with the paper

"Maximum Filter Vibrato Suppression for Onset Detection"
Sebastian Böck and Gerhard Widmer.
Proceedings of the 16th International Conference on Digital Audio Effects
(DAFx-13), Maynooth, Ireland, September 2013

is not tuned in any way for speed/memory efficiency. However, it can be used
as a reference implementation for the described onset detection with a maximum
filter for vibrato suppression.

It also serves as a reference implementation of the local group delay (LGD)
based weighting extension described in:

"Local group delay based vibrato and tremolo suppression for onset detection"
Sebastian Böck and Gerhard Widmer.
Proceedings of the 13th International Society for Music Information
Retrieval Conference (ISMIR), 2013.

If you use this software, please cite the corresponding paper.

Please send any comments, enhancements, errata, etc. to the main author.

"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fft
import scipy.io as sio
from scipy.io import wavfile
from scipy.ndimage.filters import (maximum_filter, maximum_filter1d,
                                   uniform_filter1d)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.optimize.nnls as nnls

# import spams  (deprecated)
# TODO: 
# replace spams.lasso to sklearn.lasso



class Filter(object):
    """
    Filter Class.

    """
    def __init__(self, num_fft_bins, fs, bands=24, fmin=30, fmax=17000, equal=False):
        """
        Creates a new Filter object instance.

        :param num_fft_bins: number of FFT coefficients
        :param fs:           sample rate of the audio file
        :param bands:        number of filter bands
        :param fmin:         the minimum frequency [Hz]
        :param fmax:         the maximum frequency [Hz]
        :param equal:        normalize the area of each band to 1

        """
        # sample rate
        self.fs = fs
        # reduce fmax if necessary
        if fmax > fs / 2:
            fmax = fs / 2
        # get a list of frequencies
        frequencies = self.frequencies(bands, fmin, fmax)
        # conversion factor for mapping of frequencies to spectrogram bins
        factor = (fs / 2.0) / num_fft_bins
        # map the frequencies to the spectrogram bins
        frequencies = np.round(np.asarray(frequencies) / factor).astype(int)
        # only keep unique bins
        frequencies = np.unique(frequencies)
        # filter out all frequencies outside the valid range
        frequencies = [f for f in frequencies if f < num_fft_bins]
        # number of bands
        bands = len(frequencies) - 2
        assert bands >= 3, 'cannot create filterbank with less than 3 ' \
                           'frequencies'
        # init the filter matrix with size: number of FFT bins x filter bands
        self.filterbank = np.zeros([num_fft_bins, bands], dtype=np.float)
        # process all bands
        for band in range(bands):
            # edge & center frequencies
            start, mid, stop = frequencies[band:band + 3]
            # create a triangular filter
            triangular_filter = self.triangular_filter(start, mid, stop, equal)
            self.filterbank[start:stop, band] = triangular_filter
        self.hz = np.multiply(factor,frequencies)[1:-1]
    @staticmethod
    def frequencies(bands, fmin, fmax, a=440):
        """
        Returns a list of frequencies aligned on a logarithmic scale.

        :param bands: number of filter bands per octave
        :param fmin:  the minimum frequency [Hz]
        :param fmax:  the maximum frequency [Hz]
        :param a:     frequency of A0 [Hz]
        :returns:     a list of frequencies

        Using 12 bands per octave and a=440 corresponding to the MIDI notes.

        """
        # factor 2 frequencies are apart
        factor = 2.0 ** (1.0 / bands)
        # start with A0
        freq = a
        frequencies = [freq]
        # go upwards till fmax
        while freq <= fmax:
            # multiply once more, since the included frequency is a frequency
            # which is only used as the right corner of a (triangular) filter
            freq *= factor
            frequencies.append(freq)
        # restart with a and go downwards till fmin
        freq = a
        while freq >= fmin:
            # divide once more, since the included frequency is a frequency
            # which is only used as the left corner of a (triangular) filter
            freq /= factor
            frequencies.append(freq)
        # sort frequencies
        frequencies.sort()
        # return the list
        return frequencies

    @staticmethod
    def triangular_filter(start, mid, stop, equal=False):
        """
        Calculates a triangular filter of the given size.

        :param start: start bin (with value 0, included in the filter)
        :param mid:   center bin (of height 1, unless norm is True)
        :param stop:  end bin (with value 0, not included in the filter)
        :param equal: normalize the area of the filter to 1
        :returns:     a triangular shaped filter

        """
        # height of the filter
        height = 1.
        # normalize the height
        if equal:
            height = 2. / (stop - start)
        # init the filter
        triangular_filter = np.empty(stop - start)
        # rising edge
        rising = np.linspace(0, height, (mid - start), endpoint=False)
        triangular_filter[:mid - start] = rising
        # falling edge
        falling = np.linspace(height, 0, (stop - mid), endpoint=False)
        triangular_filter[mid - start:] = falling
        # return
        return triangular_filter


class Wav(object):
    """
    Wav Class is a simple wrapper around scipy.io.wavfile.

    """
    def __init__(self, filename):
        """
        Creates a new Wav object instance of the given file.

        :param filename: name of the .wav file

        """
        # read in the audio
        self.sample_rate, self.audio = wavfile.read(filename)
        # set the length
        self.num_samples = np.shape(self.audio)[0]
        self.length = float(self.num_samples) / self.sample_rate
        # set the number of channels
        try:
            # multi channel files
            self.num_channels = np.shape(self.audio)[1]
        except IndexError:
            # catch mono files
            self.num_channels = 1

    def attenuate(self, attenuation):
        """
        Attenuate the audio signal.

        :param attenuation: attenuation level given in dB

        """
        att = np.power(np.sqrt(10.), attenuation / 10.)
        self.audio = np.asarray(self.audio / att, dtype=self.audio.dtype)

    def downmix(self):
        """
        Down-mix the audio signal to mono.

        """
        if self.num_channels > 1:
            self.audio = np.mean(self.audio, axis=-1, dtype=self.audio.dtype)

    def normalize(self):
        """
        Normalize the audio signal.

        """
        self.audio = self.audio.astype(np.float) / np.max(self.audio)


class Spectrogram(object):
    """
    Spectrogram Class.

    """
    def __init__(self, wav, frame_size=2048, fps=200, filterbank=None,
                 log=False, mul=1, add=1, online=True, block_size=2048,
                 lgd=False, cd=False):
        """
        Creates a new Spectrogram object instance and performs a STFT on the
        given audio.

        :param wav:        a Wav object
        :param frame_size: the size for the window [samples]
        :param fps:        frames per second
        :param filterbank: use the given filterbank for dimensionality
                           reduction
        :param log:        use logarithmic magnitude
        :param mul:        multiply the magnitude by this factor before taking
                           the logarithm
        :param add:        add this value to the magnitude before taking the
                           logarithm
        :param online:     work in online mode (i.e. use only past information)
        :param block_size: perform the filtering in blocks of the given size
        :param lgd:        compute the local group delay (needed for the
                           ComplexFlux algorithm)

        """
        # init some variables
        self.wav = wav
        self.fps = fps
        self.filterbank = filterbank
        if add <= 0:
            raise ValueError("a positive value must be added before taking "
                             "the logarithm")
        if mul <= 0:
            raise ValueError("a positive value must be multiplied before "
                             "taking the logarithm")
        # derive some variables
        # use floats so that seeking works properly
        self.hop_size = float(self.wav.sample_rate) / float(self.fps)
        self.num_frames = int(np.ceil(self.wav.num_samples / self.hop_size))
        self.num_fft_bins = int(frame_size / 2)
        # initial number of bins equal to fft bins, but those can change if
        # filters are used
        self.num_bins = int(frame_size / 2)
        # init spec matrix
        if filterbank is None:
            # init with number of FFT frequency bins
            self.spec = np.empty([self.num_frames, self.num_fft_bins],
                                 dtype=np.float32)
        else:
            # init with number of filter bands
            self.spec = np.empty([self.num_frames, np.shape(filterbank)[1]],
                                 dtype=np.float32)
            # set number of bins
            self.num_bins = np.shape(filterbank)[1]
            # set the block size
            if not block_size or block_size > self.num_frames:
                block_size = self.num_frames
            # init block counter
            block = 0
            # init a matrix of that size
            spec = np.zeros([block_size, self.num_fft_bins])
        # init the local group delay matrix
        self.lgd = None
        self.cd = None
        if lgd:
            self.lgd = np.zeros([self.num_frames, self.num_fft_bins],
                                dtype=np.float32)
        if cd:    
            self.cd = np.zeros([self.num_frames, self.num_fft_bins],
                                dtype=np.float32)
            self.dcd = np.zeros([self.num_frames, self.num_fft_bins],
                                dtype=np.float32)
        # create windowing function for DFT
        self.window = np.hanning(frame_size)
        try:
            # the audio signal is not scaled, scale the window accordingly
            max_value = np.iinfo(self.wav.audio.dtype).max
            self._fft_window = self.window / max_value
        except ValueError:
            self._fft_window = self.window
        # step through all frames
        for frame in range(self.num_frames):
            # seek to the right position in the audio signal
            if online:
                # step back one frame_size after moving forward 1 hop_size
                # so that the current position is at the end of the window
                seek = int((frame + 1) * self.hop_size - frame_size)
            else:
                # step back half of the frame_size so that the frame represents
                # the centre of the window
                seek = int(frame * self.hop_size - frame_size / 2)
            # read in the right portion of the audio
            if seek >= self.wav.num_samples:
                # end of file reached
                break
            elif seek + frame_size >= self.wav.num_samples:
                # end behind the actual audio, append zeros accordingly
                zeros = np.zeros(seek + frame_size - self.wav.num_samples)
                signal = self.wav.audio[seek:]
                signal = np.append(signal, zeros)
            elif seek < 0:
                # start before the actual audio, pad with zeros accordingly
                zeros = np.zeros(-seek)
                signal = self.wav.audio[0:seek + frame_size]
                signal = np.append(zeros, signal)
            else:
                # normal read operation
                signal = self.wav.audio[seek:seek + frame_size]
            # multiply the signal with the window function
            signal = signal * self._fft_window
            # perform DFT
            stft = fft.fft(signal)[:self.num_fft_bins]
            # compute the local group delay
            if lgd:
                # unwrap the phase
                unwrapped_phase = np.unwrap(np.angle(stft))
                # local group delay is the derivative over frequency
                self.lgd[frame, :-1] = (unwrapped_phase[:-1] -
                                        unwrapped_phase[1:])
            # compute the phase matrix
            if cd: 
                self.cd[frame] = np.unwrap(np.angle(stft))
            # is block-wise processing needed?
            if filterbank is None:
                # no filtering needed, thus no block wise processing needed
                self.spec[frame] = np.abs(stft)
            else:
                # filter in blocks
                spec[frame % block_size] = np.abs(stft)
                # end of a block or end of the signal reached
                end_of_block = (frame + 1) / block_size > block
                end_of_signal = (frame + 1) == self.num_frames
                if end_of_block or end_of_signal:
                    start = block * block_size
                    stop = min(start + block_size, self.num_frames)
                    filtered_spec = np.dot(spec[:stop - start], filterbank)
                    self.spec[start:stop] = filtered_spec
                    # increase the block counter
                    block += 1
            # next frame
        # commpute first derivative of phase along time axis
        if cd:
            self.dcd[1:] = self.cd[1:] - self.cd[:-1]
        # take the logarithm
        if log:
            np.log10(mul * self.spec + add, out=self.spec)


class SpectralODF(object):
    """
    The SpectralODF class implements most of the common onset detection
    function based on the magnitude or phase information of a spectrogram.

    """
    def __init__(self, spectrogram, ratio=0.5, max_bins=3, diff_frames=None,
                 temporal_filter=3, temporal_origin=0, lam=1, predict_frames=1, pos=False, var1=0.5, var2=0.5):
        """
        Creates a new ODF object instance.

        :param spectrogram:     a Spectrogram object on which the detection
                                functions operate
        :param ratio:           calculate the difference to the frame which
                                has the given magnitude ratio
        :param max_bins:        number of bins for the maximum filter
        :param diff_frames:     calculate the difference to the N-th previous
                                frame
        :param temporal_filter: temporal maximum filtering of the local group
                                delay for the ComplexFlux algorithms
        :param temporal_origin: origin of the temporal maximum filter

        If no diff_frames are given, they are calculated automatically based on
        the given ratio.
        
        Linear Resconstruction arguments
        :param lam              constraint of regularization term
        :param pos              positive constraint (type=bool)
        :param predict_frames   reconstraction length (\tau)
        :param var1, var2       reserved arguemnts for experimental test

        """
        self.s = spectrogram
        # determine the number off diff frames
        if diff_frames is None:
            # get the first sample with a higher magnitude than given ratio
            sample = np.argmax(self.s.window > ratio)
            diff_samples = self.s.window.size / 2 - sample
            # convert to frames
            diff_frames = int(round(diff_samples / self.s.hop_size))
            # set the minimum to 1
            if diff_frames < 1:
                diff_frames = 1
        self.diff_frames = diff_frames
        # number of bins used for the maximum filter
        self.max_bins = max_bins
        self.temporal_filter = temporal_filter
        self.temporal_origin = temporal_origin
        self.predict_frames = predict_frames
        self.lam = lam
        self.pos = pos
        self.var1 = var1
        self.var2 = var2 

    @staticmethod
    def _superflux_diff_spec(spec, diff_frames=1, max_bins=3):
        """
        Calculate the difference spec used for SuperFlux.

        :param spec:        magnitude spectrogram
        :param diff_frames: calculate the difference to the N-th previous frame
        :param max_bins:    number of neighboring bins used for maximum
                            filtering
        :return:            difference spectrogram used for SuperFlux

        Note: If 'max_bins' is greater than 0, a maximum filter of this size
              is applied in the frequency direction. The difference of the
              k-th frequency bin of the magnitude spectrogram is then
              calculated relative to the maximum over m bins of the N-th
              previous frame (e.g. m=3: k-1, k, k+1).

              This method works only properly if the number of bands for the
              filterbank is chosen carefully. A values of 24 (i.e. quarter-tone
              resolution) usually yields good results.

        """
        # init diff matrix
        diff_spec = np.zeros_like(spec)
        if diff_frames < 1:
            raise ValueError("number of diff_frames must be >= 1")
        # widen the spectrogram in frequency dimension by `max_bins`
        max_spec = maximum_filter(spec, size=[1, max_bins])
        # calculate the diff
        diff_spec[diff_frames:] = spec[diff_frames:] - max_spec[:-diff_frames]
        # keep only positive values
        np.maximum(diff_spec, 0, diff_spec)
        # return diff spec
        return diff_spec

    @staticmethod
    def _rectification_mask(spec, diff_frames, max_bins):
        """
        Calculate the difference and create binary mask of negative flux

        :param spec:        magnitude spectrogram
        :param diff_frames: calculate the difference to the N-th previous frame
        :param max_bins:    number of neighboring bins used for maximum
                            filtering
        :return:            binary mask obtained from SuperFlux

        """
        # initialize diff_spec
        diff_spec = np.zeros_like(spec)
        # applying maximum filter
        max_spec = maximum_filter(spec, size=[1, max_bins])
        # calculate the diff to the N-th previous frame
        diff_spec[diff_frames:] = spec[diff_frames:] - max_spec[:-diff_frames]
        #### this step might not be indicated clearly in SPL

        # create offset binary mask
        offset_mask = diff_spec > 0
        # return offset_mask
        return offset_mask

    @staticmethod
    def _lgd_mask(spec, lgd, filterbank=None, temporal_filter=0,
                  temporal_origin=0):
        """
        Calculates a weighting mask for the magnitude spectrogram based on the
        local group delay.

        :param spec:            the magnitude spectrogram
        :param lgd:             local group delay of the spectrogram
        :param filterbank:      filterbank used for dimensionality reduction of
                                the magnitude spectrogram
        :param temporal_filter: temporal maximum filtering of the local group
                                delay
        :param temporal_origin: origin of the temporal maximum filter

        "Local group delay based vibrato and tremolo suppression for onset
         detection"
        Sebastian Böck and Gerhard Widmer.
        Proceedings of the 13th International Society for Music Information
        Retrieval Conference (ISMIR), 2013.

        """
        from scipy.ndimage import maximum_filter, minimum_filter
        # take only absolute values of the local group delay
        lgd = np.abs(lgd)

        # maximum filter along the temporal axis
        if temporal_filter > 0:
            lgd = maximum_filter(lgd, size=[temporal_filter, 1],
                                 origin=temporal_origin)
        # lgd = uniform_filter(lgd, size=[1, 3])  # better for percussive onsets

        # create the weighting mask
        if filterbank is not None:
            # if the magnitude spectrogram was filtered, use the minimum local
            # group delay value of each filterbank (expanded by one frequency
            # bin in both directions) as the mask
            mask = np.zeros_like(spec)
            num_bins = lgd.shape[1]
            for b in range(mask.shape[1]):
                # determine the corner bins for the mask
                corner_bins = np.nonzero(filterbank[:, b])[0]
                # always expand to the next neighbour
                start_bin = corner_bins[0] - 1
                stop_bin = corner_bins[-1] + 2
                # constrain the range
                if start_bin < 0:
                    start_bin = 0
                if stop_bin > num_bins:
                    stop_bin = num_bins
                # set mask
                mask[:, b] = np.amin(lgd[:, start_bin: stop_bin], axis=1)
        else:
            # if the spectrogram is not filtered, use a simple minimum filter
            # covering only the current bin and its neighbours
            mask = minimum_filter(lgd, size=[1, 3])
        # return the normalized mask
        return mask / np.pi
    
    # TODO: replace spams to scipy

    # @staticmethod 
    # def _linear_reconstruction_bdpn(spec, diff_frames, lam, predict_frames, max_bins, pos):
    #     """
    #     Calculates the residual spectromgram of Lasso approximation

    #     return: Values(type: dictionary) contains of alpha_matrix, residual_spec, approximate_spec

    #     """
    #     if predict_frames < 1:
    #         raise ValueError("number of predict_frames must be >= 1")
    #     # maximum filter by SB
    #     spec = maximum_filter(spec, size=[1, max_bins]) 
    #     # init residual spec
    #     m,n = np.shape(spec)
    #     residual_spec = np.zeros_like(spec)
    #     approximate_spec = np.zeros_like(spec)
    #     alpha_matrix = np.zeros((m,predict_frames))
    #     start_frame = predict_frames + diff_frames - 1
    #     # padding dictionary
    #     D = np.concatenate((np.tile(spec[0], (predict_frames + diff_frames, 1)), spec), axis=0)
    #     # normalization dictionary
    #     Norm = np.transpose(np.tile(np.sqrt(np.sum((D * D), axis=1)), (np.shape(D)[1],1)))
    #     # re-assign Norm if Norm = 0, prevent overflow
    #     index = np.where(Norm == 0)[0]
    #     Norm[index] = 1E-10
    #     Norm_D = np.asfortranarray(D / Norm, dtype=float)
    #     # approximate the spectrogram
    #     for i in range(m):
    #         Xi = np.asfortranarray(Norm_D[i + predict_frames + diff_frames][np.newaxis].T)
    #         Di = np.asfortranarray(Norm_D[range(i, i + predict_frames)].T)
    #         alpha = spams.lasso(Xi, D = Di, pos = pos, lambda1 = lam, lambda2 = 0,
    #                             mode = 2, cholesky = False, ols = False, verbose = False)
    #         alpha_matrix[i] = alpha.todense().T
    #         approximate_spec[i] = np.dot(Di, alpha.todense()).T
    #     # prevent overflow
    #     approximate_spec[approximate_spec >= 1] = 0    
    #     # calculate the residual spectrogram
    #     residual_spec = Norm_D[range(start_frame, start_frame + m)] - approximate_spec
    #     # store results into the dictionary     
    #     Values = {'spec': spec, 'residual_spec': residual_spec, 'alpha_matrix': alpha_matrix,
    #               'approximate_spec': approximate_spec,
    #               'Norm_D': Norm_D[range(start_frame, start_frame + m)]}     
    #     return Values
    
    @staticmethod 
    def _linear_reconstruction_nnls(spec, diff_frames, lam, predict_frames, max_bins):
        """
        Calculates the residual spectromgram of Non-negative Least Square approximation

        return: Values of alpha_matrix, residual_spec, approximate_spec

        """
        if predict_frames < 1:
            raise ValueError("number of predict_frames must be >= 1")
        # maximum filter by SB
        spec = maximum_filter(spec, size=[1, max_bins]) 
        # init residual spec
        m,n = np.shape(spec)
        residual_spec = np.zeros_like(spec)
        approximate_spec = np.zeros_like(spec)
        alpha_matrix = np.zeros((m,predict_frames))
        start_frame = predict_frames + diff_frames - 1
        # padding dictionary
        D = np.concatenate((np.tile(spec[0], (predict_frames + diff_frames, 1)), spec), axis=0)
        # normalization dictionary
        Norm = np.transpose(np.tile(np.sqrt(np.sum((D * D), axis=1)), (np.shape(D)[1],1)))
        # re-assign Norm if Norm = 0, prevent overflow
        index = np.where(Norm == 0)[0]
        Norm[index] = 1E-10
        Norm_D = np.asfortranarray(D / Norm, dtype=float)
        # approximate the spectrogram
        for i in range(m):
            Xi = np.asfortranarray(Norm_D[i + predict_frames + diff_frames])
            Di = np.asfortranarray(Norm_D[range(i, i + predict_frames)].T)
            alpha = nnls(Di, Xi.T)[0]
            alpha_matrix[i] = alpha.T
            approximate_spec[i] = np.dot(Di, alpha).T
        # prevent overflow
        approximate_spec[approximate_spec >= 1] = 0    
        # calculate the residual spectrogram
        residual_spec = Norm_D[range(start_frame, start_frame + m)] - approximate_spec 
        # store results into the dictionary  
        Values = {'spec': spec, 'residual_spec': residual_spec, 'alpha_matrix': alpha_matrix,
                  'approximate_spec': approximate_spec,
                  'Norm_D': Norm_D[range(start_frame, start_frame + m)]}     
        return Values

    @staticmethod 
    def _linear_reconstruction_ols(spec, diff_frames, lam, predict_frames, max_bins):
        """
        Calculates the residual spectromgram of Ordinary Least Square reconstruction
        
        return: Values of alpha_matrix, residual_spec, approximate_spec

        """
        if predict_frames < 1:
            raise ValueError("number of predict_frames must be >= 1")
        # maximum filter by SB
        spec = maximum_filter(spec, size=[1, max_bins]) 
        # init residual spec
        m,n = np.shape(spec)
        residual_spec = np.zeros_like(spec)
        approximate_spec = np.zeros_like(spec)
        alpha_matrix = np.zeros((m,predict_frames))
        start_frame = predict_frames + diff_frames - 1
        # padding dictionary
        D = np.concatenate((np.tile(spec[0], (predict_frames + diff_frames, 1)), spec), axis=0)
        # normalization dictionary
        Norm = np.transpose(np.tile(np.sqrt(np.sum((D * D), axis=1)), (np.shape(D)[1],1)))
        # re-assign Norm if Norm = 0, prevent overflow
        index = np.where(Norm == 0)[0]
        Norm[index] = 1E-10
        Norm_D = np.asfortranarray(D / Norm, dtype=float)
        # approximate the spectrogram
        for i in range(m):
            Xi = np.asfortranarray(Norm_D[i + predict_frames + diff_frames])
            Di = np.asfortranarray(Norm_D[range(i, i + predict_frames)].T)
            alpha = np.linalg.lstsq(Di, Xi)[0]
            alpha_matrix[i] = alpha.T
            approximate_spec[i] = np.dot(Di, alpha).T
        # prevent overflow
        approximate_spec[approximate_spec >= 1] = 0    
        # calculate the residual spectrogram
        residual_spec = Norm_D[range(start_frame, start_frame + m)] - approximate_spec 
        # store results into the dictionary  
        Values = {'spec': spec, 'residual_spec': residual_spec, 'alpha_matrix': alpha_matrix,
                  'approximate_spec': approximate_spec,
                  'Norm_D': Norm_D[range(start_frame, start_frame + m)]}     
        return Values

    # Onset Detection Functions
    def superflux(self):
        """
        SuperFlux with a maximum filter based vibrato suppression.

        :return: SuperFlux onset detection function

        "Maximum Filter Vibrato Suppression for Onset Detection"
        Sebastian Böck and Gerhard Widmer.
        Proceedings of the 16th International Conference on Digital Audio
        Effects (DAFx-13), Maynooth, Ireland, September 2013

        """
        # compute the difference spectrogram as in the SuperFlux algorithm
        diff_spec = self._superflux_diff_spec(self.s.spec, self.diff_frames,
                                              self.max_bins)
        # sum all positive 1st order max. filtered differences
        return np.sum(diff_spec, axis=1)

    def complex_domain(self):
        """
        implement of Simon Dixon DAFx2006 onset detection revised
        
        """
        CD = np.zeros_like(self.s.spec)
        X = np.multiply(self.s.spec[1:], np.exp(1.0j * self.s.cd[1:]))
        Xt = np.multiply(self.s.spec[:-1], np.exp(1.0j * (self.s.cd[:-1] + self.s.dcd[:-1])))
        CD[1:] = np.abs(np.subtract(X, Xt))

        return np.sum(CD, axis=1)

    def complex_flux(self):
        """
        Complex Flux with a local group delay based tremolo suppression.

        Calculates the difference of bin k of the magnitude spectrogram
        relative to the N-th previous frame of the (maximum filtered)
        spectrogram.

        :return: complex flux onset detection function

        "Local group delay based vibrato and tremolo suppression for onset
         detection"
        Sebastian Böck and Gerhard Widmer.
        Proceedings of the 13th International Society for Music Information
        Retrieval Conference (ISMIR), 2013.

        """
        # compute the difference spectrogram as in the SuperFlux algorithm
        diff_spec = self._superflux_diff_spec(self.s.spec, self.diff_frames,
                                              self.max_bins)
        # create a mask based on the local group delay information
        mask = self._lgd_mask(self.s.spec, self.s.lgd, self.s.filterbank,
                              self.temporal_filter, self.temporal_origin)
        # weight the differences with the mask
        diff_spec *= mask
        # sum all positive 1st order max. filtered and weighted differences
        return np.sum(diff_spec, axis=1)
    
    def linear_reconstruction_bdpn(self):
        """
        Linear reconstruction by Lasso with feature of SuperFlux

        :return 
            odf: onset detection function obtained from residual spec
            odf_offset: residual_spec with offset mask
            odf_denormalization: odf with amplitude modulation
            odf_offset_denormalization : odf with rectification and denormalization


        """
        Values = self._linear_reconstruction_bdpn(self.s.spec, self.diff_frames, self.lam,
                                                 self.predict_frames, self.max_bins, self.pos)
        # spectrogram
        spec = self.s.spec
        # offset mask
        offset_mask = self._rectification_mask(self.s.spec, self.diff_frames, self.max_bins)
        # de-normalization factor
        Values['denormalization'] = np.sqrt(np.sum(spec ** 2, axis = 1))
        # calculate odf with different masks
        Values['odf'] = np.sqrt(np.sum(Values['residual_spec'] ** 2, axis = 1))
        Values['odf_rectification'] = np.sqrt(np.sum((Values['residual_spec'] * offset_mask) ** 2, axis = 1))      
        Values['odf_denormalization'] = Values['odf'] * Values['denormalization']
        Values['odf_offset_denormalization'] = np.sqrt(np.sum((Values['residual_spec'] * offset_mask) ** 2, axis = 1)) * Values['denormalization']
        return Values

    def linear_reconstruction_nnls(self):
        """
        Linear reconstruction by NNLS with feature of SuperFlux

        :return 
            odf: onset detection function obtain from residual spec
            odf_offset: residual_spec with offset mask
            odf_denormalization: odf with amplitude modulation
            odf_offset_denormalization : odf with rectification and denormalization
        """
        Values = self._linear_reconstruction_nnls(self.s.spec, self.diff_frames, self.lam,
                                                 self.predict_frames, self.max_bins)
        # spectrogram
        spec = self.s.spec
        # offset mask
        offset_mask = self._rectification_mask(self.s.spec, self.diff_frames, self.max_bins)
        # de-normalization factor
        Values['denormalization'] = np.sqrt(np.sum(spec ** 2, axis = 1))
        # calculate odf with different masks
        Values['odf'] = np.sqrt(np.sum(Values['residual_spec'] ** 2, axis = 1))
        Values['odf_rectification'] = np.sqrt(np.sum((Values['residual_spec'] * offset_mask) ** 2, axis = 1))      
        Values['odf_denormalization'] = Values['odf'] * Values['denormalization']
        Values['odf_offset_denormalization'] = np.sqrt(np.sum((Values['residual_spec'] * offset_mask) ** 2, axis = 1)) * Values['denormalization']
        return Values

    def linear_reconstruction_ols(self):
        """
        Linear reconstruction by OLS with feature of SuperFlux

        :return 
            odf: onset detection function obtain from residual spec
            odf_offset: residual_spec with offset mask
            odf_denormalization: odf with amplitude modulation
            odf_offset_denormalization : odf with rectification and denormalization
        """
        Values = self._linear_reconstruction_ols(self.s.spec, self.diff_frames, self.lam,
                                                 self.predict_frames, self.max_bins)
        # spectrogram
        spec = self.s.spec
        # offset mask
        offset_mask = self._rectification_mask(self.s.spec, self.diff_frames, self.max_bins)
        # de-normalization factor
        Values['denormalization'] = np.sqrt(np.sum(spec ** 2, axis = 1))
        # calculate odf with different masks
        Values['odf'] = np.sqrt(np.sum(Values['residual_spec'] ** 2, axis = 1))
        Values['odf_rectification'] = np.sqrt(np.sum((Values['residual_spec'] * offset_mask) ** 2, axis = 1))      
        Values['odf_denormalization'] = Values['odf'] * Values['denormalization']
        Values['odf_offset_denormalization'] = np.sqrt(np.sum((Values['residual_spec'] * offset_mask) ** 2, axis = 1)) * Values['denormalization']
        return Values


class Onset(object):
    """
    Onset Class.

    """
    def __init__(self, activations, fps, online=True, sep=''):
        """
        Creates a new Onset object instance with the given activations of the
        ODF (OnsetDetectionFunction). The activations can be read from a file.

        :param activations: an array containing the activations of the ODF
        :param fps:         frame rate of the activations
        :param online:      work in online mode (i.e. use only past
                            information)

        """
        self.activations = None     # activations of the ODF
        self.fps = fps              # frame rate of the activation function
        self.online = online        # online peak-picking
        self.detections = []        # list of detected onsets (in seconds)
        # set / load activations
        if isinstance(activations, np.ndarray):
            # activations are given as an array
            self.activations = activations
        else:
            # read in the activations from a file
            self.load(activations, sep)

    def detect(self, threshold, combine=0.03, pre_avg=0.15, pre_max=0.01,
               post_avg=0, post_max=0.05, delay=0):
        """
        Detects the onsets.

        :param threshold: threshold for peak-picking
        :param combine:   only report 1 onset for N seconds
        :param pre_avg:   use N seconds past information for moving average
        :param pre_max:   use N seconds past information for moving maximum
        :param post_avg:  use N seconds future information for moving average
        :param post_max:  use N seconds future information for moving maximum
        :param delay:     report the onset N seconds delayed

        In online mode, post_avg and post_max are set to 0.

        Implements the peak-picking method described in:

        "Evaluating the Online Capabilities of Onset Detection Methods"
        Sebastian Böck, Florian Krebs and Markus Schedl
        Proceedings of the 13th International Society for Music Information
        Retrieval Conference (ISMIR), 2012

        """
        # online mode?
        if self.online:
            post_max = 0
            post_avg = 0
        # convert timing information to frames
        pre_avg = int(round(self.fps * pre_avg))
        pre_max = int(round(self.fps * pre_max))
        post_max = int(round(self.fps * post_max))
        post_avg = int(round(self.fps * post_avg))
        # convert to seconds
        combine /= 1000.
        delay /= 1000.
        # init detections
        self.detections = []
        # moving maximum
        max_length = pre_max + post_max + 1
        max_origin = int(np.floor((pre_max - post_max) / 2))
        mov_max = maximum_filter1d(self.activations, max_length,
                                   mode='constant', origin=max_origin)
        # moving average
        avg_length = pre_avg + post_avg + 1
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        mov_avg = uniform_filter1d(self.activations, avg_length,
                                   mode='constant', origin=avg_origin)
        # detections are activation equal to the moving maximum
        detections = self.activations * (self.activations == mov_max)
        # detections must be greater or equal than the mov. average + threshold
        detections *= (detections >= mov_avg + threshold)
        # convert detected onsets to a list of timestamps
        detections = np.nonzero(detections)[0].astype(np.float) / self.fps
        # shift if necessary
        if delay != 0:
            detections += delay
        # always use the first detection and all others if none was reported
        # within the last `combine` seconds
        if detections.size > 1:
            # filter all detections which occur within `combine` seconds
            combined_detections = detections[1:][np.diff(detections) > combine]
            # add them after the first detection
            self.detections = np.append(detections[0], combined_detections)
        else:
            self.detections = detections

    def write(self, filename):
        """
        Write the detected onsets to the given file.

        :param filename: the target file name

        Only useful if detect() was invoked before.

        """
        with open(filename, 'w') as f:
            for pos in self.detections:
                f.write(str(pos) + '\n')

    def save(self, filename, sep):
        """
        Save the onset activations to the given file.

        :param filename: the target file name
        :param sep: separator between activation values

        Note: using an empty separator ('') results in a binary numpy array.

        """
        self.activations.tofile(filename, sep=sep)

    def load(self, filename, sep):
        """
        Load the onset activations from the given file.

        :param filename: the target file name
        :param sep: separator between activation values

        Note: using an empty separator ('') results in a binary numpy array.

        """
        self.activations = np.fromfile(filename, sep=sep)

class Plot(object):
    """
    plot the spectrogram, ODFs, Labels
    :pararm Values:     spectrograms, alpha matrixs...
    :param mask_method: binary masks
    :param label:       estimation labels
    :param ans:         grond truth labels

    """
    def __init__(self, spec, hz, Values, mask_method, label, ans, act2, Values2):
        # setup matplotlin style
        font = {'family':'serif', 'weight':'normal','size': 32}
        matplotlib.rc('font', **font)
        self.layout = {"left": 0.08, "right": 0.99, "top": 0.9, "bottom": 0.19}
        # initialize 
        self.spec = spec.T
        self.approximate_spec = Values['approximate_spec'].T
        self.alpha_matrix = Values['alpha_matrix'].T
        self.residual_spec = Values['residual_spec'].T
        self.activation = Values[mask_method]
        self.amplitude = Values['denormalization']
        self.Norm_D = Values['Norm_D'].T
        self.label = label
        self.ans = ans
        self.act2 = act2
        self.activation2 = Values2[mask_method]
        self.hz = hz

    def plot(self, foldername, filename, fps):
        """
        Save plot of spectrogram and onset detection functions

        :param foldername: save fig in targe foldername 
        :param filename:   target file name
        :param fps:        frame per seconds
    
        """
        # save figure to foldername
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        # plot spectrogram
        fig1 = plt.figure(figsize=(15,5), dpi=600)
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (kHz)')        
        plt.imshow(self.spec, aspect='auto',
                   extent=[0, np.shape(self.spec)[1] / float(fps), self.hz[-1] / 1000., self.hz[0] / 1000.])
        plt.axis([0, np.shape(self.spec)[1] / float(fps), self.hz[0] / 1000. / 2, self.hz[-1] / 1000.])
        # for SPL paper 
        #plt.axis([0, 4.3, self.hz[0] / 1000., self.hz[-1] / 1000. /2])
        plt.subplots_adjust(**self.layout)
        plt.savefig('%s/1_%s.pdf' % (foldername, filename))
        plt.close()
        plt.cla()

        # plot coefficient matrix
        fig2 = plt.figure(figsize=(15,5), dpi=600)
        plt.xlabel('Time (sec)')
        plt.ylabel('Reconstruction index')
        img = plt.imshow(self.alpha_matrix ** 0.4, aspect='auto', cmap = 'gray_r',interpolation="nearest",
                         extent=[0, np.shape(self.alpha_matrix)[1] / float(fps), 0, np.shape(self.alpha_matrix)[0]])
        # for SPL paper 
        #plt.axis([0, 4.3, 0, np.shape(self.alpha_matrix)[0]])
        ax2 = plt.gca()
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("top", size="5%", pad=0.15)
        ax2 = plt.colorbar(img, cax=cax, orientation='horizontal')
        ax2.ax.tick_params(labeltop='on', labelbottom='off', size=30)
        plt.subplots_adjust(**self.layout)
        plt.savefig('%s/2_%s.pdf' % (foldername, filename))
        plt.close()
        plt.cla()

        # plot ODF
        fig3 = plt.figure(figsize=(15,5), dpi=600)
        plt.xlabel('Time (sec)')
        plt.ylabel('ODF')
        plt.plot(np.arange(np.size(self.activation)) / float(fps), self.activation, 'b', label=r'NNLS $\tau=%s$' % np.shape(self.alpha_matrix)[0])
        plt.plot(np.arange(np.size(self.activation2)) / float(fps), self.activation2, linestyle='-', color='#165833', alpha=0.7, label=r'NNLS $\tau=1$')
        #for i in self.label:
        #    plt.plot([i, i], [0.7 * np.max(self.activation), np.max(self.activation)], 'g')
        # plot label
        plt.scatter(self.ans, 0.7 * np.max(self.activation) * np.ones(len(self.ans)), s=80, marker='v', label='Ground truth', color='k')
        plt.legend(loc=1,prop={'size':30}, scatterpoints=1, ncol=3, columnspacing=0.8, handletextpad=0.1, handlelength=1.5, labelspacing=0.1)
        plt.axis([0, np.size(self.activation) / float(fps), 0, np.max(self.activation)])
        # for SPL paper
        #plt.axis([0, 4.3, 0, np.max(self.activation)])
        plt.subplots_adjust(**self.layout)
        plt.savefig('%s/3_%s.pdf' % (foldername, filename))
        plt.close()
        plt.cla()

        # plot ODF Superflux
        fig4 = plt.figure(figsize=(15,5), dpi=600)
        plt.xlabel('Time (sec)')
        plt.ylabel('ODF')
        plt.plot(np.arange(np.size(self.act2)) / float(fps), self.act2, 'b',label='Superflux')
        #for i in self.label:
        #    plt.plot([i, i], [0.7 * np.max(self.act2), np.max(self.act2)], 'g')
    
        # plot label
        plt.scatter(self.ans, 0.7 * np.max(self.act2) * np.ones(len(self.ans)), s=80, marker='v', label='Ground truth', color='k')
        plt.axis([0, np.size(self.act2) / float(fps), 0, np.max(self.act2)])
        # for SPL paper 
        #plt.axis([0, 4.3, 0, np.max(self.act2)])
        plt.legend(loc=1,prop={'size':30}, scatterpoints=1, ncol=2, columnspacing=0.8, handletextpad=0.1, handlelength=1.5, labelspacing=0.1)
        plt.subplots_adjust(**self.layout)
        plt.draw()
        plt.savefig('%s/4_%s.pdf' % (foldername, filename))

        # save figure
        plt.close()
        plt.cla()


def plot_superflux(spec, act, foldername, filename, fps, label, ans):
    """
    save plot of supeflux
    """
    # init matplotlib style
    font = {'family':'serif','weight':'normal','size': 22}
    matplotlib.rc('font', **font)
    layout = {"left": 0.07, "right": 0.995, "top": 0.97, "bottom": 0.145}
    # plot onset detection function
    plt.figure(figsize=(15.0, 5.0), dpi=600)
    plt.xlabel('Time (sec)')
    plt.ylabel('ODF')
    fig = plt.plot(np.arange(np.size(act)) / float(fps), act)
    # plot label
    plt.axis([0, np.size(act) / float(fps), 0, np.max(act)])
    for i in label:
         plt.plot([i, i], [0.7 * np.max(act), np.max(act)], 'r')
    plt.scatter(ans, 0.7 * np.max(act) * np.ones(len(ans)), s=60, marker='v', label='Ground truth', color='k')
    plt.subplots_adjust(**layout)
    # save figure to foldername
    if not os.path.isdir(foldername):
        os.makedirs(foldername)
    plt.savefig('%s/%s.pdf' % (foldername, filename))
    plt.cla()

def parser():
    """
    Parses the command line arguments.

    :param lgd:       use local group delay weighting by default
    :param threshold: default value for threshold

    """
    import argparse
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters, the software detects all onsets in
    the given files according to the method proposed in:

    "Maximum Filter Vibrato Suppression for Onset Detection"
    Sebastian Böck and Gerhard Widmer.
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), Maynooth, Ireland, September 2013

    If the '--lgd' switch is set, it additionally applies a local group delay
    based weighting according to the method proposed in:

    "Local group delay based vibrato and tremolo suppression for onset
     detection"
    Sebastian Böck and Gerhard Widmer.
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.

    The single most important parameter is the threshold ('-t'). Adjusting
    this parameter might help to improve performance considerably. Please note
    that if the local group delay weighting scheme is applied, the threshold
    should be adjusted to a lower value, e.g. 0.25.

    """)
    # general options
    p.add_argument('files', metavar='files', nargs='+',
                   help='files to be processed')
    p.add_argument('-v', dest='verbose', action='store_true',
                   help='be verbose')
    p.add_argument('-p', dest='plot', action='store_true', default=False,
                   help='save plot of ODF and spectrogram')
    p.add_argument('-s', dest='save', action='store_true', default=False,
                   help='save the activations of the onset detection function')
    p.add_argument('-l', dest='load', action='store_true', default=False,
                   help='load the activations of the onset detection function')
    p.add_argument('--sep', action='store', default='',
                   help='separator for saving/loading the onset detection '
                        'function [default=numpy binary]')
    p.add_argument('--act_suffix', action='store', default='.act',
                   help='filename suffix of the activations files '
                        '[default=%(default)s]')
    p.add_argument('--det_suffix', action='store', default='.superflux.txt',
                   help='filename suffix of the detection files '
                        '[default=%(default)s]')
    # online / offline mode
    p.add_argument('--online', action='store_true', default=False,
                   help='operate in online mode (i.e. no future information '
                        'will be used for computation)')
    # wav options
    wav = p.add_argument_group('audio arguments')
    wav.add_argument('--norm', action='store_true', default=None,
                     help='normalize the audio (switches to offline mode)')
    wav.add_argument('--att', action='store', type=float, default=None,
                     help='attenuate the audio by ATT dB')
    # spectrogram options
    spec = p.add_argument_group('spectrogram arguments')
    spec.add_argument('--fps', action='store', default=200, type=int,
                      help='frames per second [default=%(default)s]')
    spec.add_argument('--frame_size', action='store', type=int, default=2048,
                      help='frame size [samples, default=%(default)s]')
    spec.add_argument('--ratio', action='store', type=float, default=0.5,
                      help='window magnitude ratio to calc number of diff '
                           'frames [default=%(default)s]')
    spec.add_argument('--diff_frames', action='store', type=int, default=3,
                      help='diff frames')
    spec.add_argument('--max_bins', action='store', type=int, default=3,
                      help='bins used for maximum filtering '
                           '[default=%(default)s]')
    
    # LGD stuff
    mask = p.add_argument_group('local group delay based weighting')
    mask.add_argument('--lgd', action='store_true', default=False,
                      help='apply local group delay based weighting '
                           '[default=%(default)s]')
    mask.add_argument('--cd', action='store_true', default=False,
                      help='complex domain ODF '
                           '[default=%(default)s]')
    mask.add_argument('--sf', action='store_true', default=False,
                      help='SuperFlux ODF '
                           '[default=%(default)s]')
    mask.add_argument('--temporal_filter', action='store', default=3, type=int,
                      help='apply a temporal filter of N frames before '
                           'calculating the LGD weighting mask '
                           '[default=%(default)s]')
    # Proposed Approximation method
    approixmation = p.add_argument_group('linear reconstruction as reduction function')

    approixmation.add_argument('--bdpn', action='store_true', default=False,
                               help='solve linear reconstruction with bdpn'
                               '[default=%(default)s]')    
    approixmation.add_argument('--nnls', action='store_true', default=False,
                               help='solve linear reconstruction with nnls')
    approixmation.add_argument('--ols', action='store_true', default=False,
                               help='solve linear reconstruction with ols')      
    approixmation.add_argument('--pos', action='store_true', default=False,
                               help='Enable non-negative constraint')
    approixmation.add_argument('--lam', action='store', default=0.001, type=float,
                               help='value of lambda''[default=%(default)f]')
    approixmation.add_argument('--predict_frames', action='store', default=5, type=int,
                                help='value of predict length''[default=%(default)d]')
    approixmation.add_argument('--mask', action='store', default='odf_offset_denormalization',
                               help='Return odf with given mask')   
    approixmation.add_argument('--var1', action='store', default=0.5, type=float,
                               help='value of var1''[default=%(default)d]')
    approixmation.add_argument('--var2', action='store', default=0.5, type=float,
                               help='value of var2''[default=%(default)d]')                                

    # filtering
    filt = p.add_argument_group('magnitude spectrogram filtering arguments')
    filt.add_argument('--no_filter', dest='filter', action='store_false',
                      default=True, help='do not filter the magnitude '
                                         'spectrogram with a filterbank')
    filt.add_argument('--fmin', action='store', default=30, type=float,
                      help='minimum frequency of filter '
                           '[Hz, default=%(default)s]')
    filt.add_argument('--fmax', action='store', default=17000, type=float,
                      help='maximum frequency of filter '
                           '[Hz, default=%(default)s]')
    filt.add_argument('--bands', action='store', type=int, default=24,
                      help='number of bands per octave [default=%(default)s]')
    filt.add_argument('--equal', action='store_true', default=False,
                      help='equalize triangular windows to have equal area')
    filt.add_argument('--block_size', action='store', default=2048, type=int,
                      help='perform filtering in blocks of N frames '
                           '[default=%(default)s]')
    # logarithm
    log = p.add_argument_group('logarithmic magnitude spectrogram arguments')
    log.add_argument('--no_log', dest='log', action='store_false',
                     default=True, help='use linear magnitude scale')
    log.add_argument('--mul', action='store', default=1, type=float,
                     help='multiplier (before taking the log) '
                          '[default=%(default)s]')
    log.add_argument('--add', action='store', default=1, type=float,
                     help='value added (before taking the log) '
                          '[default=%(default)s]')
    # onset detection
    onset = p.add_argument_group('onset peak-picking arguments')
    onset.add_argument('-t', dest='threshold', action='store', type=float,
                       default=0.3, help='detection threshold '
                                               '[default=%(default)s]')
    onset.add_argument('--combine', action='store', type=float, default=0.03,
                       help='combine onsets within N seconds '
                            '[default=%(default)s]')
    onset.add_argument('--pre_avg', action='store', type=float, default=0.15,
                       help='build average over N previous seconds '
                            '[default=%(default)s]')
    onset.add_argument('--pre_max', action='store', type=float, default=0.01,
                       help='search maximum over N previous seconds '
                            '[default=%(default)s]')
    onset.add_argument('--post_avg', action='store', type=float, default=0,
                       help='build average over N following seconds '
                            '[default=%(default)s]')
    onset.add_argument('--post_max', action='store', type=float, default=0.05,
                       help='search maximum over N following seconds '
                            '[default=%(default)s]')
    onset.add_argument('--delay', action='store', type=float, default=0,
                       help='report the onsets N seconds delayed '
                            '[default=%(default)s]')
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)s (2016-7-07)')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return args
    return args


def main(args):
    """
    Main SuperFlux (Linear Reconstruction) program.

    :param args: parsed arguments

    """
    import os.path
    import glob
    import fnmatch
    from mir_eval.io import load_events 
    
    # determine the files to process
    files = []
    onsets = []
    for f in args.files:
        # check what we have (file/path)
        if os.path.isdir(f):
            # use all files in the given path
            files = glob.glob(f + '/*.wav')
            try:           
                onsets = glob.glob(f + '/*.onsets')
            except:
                pass
        else:
            # file was given, append to list
            files.append(f)
            try:
                onsets = glob.glob(os.path.dirname(os.path.abspath(f))+'/*.onsets')
            except:
                pass
    # only process .wav files
    files = fnmatch.filter(files, '*.wav')
    files.sort()
    try:
        onsets = fnmatch.filter(onsets, '*.onsets')
    except:
        pass
    onsets.sort()
    # init filterbank
    filt = None
    filterbank = None

    # process the files
    for i, f in enumerate(files):
        if args.verbose:
            print 'processing file %s' % f
        # load onsets answer
        if args.plot:
            try:
                ons = load_events(onsets[i])

            except:
                raise Exception("To plot figure, ground truths (*.onsets) must be in the same folder with wavfile")
        # use the name of the file without the extension
        filename = os.path.splitext(f)[0]
        # do the processing stuff unless the activations are loaded from file
        if args.load:
            # load the activations from file
            o = Onset("%s.act" % filename, args.fps, args.online, args.sep)
        else:
            # open the wav file
            w = Wav(f)
            # normalize audio
            if args.norm:
                w.normalize()
                args.online = False  # switch to offline mode
            # down-mix to mono
            if w.num_channels > 1:
                w.downmix()
            # attenuate signal
            if args.att:
                w.attenuate(args.att)
            # create filterbank if needed
            if args.filter:
                # re-create filterbank if the sample rate of the audio changes
                if filt is None or filt.fs != w.sample_rate:
                    filt = Filter(args.frame_size / 2, w.sample_rate,
                                  args.bands, args.fmin, args.fmax, args.equal)
                    filterbank = filt.filterbank
            # spectrogram
            s = Spectrogram(w, frame_size=args.frame_size, fps=args.fps,
                            filterbank=filterbank, log=args.log,
                            mul=args.mul, add=args.add, online=args.online,
                            block_size=args.block_size, lgd=args.lgd, cd=args.cd)
            # use the spectrogram to create an SpectralODF object
            sodf = SpectralODF(s, ratio=args.ratio, max_bins=args.max_bins,
                               diff_frames=args.diff_frames, lam=args.lam,
                               predict_frames=args.predict_frames, pos=args.pos, var1=args.var1, var2=args.var2)
            if args.plot:
                sodf2 = SpectralODF(s, ratio=args.ratio, max_bins=args.max_bins,
                                    diff_frames=args.diff_frames, lam=args.lam,
                                    predict_frames=1, pos=args.pos, var1=args.var1, var2=args.var2)
            # perform detection function on the object
            # BPDN+, BPDN
            if args.bdpn:
                raise Exception("new BPDN hasn't implemented")
                Values = sodf.linear_reconstruction_bdpn()            
                act = Values[args.mask]
                if args.plot:
                    Values2 = sodf2.linear_reconstruction_bdpn()      
                    act2 = sodf.superflux()
                odf_method = 'BDPN'
            # NNLS
            elif args.nnls:
                Values = sodf.linear_reconstruction_nnls()
                act = Values[args.mask]
                if args.plot:
                    Values2 = sodf2.linear_reconstruction_nnls()
                    act2 = sodf.superflux()
                odf_method = 'NNLS'
            # OLS
            elif args.ols:
                Values = sodf.linear_reconstruction_ols()
                act = Values[args.mask]
                if args.plot:
                    Values2 = sodf2.linear_reconstruction_ols()
                    act2 = sodf.superflux()
                odf_method = 'OLS'
            # Local Groupd Dealy
            elif args.lgd:
                act = sodf.complex_flux()
                odf_method = 'lgd' 
            # Complex Domain
            elif args.cd:
                act = sodf.complex_domain() 
                odf_method = 'CD'
            # SuperFlux
            elif args.sf:
                act = sodf.superflux()
                odf_method = 'SuperFlux'
            else:
                print "Onset detection algortihm is not specified. Using default: --nnls)"
                Values = sodf.linear_reconstruction_nnls()
                act = Values[args.mask]
                if args.plot:
                    Values2 = sodf2.linear_reconstruction_nnls()
                    act2 = sodf.superflux()
                odf_method = 'NNLS'
            # create an Onset object with the activations
            o = Onset(act, args.fps, args.online)
            if args.save:
                # save the raw ODF activations
                o.save("%s%s" % (filename, args.act_suffix), args.sep)
        # detect the onsets
        o.detect(args.threshold, args.combine, args.pre_avg, args.pre_max,
                 args.post_avg, args.post_max, args.delay)
        # write the onsets to a file
        o.write("%s.txt" % filename)
        # plot spectrogram and ODF  
        if args.plot:
            if args.cd or args.sf: 
                plot_superflux(s.spec, act, filename, "%s_%s" % (os.path.split(filename)[1], odf_method),
                               args.fps, o.detections, ons)  
            else:
                p = Plot(s.spec, filt.hz, Values, args.mask, o.detections, ons , act2, Values2)
                p.plot(filename, "%s_%s_Lambda%s_Pred%s_Pos_%s_Thres_%s_%s" % (os.path.split(filename)[1],
                       odf_method, args.lam, args.predict_frames, args.pos, args.threshold, args.mask), args.fps)
          
        # also output them to stdout if verbose
        if args.verbose:
            print 'detections:', o.detections


if __name__ == '__main__':
    # parse arguments
    args = parser()
    # and run the main SuperFlux program
    main(args)
