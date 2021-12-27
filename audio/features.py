#!/usr/bin/env python3

# Load essentia and try to extract audio features

import essentia
import essentia.standard


# There's a couple of loaders: AudioLoaders returning audio samples
a_loader = essentia.standard.AudioLoader(filename='/tmp/g.ogg')
m = a_loader()

# Two channels are both given here, 
m[0].shape

# m[4] is the encoding bitrate
# m[5] is the encoding technique 'vorbis'


# Single channel, not sure what it picked.
loader = essentia.standard.MonoLoader(filename='/tmp/g.ogg')
audio = loader()


# Get all the beats in the sample
beat_tracker = essentia.standard.BeatTrackerDegara()

# You can get the beats of the right and left channel separately, or
# you can get it for the mono_audio
right_beats = beat_tracker(m[0][0])
left_beats = beat_tracker(m[0][1])

mono_beats = beat_tracker(audio)


# We need to get the frequency and magnitudes of spectral peaks for
# the HPCP (Harmonic Pitch Class Profile)
spec = essentia.standard.SpectralPeaks()

(freq, mag) = spec(audio)

# Then we can run
pitch = essentia.standard.HPCP()
class_profile = pitch(freq, mag)

cdb = essentia.standard.ChordsDetectionBeats()
(chords, strength) = cdb(class_profile, mono_beats)




s = essentia.standard.LoopBpmEstimator()
s(audio[100:300])

# Doing s(audio) doesn't work because there is silence in the
# beginning. I should find a way to get the beats, and then find the
# bpm for a few segments, and then average, or present what they are.



