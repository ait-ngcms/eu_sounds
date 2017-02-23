'''
Created on 27.05.2015

@author: SchindlerA
'''

# === IMPORTS ===========================================================
#
# --- Web-Framework relevant
#

import redis
import urlparse
from werkzeug.wrappers import Request, Response
from werkzeug.routing import Map, Rule
from werkzeug.exceptions import HTTPException, NotFound
from werkzeug.wsgi import SharedDataMiddleware
from werkzeug.utils import redirect

import gc

from jinja2 import Environment, FileSystemLoader

#
# --- Scientific calcualtions
#

import numpy  as np
import pandas as pd

#
# --- general
#
import sys
import os
#import urllib
#import tempfile
#from requests.exceptions import HTTPError
import warnings
warnings.filterwarnings('ignore')

#
# --- soundcloud feature extraction
#

# feature extraction
import librosa 

sys.path.append("C:/Work/IFS/rp_extract")
os.environ['PATH'] += os.pathsep + "D:/Research/Tools/ffmpeg/bin"

import audiofile_read as ar
from rp_extract import rp_extract

# soundcloud access
import tempfile
import soundcloud # (installed via pip)
import urllib
from requests.exceptions import HTTPError



# ===========================================================================================

# TEMPLATE_PATH     = "C:/Work/Eclipse/EU_Sounds/MIR_Prototype/templates"
# FEATURE_FILE_PATH = "E:/Data/MIR/EU_SOUNDS_FEATURES/combined_features.npz"
# METADATA_PATH     = "E:/Data/MIR/EU_SOUNDS_FEATURES/combined_metadata.p"
TEMPLATE_PATH     = "C:/Work/Eclipse/EU_Sounds/MIR_Prototype/templates"
FEATURE_FILE_PATH = "E:/Data/MIR/Evaluation_intermediate_data_DELETE_AFTER_2015/combined_features.npz"
METADATA_PATH     = "E:/Data/MIR/Evaluation_intermediate_data_DELETE_AFTER_2015/combined_metadata.p"
LIBROSA_FEATURES  = "E:/Data/MIR/Evaluation_intermediate_data_DELETE_AFTER_2015/combined_librosa_features.npz"

# ===========================================================================================

def expand_query_space(p,q):
    
    query_space = np.zeros(tuple(p.shape))
    query_space[:,:] = q
    
    return query_space


distance_measures= {}

distance_measures["l1"] = lambda data, i : \
    np.sum(abs(data - data[i,:]), axis=1)

distance_measures["canberra"] = lambda data, query : \
    np.nansum(abs(data - query) / (abs(data) + abs(query)), axis=1)

distance_measures["wave_hedges"] = lambda data, i : \
    np.sum( np.abs(data - data[i,:]) / np.max([data, expand_query_space(data,data[1,:])], axis=0), axis=1)
    
distance_measures["dice"] = lambda data, i : \
    np.sum( (data - data[i,:])**2, axis=1) / \
    (np.sum( data**2, axis=1) + np.sum( expand_query_space(data,data[1,:])**2, axis=1))
    
distance_measures["braycurtis"] = lambda data, i : \
    np.abs(data - data[i,:]).sum(axis=1) / np.abs(data + data[i,:]).sum(axis=1)
    
distance_measures["cosine"] = lambda dat, i : \
    np.abs(1 - np.sum( dat * dat[i,:], axis=1) / \
    (np.sqrt(np.sum( dat**2, axis=1)) * np.sqrt(np.sum(expand_query_space(dat,dat[i,:])**2, axis=1))))
    
    
# ===========================================================================================

feature_extractor = {}

                                                               
feature_extractor["mfcc"] = lambda S, sr : \
\
        librosa.feature.mfcc(S      = S,                       # power spectrogram
                             n_mfcc = 13)                      # number of MFCCs to return
        
feature_extractor["chroma"] = lambda S, sr : \
\
        librosa.feature.chroma_stft(S          = S,         # power spectrogram
                                    norm       = np.inf,       # Column-wise normalization
                                    tuning     = None)         # Deviation from A440 tuning in fractional bins (cents). 

feature_extractor["rmse"] = lambda S, sr : \
\
        librosa.feature.rmse(S  = S)                           # power spectrogram
        
feature_extractor["spectral_centroid"] = lambda S, sr : \
\
        librosa.feature.spectral_centroid(S  = S)             # samplerate
        
feature_extractor["spectral_bandwidth"] = lambda S, sr : \
\
        librosa.feature.spectral_bandwidth(S  = S)            # samplerate

feature_extractor["spectral_contrast"] = lambda S, sr : \
\
        librosa.feature.spectral_contrast(S          = S,      # power spectrogram
                                          freq       = None,   # Center frequencies for spectrogram bins. If None, then 
                                                               #   FFT bin center frequencies are used. Otherwise, it 
                                                               #   can be a single array of d center frequencies, or 
                                                               #   a matrix of center frequencies as constructed by 
                                                               #   librosa.core.ifgram, centroid=None, norm=True, p=2)
                                          fmin       = 200.0,  # Frequency cutoff for the first bin [0, fmin] Subsequent
                                                               #   bins will cover [fmin, 2*fmin], [2*fmin, 4*fmin], etc.
                                          n_bands    = 6,      # number of frequency bands
                                          quantile   = 0.02,   # quantile for determining peaks and valleys
                                          linear     = False)  # If True, return the linear difference of magnitudes: 
                                                               
feature_extractor["spectral_rolloff"] = lambda S, sr : \
\
        librosa.feature.spectral_rolloff(S            = S,      # power spectrogram
                                         freq         = None,   # Center frequencies for spectrogram bins. If None, then 
                                                                #   FFT bin center frequencies are used. Otherwise, it 
                                                                #   can be a single array of d center frequencies, or 
                                                                #   a matrix of center frequencies as constructed by 
                                                                #   librosa.core.ifgram, centroid=None, norm=True, p=2)
                                         roll_percent = 0.85)
        
feature_extractor["tonnetz"] = lambda chroma, sr : \
\
        librosa.feature.tonnetz(chroma = chroma)                # Normalized energy for each chroma bin at each frame.
                                                                #   If None, a cqt chromagram is performed.
                                                                
feature_extractor["zero_crossing_rate"] = lambda y : \
\
        librosa.feature.zero_crossing_rate(y            = y,    # audio time series
                                           frame_length = 2048, # Length of the frame over which to compute 
                                                                # zero crossing rates
                                           hop_length   = 512,  # hop length if provided y, sr instead of S
                                           center       = True) # If True, frames are centered by padding the edges of y. 
                                                                #   This is similar to the padding in librosa.core.stft, 
                                                                #   but uses edge-value copies instead of reflection.
                                                                
def calc_bpm(S, sr, audio_length):
             
    onset_env = librosa.onset.onset_strength(S         = S,     # pre-computed (log-power) spectrogram
                                             detrend   = False, # Filter the onset strength to remove the DC component
                                             centering = True,  # Shift the onset function by n_fft / (2 * hop_length) frames
                                             feature   = None,  # Function for computing time-series features, eg, scaled
                                                                #   spectrograms. By default, uses 
                                                                #   librosa.feature.melspectrogram with fmax=8000.0
                                             aggregate = None)  # Aggregation function to use when combining onsets at 
                                                                #   different frequency bins.Default: np.mean
    
    
    if audio_length > 60:
        duration = 40.0
        offset   = 10.0
    else:
        duration = audio_length
        offset   = 0.0
                          
             
    bpm = librosa.beat.estimate_tempo(onset_env,                # onset_envelope : onset strength envelope
                                      sr         = sr,          # sampling rate of the time series
                                      start_bpm  = 120,         # initial guess of the BPM
                                      std_bpm    = 1.0,         # standard deviation of tempo distribution
                                      ac_size    = 4.0,         # length (in seconds) of the auto-correlation window
                                      duration   = duration,    # length of signal (in seconds) to use in estimating tempo
                                      offset     = offset)      # offset (in seconds) of signal sample to use in estimating
                                                                # tempo
             
    return bpm

# ===========================================================================================

def base36_encode(number):
    assert number >= 0, 'positive integer required'
    if number == 0:
        return '0'
    base36 = []
    while number != 0:
        number, i = divmod(number, 36)
        base36.append('0123456789abcdefghijklmnopqrstuvwxyz'[i])
    return ''.join(reversed(base36))


def is_valid_url(url):
    parts = urlparse.urlparse(url)
    return parts.scheme in ('http', 'https')


def get_hostname(url):
    return urlparse.urlparse(url).netloc

# ===========================================================================================

    
class MIR_enabled_music_search_engine(object):

    def __init__(self, config):
        
        self.redis                         = redis.Redis(config['redis_host'], config['redis_port'])
        self.jinja_env                     = Environment(loader=FileSystemLoader(TEMPLATE_PATH), autoescape=True)
        self.jinja_env.filters['hostname'] = get_hostname

        self.url_map = Map([
            Rule('/', endpoint='new_url'),
            Rule('/<short_id>', endpoint='follow_short_link'),
            Rule('/<short_id>+', endpoint='short_link_details')
        ])
        
        # soundcloud access
        self.client = soundcloud.Client(client_id='808da19eeda3bd78fc7e8483e8a97406')
        
        
        self.metadata = pd.read_pickle(METADATA_PATH)
        
        # open pickle
        self.features = {}
         
        # load features
        npz = np.load(FEATURE_FILE_PATH)
        self.features["ssd"] = npz["ssd"][self.metadata["index_rp"], :]
        self.features["rp"]  = npz["rp"][self.metadata["index_rp"], :]
        npz.close()
 
        
        npz = np.load(LIBROSA_FEATURES)
        
        self.features["mfcc"] = npz["mfcc"][self.metadata["index_librosa"].values,:]
        #self.features["mfcc"] -= self.features["mfcc"].min(axis=0)
        #self.features["mfcc"] /= self.features["mfcc"].max(axis=0)
        
        self.features["chroma"] = npz["chroma"][self.metadata["index_librosa"].values,:]
        #self.features["chroma"] -= self.features["chroma"].min(axis=0)
        #self.features["chroma"] /= self.features["chroma"].max(axis=0)
        
        self.features["rmse"] = npz["rmse"][self.metadata["index_librosa"].values,:]
        #self.features["rmse"] -= self.features["rmse"].min(axis=0)
        #self.features["rmse"] /= self.features["rmse"].max(axis=0)
        
        self.features["spectral_centroid"] = npz["spectral_centroid"][self.metadata["index_librosa"].values,:]
        #self.features["spectral_centroid"] -= self.features["spectral_centroid"].min(axis=0)
        #self.features["spectral_centroid"] /= self.features["spectral_centroid"].max(axis=0)
        
        self.features["spectral_bandwidth"] = npz["spectral_bandwidth"][self.metadata["index_librosa"].values,:]
        
        self.features["spectral_contrast"] = npz["spectral_contrast"][self.metadata["index_librosa"].values,:]
        
        self.features["spectral_rolloff"] = npz["spectral_rolloff"][self.metadata["index_librosa"].values,:]
        
        self.features["tonnetz"] = npz["tonnetz"][self.metadata["index_librosa"].values,:]
        #self.features["tonnetz"] -= self.features["tonnetz"].min(axis=0)
        #self.features["tonnetz"] /= self.features["tonnetz"].max(axis=0)
        
        self.features["zero_crossing_rate"] = npz["zero_crossing_rate"][self.metadata["index_librosa"].values,:]
        #self.features["zero_crossing_rate"] -= self.features["zero_crossing_rate"].min(axis=0)
        #self.features["zero_crossing_rate"] /= self.features["zero_crossing_rate"].max(axis=0)
        
        self.features["bpm"] = npz["bpm"][self.metadata["index_librosa"].values]
        self.features["bpm"] = self.features["bpm"].reshape((self.features["bpm"].shape[0],1))
        
        npz.close()
        
        
        print "loaded"
        print "dataset-size", self.metadata.shape[0]
    
        print "normalize data"
        for key in self.features.keys():
            self.features[key] -= self.features[key].min(axis=0)  
            self.features[key] /= self.features[key].max(axis=0)
    
    
        self.metadata["index"] = np.arange(self.features["ssd"].shape[0])
        
        #i = 0
        #for key in self.metadata.columns:
        #    print i, key
        #    i += 1
        
        
#     def calc_similar_items(self, idx):
# 
#         dists_ssd = distance_measures["canberra"](self.features["ssd"], 
#                                                   self.features["ssd"][idx,:])
#         dists_ssd = 1 - (dists_ssd / dists_ssd.max())
# 
#         dists_rp = distance_measures["canberra"](self.features["rp"],
#                                                  self.features["rp"][idx,:])
#         dists_rp = 1 - (dists_rp / dists_rp.max())
# 
#         dists_mfcc = distance_measures["canberra"](self.features["mfcc"], 
#                                                    self.features["mfcc"][idx,:])
#         dists_mfcc = 1 - (dists_mfcc / dists_mfcc.max())
# 
#         dists_chroma = distance_measures["canberra"](self.features["chroma"], 
#                                                      self.features["chroma"][idx,:])
#         dists_chroma = 1 - (dists_chroma / dists_chroma.max())
# 
#         dists_bpm = distance_measures["canberra"](self.features["bpm"], 
#                                                   self.features["bpm"][idx,:])
#         dists_bpm = 1 - (dists_bpm / dists_bpm.max())
#         
#         dists_rmse = distance_measures["canberra"](self.features["rmse"], 
#                                                    self.features["rmse"][idx,:])
#         dists_rmse = 1 - (dists_rmse / dists_rmse.max())
#         
#         dists_spectral_centroid = distance_measures["canberra"](self.features["spectral_centroid"], 
#                                                                 self.features["spectral_centroid"][idx,:])
#         dists_spectral_centroid = 1 - (dists_spectral_centroid / dists_spectral_centroid.max())
#         
#         dists_tonnetz = distance_measures["canberra"](self.features["tonnetz"], 
#                                                       self.features["tonnetz"][idx,:])
#         dists_tonnetz = 1 - (dists_tonnetz / dists_tonnetz.max())
#         
#         dists_zero_crossing_rate = distance_measures["canberra"](self.features["zero_crossing_rate"], 
#                                                                  self.features["zero_crossing_rate"][idx,:])
#         dists_zero_crossing_rate = 1 - (dists_zero_crossing_rate / dists_zero_crossing_rate.max())
#         
#         sims = (dists_ssd                 * 0.9 + \
#                  dists_rp                 * 1.99 + \
#                  dists_mfcc               * 2.5 + \
#                  dists_chroma             * 1.3 + \
#                  dists_bpm                * 0.8 + \
#                  dists_rmse               * 1.0 + \
#                  dists_spectral_centroid  * 0.9 + \
#                  dists_tonnetz            * 1.3 + \
#                  dists_zero_crossing_rate * 0.2)
#         
#         sims /= sims.max()
# 
#         nn = np.argsort(sims)[::-1]
# 
#         return sims, nn
    
    def calc_similar_items(self, tssd, rp, chroma, mfcc, rmse, spectral_centroid, 
                           spectral_bandwidth, spectral_contrast, spectral_rolloff,  
                           tonnetz, zero_crossing_rate, bpm):

        dists_ssd = distance_measures["canberra"](self.features["ssd"], 
                                                  tssd)
        dists_ssd = 1 - (dists_ssd / dists_ssd.max())

        dists_rp = distance_measures["canberra"](self.features["rp"],
                                                 rp)
        dists_rp = 1 - (dists_rp / dists_rp.max())

        dists_mfcc = distance_measures["canberra"](self.features["mfcc"], 
                                                   mfcc)
        dists_mfcc = 1 - (dists_mfcc / dists_mfcc.max())

        dists_chroma = distance_measures["canberra"](self.features["chroma"], 
                                                     chroma)
        dists_chroma = 1 - (dists_chroma / dists_chroma.max())

        dists_bpm = distance_measures["canberra"](self.features["bpm"], 
                                                  bpm)
        dists_bpm = 1 - (dists_bpm / dists_bpm.max())
        
        dists_rmse = distance_measures["canberra"](self.features["rmse"], 
                                                   rmse)
        dists_rmse = 1 - (dists_rmse / dists_rmse.max())
        
        dists_spectral_centroid = distance_measures["canberra"](self.features["spectral_centroid"], 
                                                                spectral_centroid)
        dists_spectral_centroid = 1 - (dists_spectral_centroid / dists_spectral_centroid.max())
        
        dists_tonnetz = distance_measures["canberra"](self.features["tonnetz"], 
                                                      tonnetz)
        dists_tonnetz = 1 - (dists_tonnetz / dists_tonnetz.max())
        
        dists_zero_crossing_rate = distance_measures["canberra"](self.features["zero_crossing_rate"], 
                                                                 zero_crossing_rate)
        dists_zero_crossing_rate = 1 - (dists_zero_crossing_rate / dists_zero_crossing_rate.max())
        
        sims = (dists_ssd                 * 0.9 + \
                 dists_rp                 * 1.99 + \
                 dists_mfcc               * 2.5 + \
                 dists_chroma             * 1.3 + \
                 dists_bpm                * 0.8 + \
                 dists_rmse               * 1.0 + \
                 dists_spectral_centroid  * 0.9 + \
                 dists_tonnetz            * 1.3 + \
                 dists_zero_crossing_rate * 0.2)
        
        sims /= sims.max()

        nn = np.argsort(sims)[::-1]
        
        print sims

        return sims[nn], nn
    

    def extract_librosa_features(self, wavedata, samplerate):
        
        try:
            
            # merge audio channels
            wavedata           = wavedata.mean(axis=1)
            
            # calculate spectrogram
            spectrogram, phase = librosa.magphase(librosa.stft(wavedata, n_fft = 2048))
    
    
            # extract features
            feat_chroma             = feature_extractor["chroma"](spectrogram, samplerate)
            chroma = np.concatenate([feat_chroma.mean(axis=1).flatten(), 
                                     feat_chroma.std(axis=1).flatten()], axis=1)
            
            feat               = feature_extractor["mfcc"](spectrogram, samplerate)
            mfcc = np.concatenate([feat.mean(axis=1).flatten(), 
                                     feat.std(axis=1).flatten()], axis=1)
            
            feat               = feature_extractor["rmse"](spectrogram, samplerate)
            rmse = np.concatenate([feat.mean(axis=1).flatten(), 
                                     feat.std(axis=1).flatten()], axis=1)
            
            feat  = feature_extractor["spectral_centroid"](spectrogram, samplerate)
            spectral_centroid = np.concatenate([feat.mean(axis=1).flatten(), 
                                     feat.std(axis=1).flatten()], axis=1)
            
            feat = feature_extractor["spectral_bandwidth"](spectrogram, samplerate)
            spectral_bandwidth = np.concatenate([feat.mean(axis=1).flatten(), 
                                     feat.std(axis=1).flatten()], axis=1)
            
            feat  = feature_extractor["spectral_contrast"](spectrogram, samplerate)
            spectral_contrast = np.concatenate([feat.mean(axis=1).flatten(), 
                                     feat.std(axis=1).flatten()], axis=1)
            
            feat   = feature_extractor["spectral_rolloff"](spectrogram, samplerate)
            spectral_rolloff = np.concatenate([feat.mean(axis=1).flatten(), 
                                     feat.std(axis=1).flatten()], axis=1)
            
            feat            = feature_extractor["tonnetz"](feat_chroma, samplerate)
            tonnetz = np.concatenate([feat.mean(axis=1).flatten(), 
                                     feat.std(axis=1).flatten()], axis=1)
            
            feat = feature_extractor["zero_crossing_rate"](wavedata)
            zero_crossing_rate = np.concatenate([feat.mean(axis=1).flatten(), 
                                     feat.std(axis=1).flatten()], axis=1)
            
            
            audio_length       = wavedata.shape[0] / float(samplerate)
            bpm                = calc_bpm(spectrogram, samplerate, audio_length)
            
            
        #except Exception as e:
        #    print "**", e
        finally:
            pass
            
        return chroma, mfcc, rmse, spectral_centroid, spectral_bandwidth, spectral_contrast, \
               spectral_rolloff, tonnetz, zero_crossing_rate, bpm
    
    def get_similar_for_soundcloud_track(self, sc_track_id):
        
        extracted_features = None
        
        tmp_path = tempfile.mktemp(prefix="soundcloud.", suffix=".mp3")
        
        try:
            
            # occasionally, a track has been removed and the stream is no longer available, so we have to catch a 404 error
            try:
                stream = self.client.get('/tracks/%d/streams' % sc_track_id)
                 
                if not hasattr(stream, 'http_mp3_128_url'):
                    raise Exception("No download URL available!")
                     
                mp3_download_url = stream.http_mp3_128_url
                 
                urllib.urlretrieve (mp3_download_url, tmp_path)
                
                #tmp_path = "c:/users/schind~1/appdata/local/temp/soundcloud.hziisf.mp3"
                (samplerate, samplewidth, wavedata) = ar.mp3_read(tmp_path)
                
                
                print wavedata.shape, samplerate, tmp_path
    
                extracted_features = rp_extract(wavedata,                            # the two-channel wave-data of the audio-file
                                                samplerate,                          # the samplerate of the audio-file
                                                extract_rp          = True,          # <== extract this feature!
                                                extract_ssd         = True,
                                                extract_tssd        = False,
                                                transform_db        = True,          # apply psycho-accoustic transformation
                                                transform_phon      = True,          # apply psycho-accoustic transformation
                                                transform_sone      = True,          # apply psycho-accoustic transformation
                                                fluctuation_strength_weighting=True, # apply psycho-accoustic transformation
                                                skip_leadin_fadeout = 1,             # skip lead-in/fade-out. value = number of segments skipped
                                                step_width          = 1)             # 
    
                
                chroma, mfcc, rmse, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff, \
                tonnetz, zero_crossing_rate, bpm = self.extract_librosa_features(wavedata, samplerate)
                
                print bpm
                
                sims, nn = self.calc_similar_items(
                               extracted_features["ssd"],
                               extracted_features["rp"], 
                               chroma, mfcc, rmse, spectral_centroid, 
                               spectral_bandwidth, spectral_contrast, 
                               spectral_rolloff, tonnetz, zero_crossing_rate, bpm)
    
            except HTTPError as e:
            
                if e.response.status_code == 404:
                    print "Track stream not found (404)! Skipping track."
                else:
                    raise e
            
            
        except Exception as e:
            raise e
            
        finally:
                    
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
        
        return sims, nn

    def on_new_url(self, request):
        
        error  = None
        tracks = []
        
        
        
        if request.method == 'GET' and request.args["cmd"] == "similar":
            
            print request.args["index"]
            
            
            query_idx = int(request.args["index"])
            
            sims, nn = self.calc_similar_items(
                               self.features["ssd"][query_idx,:],
                               self.features["rp"][query_idx,:], 
                               self.features["chroma"][query_idx,:], 
                               self.features["mfcc"][query_idx,:], 
                               self.features["rmse"][query_idx,:], 
                               self.features["spectral_centroid"][query_idx,:] , 
                               self.features["spectral_bandwidth"][query_idx,:] ,
                               self.features["spectral_contrast"][query_idx,:] ,
                               self.features["spectral_rolloff"][query_idx,:] ,  
                               self.features["tonnetz"][query_idx,:] ,
                               self.features["zero_crossing_rate"][query_idx,:] , 
                               self.features["bpm"][query_idx])
            
            
            for i in range(24):
                t = self.metadata.iloc[nn[i],:]#.values[:24,:]
                print sims[i], unicode(t[3][:20].encode("utf-8"), "utf8")
                
                
            tracks = [dict(mp3_link     = t[0].decode("utf-8"), 
                           index        = t[29],
                           title        = unicode(t[3][:20].encode("utf-8"), "utf8"),
                           organization = unicode(t[5].encode("utf-8"), "utf8"),
                           collection   = unicode(t[28].encode("utf-8"), "utf8"),
                           language     = unicode(t[27].encode("utf-8"), "utf8"),
                           link         = unicode(t[9].encode("utf-8"), "utf8")
                               ) for t in self.metadata.iloc[nn[:24],:].values[:24,:]]
            
            
            
        elif request.method == 'POST':
            
            query = request.form['query']
        
            print query
            
            if query != "":
                
                if query.find("https://") != -1:
                    
                    track = self.client.get('/resolve', url=query)
                    sims, nn = self.get_similar_for_soundcloud_track(track.id)
                    
                    tracks = [dict(mp3_link     = t[0].decode("utf-8"), 
                           index        = t[29],
                           title        = unicode(t[3][:20].encode("utf-8"), "utf8"),
                           organization = unicode(t[5].encode("utf-8"), "utf8"),
                           collection   = unicode(t[28].encode("utf-8"), "utf8"),
                           language     = unicode(t[27].encode("utf-8"), "utf8"),
                           link         = unicode(t[9].encode("utf-8"), "utf8")
                               ) for t in self.metadata.iloc[nn[:24],:].values[:24,:]]
                    
                else:
                
                    tracks = [dict(mp3_link     = t[0].decode("utf-8"), 
                                   index        = t[29],
                                   title        = unicode(t[3][:20].encode("utf-8"), "utf8"),
                                   organization = unicode(t[5].encode("utf-8"), "utf8"),
                                   collection   = unicode(t[28].encode("utf-8"), "utf8"),
                                   language     = unicode(t[27].encode("utf-8"), "utf8"),
                                   link         = unicode(t[9].encode("utf-8"), "utf8")
                                   ) for t in self.metadata[self.metadata["title"].str.contains(query, case=False)].values[:24,:]]
                
                
        gc.collect()

        return self.render_template('new_url.html', error=error, tracks=tracks)

    def on_follow_short_link(self, request, short_id):
        link_target = self.redis.get('url-target:' + short_id)
        if link_target is None:
            raise NotFound()
        self.redis.incr('click-count:' + short_id)
        return redirect(link_target)

    def on_short_link_details(self, request, short_id):
        link_target = self.redis.get('url-target:' + short_id)
        if link_target is None:
            raise NotFound()
        click_count = int(self.redis.get('click-count:' + short_id) or 0)
        return self.render_template('short_link_details.html',
            link_target=link_target,
            short_id=short_id,
            click_count=click_count
        )

    def error_404(self):
        response = self.render_template('404.html')
        response.status_code = 404
        return response

    def insert_url(self, url):
        short_id = self.redis.get('reverse-url:' + url)
        if short_id is not None:
            return short_id
        url_num = self.redis.incr('last-url-id')
        short_id = base36_encode(url_num)
        self.redis.set('url-target:' + short_id, url)
        self.redis.set('reverse-url:' + url, short_id)
        return short_id

    def render_template(self, template_name, **context):
        t = self.jinja_env.get_template(template_name)
        return Response(t.render(context), mimetype='text/html')

    def dispatch_request(self, request):
        adapter = self.url_map.bind_to_environ(request.environ)
        try:
            endpoint, values = adapter.match()
            return getattr(self, 'on_' + endpoint)(request, **values)
        except NotFound, e:
            return self.error_404()
        except HTTPException, e:
            return e

    def wsgi_app(self, environ, start_response):
        request = Request(environ)
        response = self.dispatch_request(request)
        return response(environ, start_response)

    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)
    
    



def create_app(redis_host='localhost', redis_port=6379, with_static=True):
    
    app = MIR_enabled_music_search_engine({
        'redis_host':       redis_host,
        'redis_port':       redis_port
    })
    
    if with_static:
        
        app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
            '/static':  os.path.join(os.path.dirname(__file__), 'static')
        })
        
    return app


if __name__ == '__main__':
    
    from werkzeug.serving import run_simple
    app = create_app()
    #run_simple('172.20.36.10', 5000, app, use_debugger=True, use_reloader=True)
    run_simple('127.0.0.1', 5000, app, use_debugger=True, use_reloader=True)

