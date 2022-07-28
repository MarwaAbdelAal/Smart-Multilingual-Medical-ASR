#!/usr/bin/env python

from __future__ import print_function

from kaldi.asr import GmmLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.feat.functions import compute_deltas, DeltaFeaturesOptions
from kaldi.feat.window import FrameExtractionOptions
from kaldi.transform.cmvn import Cmvn
from kaldi.util.table import SequentialWaveReader
import pandas as pd


# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 11.0
decoder_opts.max_active = 7000
asr = GmmLatticeFasterRecognizer.from_files(
    "final.mdl", "HCLG.fst", "words.txt", decoder_opts=decoder_opts)



# Define feature pipeline in code
def feat_pipeline_fn(wav, base, opts=DeltaFeaturesOptions()):
    def feat_pipeline(wav):
        feats = base.compute_features(wav.data()[0], wav.samp_freq, 1.0)
        cmvn = Cmvn(base.dim())
        cmvn.accumulate(feats)
        cmvn.apply(feats)
        feat_return = compute_deltas(opts, feats)
        return compute_deltas(opts, feats)
    return feat_pipeline(wav)

def make_feat_pipeline(base, opts=DeltaFeaturesOptions()):
    def feat_pipeline(wav):
        feats = base.compute_features(wav.data()[0], wav.samp_freq, 1.0)
        cmvn = Cmvn(base.dim())
        cmvn.accumulate(feats)
        cmvn.apply(feats)
        feat_return = compute_deltas(opts, feats)
        return compute_deltas(opts, feats)
    return feat_pipeline

frame_opts = FrameExtractionOptions()
frame_opts.samp_freq = 44100
frame_opts.allow_downsample = True
mfcc_opts = MfccOptions()
mfcc_opts.use_energy = False
mfcc_opts.frame_opts = frame_opts
feat_pipeline = make_feat_pipeline(Mfcc(mfcc_opts))

# mapping words
df = pd.read_excel("dict.xlsx",header=None)
dictionary = df.to_dict('records')
#print(dictionary)

finaldict = {d[0]:d[1] for d in dictionary}

#print(dictionary)
#print(finaldict)
#print(df)
    
def convert(lst):
   return ' '.join(lst).split()
   
   
def get_key(val):
    for key, value in dictionary.items():
         if val == value:
             return key
 
    return "There is no such Key"
   
def sound_file_name(audio_name:str):
    #sarah_file02_8 ./sarah_file02_8.wav  #wav.scp content
    #to use the name of the file 
    f = open("wav.scp", "a")
    f.truncate(0) # to clear the content of the file before writing over it
    #file_name  = "noura_file01_2"
    file_name = audio_name
    f.write(f"{file_name} {file_name}")
    f.close()

'''
#open and read the file after the appending:
f = open("wav.scp", "r")
print(f.read()) 
'''

# Decode
def decode():
    for key, wav in SequentialWaveReader("scp:wav.scp"):
        #print ("wav", wav)
        feats = feat_pipeline_fn(wav, Mfcc(mfcc_opts))
        #print ("feats", feats)
        out = asr.decode(feats)
        #print ("out", out)
        
        # Driver code
        lst = [out["text"]]
        lst = convert(lst)
        #print ("lst", lst)
        #print ("convert(lst)[2]", convert(lst)[2])
        #print("out", out["text"])
        result= dict((new_val,new_k) for new_k,new_val in finaldict.items()).get(convert(lst)[2])
        #print("Dictionary search by value:",result)

        res = []
        
        for wrd in range (len(lst)):
            # searching from lookp_dict
            #print("wrd", wrd)
            #print("convert(lst)[wrd]" , convert(lst)[wrd])
            result= dict((new_val,new_k) for new_k,new_val in finaldict.items()).get(convert(lst)[wrd])
            #print("Dictionary search by value:",result)
            if ( result == None):
                result = convert(lst)[wrd]
            res.append(result)
        #print ("output: ",*res)
        sentence = ' '.join (res)
        print ("sentence", sentence)
    #print ("res", res)
    return (sentence)
     

    #print(wav)
    #print(key, out["text"], flush=True)
