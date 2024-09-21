# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:38:17 2024

@author: Anna
"""

import numpy as np
#import scamp
#import os
#import librosa as lr
import plotly.graph_objects as go
#import scipy.io
#import sounddevice as sd
#from scamp import *
#from os import listdir
#from os.path import isfile, join
#import os
#from scipy.io import wavfile
from plotly.subplots import make_subplots
import librosa
#import librosa.display

#import matplotlib.pyplot as plt
from PIL import Image

#voicepath = "G:\\Andere Computer\\Mein Laptop\\Projekte\\Juta\\Reaper\\Renderings\\Vowels"
#voicepath = './data/Vowels'
names = ['Gabriele','Ieva','Karolina M.','Karolina R.','Lina','Neringa','Beata','Onute']
# new_data_load = False
# if new_data_load:
#     def load_wav_files(directory):#directory = voicepath
#         folder_arrays = {}
    
#         # Traverse through all subdirectories in the given directory
#         for folder_name in os.listdir(directory):
#             folder_path = os.path.join(directory, folder_name)
    
#             # Check if it's a directory
#             if os.path.isdir(folder_path):
#                 wav_arrays = []
                
#                 # Traverse files within the folder
#                 for file_name in os.listdir(folder_path):
#                     if file_name.endswith(".wav"):
#                         file_path = os.path.join(folder_path, file_name)
#                         print(file_path)
#                         # Read the wav file and store the data as a NumPy array
#                         sample_rate, data = wavfile.read(file_path)
#                         wav_arrays.append(data)
                
#                 # Store the list of NumPy arrays (for each wav file) with the folder name as the key
#                 folder_arrays[folder_name] = np.array(wav_arrays)
        
#         return folder_arrays
    
        
#     folder_wav_data = load_wav_files(voicepath)
#     np.savez("./data/folder_wav_data.npz",**folder_wav_data)
# else:
folder_wav_data=np.load("./data/folder_wav_data.npz")#"G:\\Andere Computer\\Mein Laptop\\Projekte\\Juta\\Python\\folder_wav_data.npz")
Vc = len(folder_wav_data)

n_fft = 2048
hop_length = int(n_fft/4)
fs = 44100
sr = fs



# Display the loaded NumPy arrays
fig_specs = []
specFig_matrix = [[None for _ in np.arange(Vc)] for _ in names]
vowCnt = 0
allaudio = []
vowels = []
allmeans = [None for _ in np.arange(Vc)] 
for folder, arrays in folder_wav_data.items():
    #print(folder)
    
    vowels.append(folder)
    #print(f"Folder: {folder}, Number of WAV files: {len(arrays)}")
    allaudio.append([])
    figmean = go.Figure()
    spectrograms = []
    mean_ffts = []
    #allaudio = allaudio.append(arrays)
    for audio_data in arrays: # audio_data = arrays[0,:]
        # Generate a spectrogram using librosa's STFT
        audio_data = audio_data/np.max(np.abs(audio_data))
        allaudio[vowCnt].append(audio_data)
        #playaudio[vowCnt,singerCnt]=audio_data
        S = librosa.stft(audio_data,n_fft=n_fft)#2048
        times = np.arange(0,S.shape[1]) * hop_length
        frequs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        spectrograms.append(S_db)
        mean_fft = np.mean(S_db, axis=1)
        mean_ffts.append(mean_fft)
        
    num_files = len(spectrograms)
    
    fig_specs.append(make_subplots(
        rows=2, cols=num_files,
        subplot_titles=[names[i] for i in range(num_files)],#[f'File {i+1}' for i in range(num_files)],
        row_heights=[0.75, 0.25],
        #shared_xaxes=True
    ))    
    for i, S_db in enumerate(spectrograms):
        
        fig_specs[vowCnt].add_trace(
            go.Heatmap(x=times/fs,y=frequs,z=S_db, colorscale='Viridis', showscale=False),
            row=1, col=i+1
        )
        figi=go.Figure()
        figi.add_trace(
            go.Heatmap(x=times/fs,y=frequs,z=S_db, colorscale='Viridis', showscale=False))

        
        specFig_matrix[i][vowCnt]=figi
        
    allmeans[vowCnt]=mean_ffts
    vowCnt = vowCnt+1



# Formantfrequenzen für die Vokale
# Die Reihenfolge der Vokale in den Zeilen: /i/, /e/, /a/, /o/, /u/, /ae/, /ɒ/, /ɘ/
formant_frequencies = np.array([
    [400, 2300, 2900, 3500, 4000, 4500, 5000, 5500],  # /i/
    [400, 1900, 2500, 3100, 3600, 4100, 4600, 5100],  # /e/
    [700, 1200, 2600, 3200, 3700, 4200, 4700, 5200],  # /a/
    [600, 800, 2400, 3000, 3500, 4000, 4500, 5000],   # /o/
    [300, 600, 2400, 2900, 3400, 3900, 4400, 4900],   # /u/
    [800, 1800, 2500, 3100, 3600, 4100, 4600, 5100],  # /ae/
    [600, 1000, 2500, 3000, 3500, 4000, 4500, 5000],  # /ɒ/
    [500, 1400, 2500, 3000, 3500, 4000, 4500, 5000]   # /ɘ/
])
FF = np.zeros((formant_frequencies.shape[0],formant_frequencies.shape[1],len(names)))
FFi = np.zeros((formant_frequencies.shape[0],formant_frequencies.shape[1],len(names)),dtype=int)
typicalv=np.array(['i','e','a','o','u','ae','a_','e_'])
# Ausgabe der Formantfrequenzen
#print(formant_frequencies)

# find formant frequencies
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
tolerance = 300
timax = np.arange(2*fs)*1/fs
sines = np.zeros((len(vowels),len(names),2*fs))
sines_transp = np.zeros((len(vowels),len(names),2*fs))
for ii,vow in enumerate(vowels):
    #print(vow)
    idxt = np.argwhere(typicalv==vow)
    peaks = formant_frequencies[idxt[0][0],:]
    for jj,singer in enumerate(names):
        #peak_indices,_=find_peaks(allmeans[ii][jj])
        #found_peaks=[]
        # for ip,p in enumerate(peaks):
        #     tol = int(p*0.1)
        #     fidx = np.argmin(np.abs(p-frequs))
        #     fidx_min =  np.argmin(np.abs((p-tol)-frequs))
        #     fidx_max =  np.argmin(np.abs((p+tol)-frequs))
        #     #frequs[fidx_min]
        #     curve = allmeans[ii][jj][fidx_min:fidx_max]          
        #     FF[ii,ip,jj]= frequs[np.argmax(curve)+fidx_min]
        #     FFi[ii,ip,jj]=np.argmax(curve)+fidx_min
        curve = allmeans[ii][jj]    
        curve_smooth = uniform_filter1d(curve,size=15)
        curve_smooth_=curve_smooth.copy()
        dd = np.argwhere(frequs>200)
        curve_smooth_[0:dd[0][0]]=-40.
        peak_indices,_=find_peaks(curve_smooth_,width = 10,rel_height=2)
        allpeaks = frequs[peak_indices]
        FF[ii,:,jj]= allpeaks[:8]
        FFi[ii,:,jj]=peak_indices[:8]
        
            
        for ip in range(FF.shape[1]):    
            sines[ii,jj,:]=sines[ii,jj,:]+np.sin(2*np.pi*FF[ii,ip,jj]*timax)
            #transposed -->
            fo = FF[ii,ip,jj].copy()
            if fo<784:
                ft=fo
            elif fo<784*2:
                ft=fo/2
            elif fo<784*4:
                ft=fo/4   
            elif fo<784*8:
                ft=fo/8   
            elif fo<784*16:
                ft=fo/8   
            else:
                ft=fo/32
            sines_transp[ii,jj,:]=sines_transp[ii,jj,:]+np.sin(2*np.pi*ft*timax)
            
sines = sines/np.max(np.abs(sines))
sines_transp = sines_transp/np.max(np.abs(sines_transp))


# FIND FORMANTS



# for vv,vvtext in enumerate(vowels):
#     fig = go.Figure()
#     for ss,sstext in enumerate(names): 
#         curve = allmeans[vv][ss]
#         curve_smooth = uniform_filter1d(curve,size=15)
#         curve_smooth_=curve_smooth.copy()
#         dd = np.argwhere(frequs>200)
#         curve_smooth_[0:dd[0][0]]=-40.
#         peak_indices,_=find_peaks(curve_smooth_,width = 10,rel_height=2)
        #fig.add_traces(go.Scatter(x=frequs, y=curve))#, mode='lines'))
        
        # fig.add_trace(go.Scatter(x=frequs, y=curve_smooth,name=names[ss]))#, mode='lines'))
        # fig.add_trace(go.Scatter(x=frequs[peak_indices],y=curve_smooth[peak_indices],mode='markers'))
        # fig.update_layout(title=vowels[vv])
        # fig.write_html(vvtext+'.html')
        


  # COMMENT FROM HERE       
#vowels = np.array(['a','ae','e','i'])# wie in Ordner vowels sortiert   
# Seitenaufbau:
    # sidebar -> auswahl vowel
    # anzeige -> alle spektrogramme nebeneinander, alle means darunter in einem fenster
#from music21 import stream, chord, pitch, metadata
from music21 import pitch


from scipy.spatial.distance import cdist
# Funktion, um Frequenz in Noten und Cent-Abweichung zu konvertieren
def frequency_to_pitch_and_cents(frequency):
    p = pitch.Pitch()
    p.frequency = frequency
    cent_diff = round(p.microtone.cents)  # Cent-Abweichung
    return p, cent_diff
# def write_score(frequencyArray,title,names,neworder):
#     # Beispiel: Eingabe eines NumPy-Arrays mit 8 Frequenzen
#     #frequencies = np.array([440, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26])    
#     # Konvertiere Frequenzen in Noten und Cent-Abweichungen
#     numberOfChords = frequencyArray.shape[1]
    
#     score = stream.Score()
#     #partitur = stream.Score()
#     #part = stream.Part()
#     #measure = stream.Measure()     
#     for nf in range(numberOfChords):
        
#         frequencies = frequencyArray[:,nf]
#         pitches = []
#         cent_values = []
#         for freq in frequencies:
#             p, cent_diff = frequency_to_pitch_and_cents(freq)
#             pitches.append(p)
#             cent_values.append(cent_diff)    
#         # Erstelle einen Akkord aus den Noten
#         akkord = chord.Chord(pitches)    
#         # Füge Cent-Angaben als Text zu den einzelnen Noten im Akkord hinzu
#         for i, p in enumerate(akkord.pitches):
#             if cent_values[i] != 0:
#                 p.microtone = pitch.Microtone(cent_values[i])    
#         # Erstelle einen Notenstream (Partitur) und füge den Akkord hinzu
           
#         # Füge den Akkord (alle Noten gleichzeitig) in die Partitur ein
#         akkord.addLyric("".join([f"{cent}," for cent in cent_values]))  # Füge Cent-Text hinzu
#         #measure.append(akkord)
#         #part.append(measure)
#         #partitur.append(part) 
#         score.append(akkord)
#         # Zeige die Partitur an (falls MuseScore oder ein MusicXML-Viewer installiert ist)
#         #partitur.show()    
    
    
#     # Speichere die Partitur als MusicXML
#     #partitur.write('musicxml', fp=title+'.xml')
#     uberschrift = '_'
#     for ii in neworder:
#         uberschrift=uberschrift+names[ii]+'_'
#     #score.metadata = metadata.Metadata()
#     #score.metadata.title(uberschrift)
#     score.write('musicxml',fp=title+uberschrift+'.xml')
#frequencyArray = FF[0,:,:] # frequenzen x stimmen

def frequency_to_midi(frequencies):
    return 69 + 12 * np.log2(frequencies / 440.0)
def find_optimal_chord_order(frequency_matrix):
    # Wandle die Frequenzen in den gewünschten Raum um (MIDI-Noten oder Oktaven)
    #if method == 'midi':
    transformed_matrix = frequency_to_midi(frequency_matrix)
    #elif method == 'octave':
    #    transformed_matrix = frequency_to_octave(frequency_matrix)
    #else:
    #    raise ValueError("Method must be 'midi' or 'octave'")
    
    # Berechne die paarweisen Distanzen zwischen den Akkorden
    distances = cdist(transformed_matrix, transformed_matrix, metric='euclidean')
    # Finde die Reihenfolge, die die Gesamtdistanz minimiert (Nearest Neighbor Ansatz)
    num_chords = len(frequency_matrix)
    current_chord = 0
    order = [current_chord]
    remaining = set(range(1, num_chords))

    while remaining:
        next_chord = min(remaining, key=lambda x: distances[current_chord, x])
        order.append(next_chord)
        remaining.remove(next_chord)
        current_chord = next_chord

    # Rückgabe der optimalen Reihenfolge
    return order
def get_distance(frequency_matrix):
    transformed_matrix = frequency_to_midi(frequency_matrix)
    distances = cdist(transformed_matrix, transformed_matrix, metric='euclidean')
    return distances

   

f_array = np.zeros((len(vowels),len(names),8))
voweldistances = np.zeros((len(names),len(names),len(vowels)))
neworders = np.zeros((len(vowels),len(names)),dtype=int)
for iv,vv in enumerate(vowels):#iv=0
    for iis,ss in enumerate(names):#iis=0
        frequ_array = FF[iv,:,iis]
        for kk,ff in enumerate(frequ_array):
            if frequ_array[kk]<784:
                f_array[iv,iis,kk]=frequ_array[kk]
            elif frequ_array[kk]<784*2:
                f_array[iv,iis,kk]=frequ_array[kk]/2
            elif frequ_array[kk]<784*4:
                f_array[iv,iis,kk]=frequ_array[kk]/4  
            elif frequ_array[kk]<784*8:
                f_array[iv,iis,kk]=frequ_array[kk]/8      
            else:
                f_array[iv,iis,kk]=frequ_array[kk]/16 
        f_array[iv,iis,:]=np.sort(f_array[iv,iis,:])
        f_array[iv,iis,7]=f_array[iv,iis,7]/2
        f_array[iv,iis,5]=f_array[iv,iis,5]/2
        #f_array[iv,iis,7]=f_array[iv,iis,7]/2
    allchords = f_array[iv,:,:]   
    neworder = find_optimal_chord_order(allchords)
    neworders[iv,:]=neworder
    voweldistances[:,:,iv]=get_distance(allchords[neworder,:])        
    #write_score(np.transpose(allchords[neworder,:]),'vowel_'+vowels[iv],names,neworder)

# # TO HERE
import streamlit as st
import pandas as pd
st.set_page_config(layout="wide")

def streamlit_fun():
    # sidebar selectbox and radio
    whichVowel = st.sidebar.radio("Vowel: ",vowels)
    chosenVowel = np.argwhere(np.array(vowels)==whichVowel)
    chosenVowel = chosenVowel[0][0]
    #print('Chosen Vowel: '+vowels[chosenVowel])  
    # cols = st.columns(7)
    # for i,col in enumerate(cols):
    #     with col:
    #         st.write(specFig_matrix[i][chosenVowel])
    st.write(fig_specs[chosenVowel])
    cols = st.columns(len(names))
    for i,col in enumerate(cols):
        with col:
            st.write(names[i])
            #st.write(specFig_matrix[i][chosenVowel])
            st.audio(allaudio[chosenVowel][i],sample_rate=fs)
    figm = go.Figure()
    for ii,nn in enumerate(names):
        #print(ii)
        #print(nn)
        curve = uniform_filter1d(allmeans[chosenVowel][ii],size=15)
        figm.add_trace(go.Scatter(x=frequs, y=curve, mode='lines', name=nn))
        figm.add_trace(go.Scatter(x=FF[chosenVowel,:,ii],y=curve[FFi[chosenVowel,:,ii]],mode='markers'))
    st.write('Mean FFT. Click on the names of the singers to make them invisible.')
    idxt = np.argwhere(typicalv==whichVowel)
    peaks = formant_frequencies[idxt[0][0],:]
    #st.write('Typical values for vowel '+vowels[chosenVowel]+': '+str(peaks))
    figm.update_layout(xaxis=dict(range=[peaks[0]-100, peaks[7]+100]))
    st.write(figm)
    cols2 = st.columns(len(names))
    for i,col in enumerate(cols2):
        with col:
            st.write(names[i])
            #st.write(specFig_matrix[i][chosenVowel])
            st.audio(sines[chosenVowel,i,:],sample_rate=fs)
    for i,col in enumerate(cols2):
        with col:
            st.write(names[i]+' transposed')
            #st.write(specFig_matrix[i][chosenVowel])
            st.audio(sines_transp[chosenVowel,i,:],sample_rate=fs)
    
    data_rounded = np.round(voweldistances[:,:,chosenVowel], 1)
    # Convert the numpy array to a pandas DataFrame for better display in Streamlit
    sortednames = [names[i] for i in neworders[chosenVowel,:]]
    df = pd.DataFrame(data_rounded, columns=sortednames,index = sortednames)# noch falsch!!

    # Display the DataFrame in Streamlit
    st.write('Distances:')
    st.dataframe(df)

    image = Image.open('./data/images/a.png')
    st.image(image, caption='Sortierte Akkorde mit Cent-Abweichungen')

streamlit_fun()
    
 