import sys
from PyQt5 import QtWidgets, QtGui, QtCore, QtMultimedia
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from lib import sound_file_name, decode
import requests
import time
import wave
import contextlib

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi("project.ui",self)
        #self.chooseFile = self.findChild(QPushButton, "chooseFile")
        self.chooseFile.clicked.connect(self.browsefiles)
        self.transcribe.clicked.connect(self.trans)
        # self.anonymise.activated.connect(self.ner)

        self.play_button.clicked.connect(self.play_audio)
        self.player = QMediaPlayer()
        
        self.none.toggled.connect(self.ner)
        self.personal_info.toggled.connect(self.ner)

        self.customEnText = None
        self.customArText = None
        self.customMedicalText = None

        self.show()

    def browsefiles(self):
        self.fname, self.file_format = QFileDialog.getOpenFileName(self, 'Load audio', 'audio_files/', 'All files (*);;wav Files (*.wav)')

        # If the file is loaded successfully
        if self.fname != "":
            self.audio_name.setText(self.fname.split('/')[-1]) # file_name
            #print(f"audio path: {fname}")
            # remove text to avoid errors on changing anonymous mode
            self.enText.setText(None)
            self.arText.setText(None)
            self.medicalText.setText(None)
            self.customEnText = None
            self.customArText = None
            self.customMedicalText = None

            #clean 
            for row in range(1, 6):
                key_att = getattr(self, f'entity_{row}{0}')
                val_att = getattr(self, f'entity_{row}{1}')
                key_att.setText(None)
                val_att.setText(None)

            # calculate audio_duration
            with contextlib.closing(wave.open(self.fname,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                # print("Audio_duration ========== ", duration)

    def play_audio(self):
        print("playing audio.....")
        url = QUrl.fromLocalFile(self.fname)
        content = QMediaContent(url)

        self.player.setMedia(content)
        self.player.play()


    def trans(self):
        if not hasattr(self, 'fname'):
            # TODO: print error msg
            print('no file selected')
            return
        sound_file_name(self.fname)

        start = time.time()
        med_sentence = decode()
        # print("KALDI Time ======== ", time.time() - start)

        self.medicalText.setText(med_sentence)

        self.customMedicalText = med_sentence
        transcription = requests.post(
            'http://localhost:5000/transcribe',
            json={'path': self.fname}
        )
        
        transJson = transcription.json()
        self.enText.setText(transJson['english']) 
        self.arText.setText(transJson['arabic']) 
        # set custom texts for ner functions
        self.customEnText = transJson['english']
        self.customArText = transJson['arabic']

        # run anonymes function in case there's a mode already selected
        self.ner()
    

    def ner(self):
        # mode  = self.anonymise.currentText()
        mode = None
        enText = self.customEnText
        arText = self.customArText
        medText = self.customMedicalText

        if enText is None or arText is None or medText is None:
            # TODO: display error msg
            print('no text found')
            return

        # if mode == "None" :
        if self.none.isChecked():
            mode = "none"
            nerMedText = requests.post(
                'http://localhost:5000/ner',
                json={'text': medText, 'mode': mode}
            )
            nerMedJson = nerMedText.json()
            self.medicalText.setText(self.customMedicalText)
            print('medicalEntities:')
            print(nerMedJson['medicalEntities'])
            
            entities_dict = nerMedJson['medicalEntities']
            print("entities_dict")
            print(entities_dict)

            #clean 
            for row in range(1, 6):
                key_att = getattr(self, f'entity_{row}{0}')
                val_att = getattr(self, f'entity_{row}{1}')
                key_att.setText(None)
                val_att.setText(None)



            for row, (key, value) in enumerate(entities_dict.items(), start=1):
                key_att = getattr(self, f'entity_{row}{0}', None)
                val_att = getattr(self, f'entity_{row}{1}', None)
                if not key_att or not val_att:
                    continue
                key_att.setText(key)
                val_att.setText(value)

            # self.entity_1.setText(list(entities_dict.keys())[0])

            return
    
        elif self.personal_info.isChecked():
            mode = "personal_info"

            nerMedText = requests.post(
                'http://localhost:5000/ner',
                json={'text': medText, 'mode': mode}
            )
            nerMedJson = nerMedText.json()
            self.medicalText.setText(nerMedJson['text'])
            print('personalEntities:')
            print(nerMedJson['personalEntities'])

        
app = QApplication(sys.argv)
ui = MainWindow()
sys.exit(app.exec_())