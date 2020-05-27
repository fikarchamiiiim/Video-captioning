import json
import os
import argparse
import torch
import pretrainedmodels.utils as utils
import NLUtils
import numpy as np
import subprocess
import glob
import shutil
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from pretrainedmodels import resnet152
from googletrans import Translator

# TkInter
import tkinter as tk
import tkinter.filedialog as fd

# gTTS (Text to Speech)
from gtts import gTTS
from time import sleep
from playsound import playsound

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # GUI
        self.opt = json.load(open("data/save/opt_info.json"))

        # Label Judul
        self.judul = tk.Label(self.parent, text='Video Captioning',font=("Arial",30))
        self.judul.pack()
        self.judul.place(relx=0.5,rely=0.1, anchor=tk.CENTER)

        # Input File (Video Path)
        self.ent1=tk.Entry(self.parent,font=40, text='okokok')
        self.ent1.pack()
        self.ent1.place(bordermode=tk.INSIDE, height=30, y=100, x=50)

        # Button Cari File
        self.b1 = tk.Button(self.parent,text="Browse",font=40,command=self.browsefunc)
        self.b1.pack()
        self.b1.place(bordermode=tk.INSIDE, y=100, x=250)

        # Button Prediksi
        self.btnPrediksi = tk.Button(self.parent, text ="Prediksi", font=40,command=lambda : self.main(self.opt))
        self.btnPrediksi.pack()
        self.btnPrediksi.place(bordermode=tk.INSIDE, y=150, x=50)

        # Button Play
        self.btnPlay = tk.Button(self.parent, text='Mainkan', font=40, command=self.playVid)
        self.btnPlay.pack()
        self.btnPlay.place(bordermode=tk.INSIDE, y=150, x=150)

        # Label Prediksi
        self.labelPred = tk.Label(self.parent, text='Hasil Prediksi : ', font=40)
        self.labelPred.pack()
        self.labelPred.place(bordermode=tk.INSIDE, y=200, x=50)

        # Label Hasil
        self.hasilPred = tk.Label(self.parent, text='ini adalah hasil nya....', font=30)
        self.hasilPred.pack()
        self.hasilPred.place(bordermode=tk.INSIDE, y=230, x=50)

        # Label Prediksi(translate)
        self.labelTrans = tk.Label(self.parent, text='Translate : ', font=40)
        self.labelTrans.pack()
        self.labelTrans.place(bordermode=tk.INSIDE, y=260, x=50)

        # Label Hasil (translate)
        self.hasiltrans = tk.Label(self.parent, text='ini adalah hasil nya....', font=30)
        self.hasiltrans.pack()
        self.hasiltrans.place(bordermode=tk.INSIDE, y=290, x=50)

        # play TTS
        self.btnTts = tk.Button(self.parent, text ="TextToSpeech", font=40, command=self.playSpeech)
        self.btnTts.pack()
        self.btnTts.place(bordermode=tk.INSIDE, y=340, x=50)

        # credit
        self.credit = tk.Label(self.parent, text='by Zulfikar Chamim', font=16)
        self.credit.pack()
        self.credit.place(relx=1, rely=1, anchor=tk.SE)

        self.parent.wm_title("Video Captioning")
        self.parent.geometry("375x400")
    
        # Extend Google Translator Class
        self.translator = Translator()
        self.currentDirectory = os.getcwd()
        self.dir_temp = "tempGTTS"
        self.path = os.path.join(self.currentDirectory,self.dir_temp)
        


    """ browsefunc()
        untuk mengambil path video yang akan di prediksi """
    def browsefunc(self):
        self.ent1.configure(textvariable=tk.StringVar(gui, value=""))
        filename =fd.askopenfilename(filetypes=(("All files","*.*"),("MP4 Files","*.mp4"),("mkv Files","*.mkv")))
        self.ent1.insert(tk.END, filename) # add this

    """ playVid()
        Function untuk memulai (play) file yang sudah dipilih"""
    def playVid(self):
        from os import startfile
        video_path = self.ent1.get().replace("/","\\")
        startfile(video_path)

    def textToSpeech(self,text):
        textObj = gTTS(text=text, lang='en', slow=False)
        file_path = "{}\\audio.mp3".format(self.path)
        try:
            textObj.save(file_path)
        except OSError as e:
            os.mkdir(path)
            textObj.save(file_path)
    
    def playSpeech(self):
        playsound("{}\\audio.mp3".format(self.path))

    """ extract_image_feats(video_path)
        Fungsi untuk meng ekstrak fitru frame dari video
        
        - video_path | string | -> variable yang berisi path video yang akan diprediksi  """
    def extract_image_feats(self,video_path):
        self.hasilPred.configure(text="Membuat Prediksi....")
        model = resnet152(pretrained='imagenet')
        model = model.cuda()
        model.last_linear = utils.Identity()
        model.eval()
        C, H, W = 3, 224, 224
        load_image_fn = utils.LoadTransformImage(model)
        dst = os.path.join(video_path.split('\\')[0], 'info')
        if os.path.exists(dst):
                print(" Menghapus Direktori: " + dst + "\\")
                shutil.rmtree(dst)
        os.makedirs(dst)
        with open(os.devnull, "w") as ffmpeg_log:
            command = 'ffmpeg -i ' + video_path + ' -vf scale=400:300 ' + '-qscale:v 2 '+ '{0}/%06d.jpg'.format(dst)
            subprocess.call(command, shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)
        list_image = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        samples = np.round(np.linspace(0, len(list_image) - 1, 80))
        list_image = [list_image[int(sample)] for sample in samples]
        images = torch.zeros((len(list_image), C, H, W))
        for i in range(len(list_image)):
            img = load_image_fn(list_image[i])
            images[i] = img
        with torch.no_grad():
            image_feats = model(images.cuda().squeeze())
        image_feats = image_feats.cpu().numpy()
        for file in os.listdir(dst):
            if file.endswith('.jpg'):
                os.remove(os.path.join(dst, file))

        return image_feats


    def main(self, opt): 
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        video_path = self.ent1.get().replace("/","\\")
        image_feats = self.extract_image_feats(video_path)
        image_feats = torch.from_numpy(image_feats).type(torch.FloatTensor).unsqueeze(0)

        encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"], bidirectional=bool(opt["bidirectional"]),
                                input_dropout_p=opt["input_dropout_p"], rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(16860, opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                                input_dropout_p=opt["input_dropout_p"],
                                rnn_dropout_p=opt["rnn_dropout_p"], bidirectional=bool(opt["bidirectional"]))
        model = S2VTAttModel(encoder, decoder).cuda()
        model.load_state_dict(torch.load("data/save/model_800.pth"))
        model.eval()
        opt = dict()
        opt['child_sum'] = True
        opt['temporal_attention'] = True
        opt['multimodel_attention'] = True
        with torch.no_grad():
            _, seq_preds = model(image_feats.cuda(), mode='inference', opt=opt)
        vocab = json.load(open('data/info.json'))['ix_to_word']
        self.sent = NLUtils.decode_sequence(vocab, seq_preds)
        hasil = self.translator.translate(self.sent[0],dest='id')
        print(self.sent[0])
        self.hasilPred.configure(text=self.sent[0])
        self.hasiltrans.configure(text=hasil.text)
        # coba = self.sent[0]
        self.textToSpeech(self.sent[0])
        del seq_preds
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    gui = tk.Tk()
    MainApplication(gui).pack(side="top", fill="both", expand=True)
    gui.mainloop()

    