import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from pretrainedmodels import resnet152
import pretrainedmodels.utils as utils
import NLUtils
import numpy as np
import subprocess
import glob
import shutil

from googletrans import Translator

# TkInter
from tkinter import *
import tkinter.messagebox as mb
import tkinter.filedialog as fd

gui = Tk()

# Extend Google Translator Class
translator = Translator()

""" browsefunc()
    untuk mengambil path video yang akan di prediksi """
def browsefunc():
    ent1.configure(textvariable=StringVar(gui, value=""))
    filename =fd.askopenfilename(filetypes=(("All files","*.*"),("MP4 Files","*.mp4"),("mkv Files","*.mkv")))
    ent1.insert(END, filename) # add this

""" playVid()
    Function untuk memulai (play) file yang sudah dipilih"""
def playVid():
    from os import startfile
    video_path = ent1.get().replace("/","\\")
    startfile(video_path)

""" extract_image_feats(video_path)
    Fungsi untuk meng ekstrak fitru frame dari video
    
    - video_path | string | -> variable yang berisi path video yang akan diprediksi  """
def extract_image_feats(video_path):
    hasilPred.configure(text="Membuat Prediksi....")
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


def main(opt):    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    video_path = ent1.get().replace("/","\\")
    image_feats = extract_image_feats(video_path)
    image_feats = torch.from_numpy(image_feats).type(torch.FloatTensor).unsqueeze(0)

    encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"], bidirectional=bool(opt["bidirectional"]),
                             input_dropout_p=opt["input_dropout_p"], rnn_dropout_p=opt["rnn_dropout_p"])
    decoder = DecoderRNN(16860, opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                            input_dropout_p=opt["input_dropout_p"],
                            rnn_dropout_p=opt["rnn_dropout_p"], bidirectional=bool(opt["bidirectional"]))
    model = S2VTAttModel(encoder, decoder).cuda()
    model.load_state_dict(torch.load("data/save/model_500.pth"))
    model.eval()
    opt = dict()
    opt['child_sum'] = True
    opt['temporal_attention'] = True
    opt['multimodel_attention'] = True
    with torch.no_grad():
        _, seq_preds = model(image_feats.cuda(), mode='inference', opt=opt)
    vocab = json.load(open('data/info.json'))['ix_to_word']
    sent = NLUtils.decode_sequence(vocab, seq_preds)
    hasil = translator.translate(sent[0],dest='id')
    print(sent[0])
    hasilPred.configure(text=sent[0])
    hasiltrans.configure(text=hasil.text)
    del seq_preds
    torch.cuda.empty_cache()

# GUI

opt = json.load(open("data/save/opt_info.json"))

# Label Judul
judul = Label(gui, text='Video Captioning',font=("Arial",30))
judul.pack()
judul.place(relx=0.5,rely=0.1, anchor=CENTER)

# Input File (Video Path)
ent1=Entry(gui,font=40, text='okokok')
ent1.pack()
ent1.place(bordermode=INSIDE, height=30, y=100, x=50)

# Button Cari File
b1=Button(gui,text="Browse",font=40,command=browsefunc)
b1.pack()
b1.place(bordermode=INSIDE, y=100, x=250)

# Button Prediksi
btnPrediksi = Button(gui, text ="Prediksi", font=40,command=lambda : main(opt))
btnPrediksi.pack()
btnPrediksi.place(bordermode=INSIDE, y=150, x=50)

# Button Play
btnPlay = Button(gui, text='Mainkan', font=40, command=playVid)
btnPlay.pack()
btnPlay.place(bordermode=INSIDE, y=150, x=150)

# Label Prediksi
labelPred = Label(gui, text='Hasil Prediksi : ', font=40)
labelPred.pack()
labelPred.place(bordermode=INSIDE, y=200, x=50)

# Label Hasil
hasilPred = Label(gui, text='ini adalah hasil nya....', font=30)
hasilPred.pack()
hasilPred.place(bordermode=INSIDE, y=230, x=50)

# Label Prediksi(translate)
labelTrans = Label(gui, text='Translate : ', font=40)
labelTrans.pack()
labelTrans.place(bordermode=INSIDE, y=260, x=50)

# Label Hasil (translate)
hasiltrans = Label(gui, text='ini adalah hasil nya....', font=30)
hasiltrans.pack()
hasiltrans.place(bordermode=INSIDE, y=290, x=50)

# credit
credit = Label(gui, text='by Zulfikar Chamim', font=16)
credit.pack()
credit.place(relx=1, rely=1, anchor=SE)

gui.wm_title("Video Captioning")
gui.geometry("375x400")
gui.mainloop()