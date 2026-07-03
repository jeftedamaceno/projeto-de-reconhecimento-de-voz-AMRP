# Diagnóstico do dataset
# Execute este arquivo para verificar onde as amostras estão sendo perdidas.

import os, wave, numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

BASE_DIR=r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"
AUDIO_ORIGINAL=os.path.join(BASE_DIR,"dataset_vozes_old")
CLASSES=["direita","esquerda","siga","pare","voltar"]
label_map={c:i for i,c in enumerate(CLASSES)}
TOTAL_SAMPLES=16000*3//2

def carregar(p):
    with wave.open(p,"rb") as w:
        a=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)
        if w.getnchannels()>1:
            a=a.reshape(-1,w.getnchannels()).mean(axis=1)
        return a

X=[];y=[];cont={c:0 for c in CLASSES}
for raiz,_,arqs in os.walk(AUDIO_ORIGINAL):
    pasta=os.path.basename(raiz).lower()
    if pasta in label_map:
        print(pasta,"->",len([f for f in arqs if f.endswith(".wav")]),"wav(s)")
        for arq in arqs:
            if arq.endswith(".wav"):
                a=carregar(os.path.join(raiz,arq))
                a=a[:TOTAL_SAMPLES] if len(a)>TOTAL_SAMPLES else np.pad(a,(0,TOTAL_SAMPLES-len(a)))
                X.append(a); y.append(label_map[pasta]); cont[pasta]+=1
print("\nCarregados:",cont)
X=np.expand_dims(np.array(X),-1)
y=np.array(y)
print("Total:",len(y))
for i,c in enumerate(CLASSES):
    print(c,np.sum(y==i))
yc=to_categorical(y,len(CLASSES))
Xtr,Xtmp,Ytr,Ytmp=train_test_split(X,yc,test_size=0.3,random_state=42,stratify=y)
Xv,Xte,Yv,Yte=train_test_split(Xtmp,Ytmp,test_size=0.5,random_state=42,stratify=np.argmax(Ytmp,1))
for nome,D in [("Treino",Ytr),("Validação",Yv),("Teste",Yte)]:
    print("\n"+nome)
    lab=np.argmax(D,1)
    for i,c in enumerate(CLASSES):
        print(c,np.sum(lab==i))
