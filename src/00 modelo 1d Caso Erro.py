import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, Layer
from tensorflow.keras.utils import to_categorical

@tf.keras.utils.register_keras_serializable(package="Custom")
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = inputs * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()

SAMPLE_RATE = 16000
DURATION = 1.5
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)

PASTA_DADOS_ORIGINAIS = "dataset_final"  
PASTA_AUDITORIA_ERROS = "auditoria_erros_reais" 

def aplicar_data_augmentation(audio):
    escolha = np.random.choice(["ruido_fundo", "time_shift", "ganho", "nenhum"])
    if escolha == "ruido_fundo":
        ruido = np.random.normal(0, 0.005, len(audio))
        return audio + ruido
    elif escolha == "time_shift":
        deslocamento = np.random.randint(-3200, 3200)
        return np.roll(audio, deslocamento)
    elif escolha == "ganho":
        fator_ganho = np.random.uniform(0.7, 1.2)
        return audio * fator_ganho
    return audio

print("Analisando e carregando os conjuntos de dados...")

X = []
y_labels = []

classes_encontradas = sorted(os.listdir(PASTA_DADOS_ORIGINAIS))
label_map = {classe: idx for idx, classe in enumerate(classes_encontradas)}
inv_label_map = {idx: classe for classe, idx in label_map.items()}

contagem_por_classe = {classe: 0 for classe in classes_encontradas}

for classe in classes_encontradas:
    caminho_classe = os.path.join(PASTA_DADOS_ORIGINAIS, classe)
    arquivos = glob.glob(os.path.join(caminho_classe, "*.wav"))
    for arq in arquivos:
        audio, sr = sf.read(arq)
        if len(audio) < TOTAL_SAMPLES:
            audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)), 'constant')
        else:
            audio = audio[:TOTAL_SAMPLES]
        X.append(audio)
        y_labels.append(label_map[classe])
        contagem_por_classe[classe] += 1

arquivos_auditoria = glob.glob(os.path.join(PASTA_AUDITORIA_ERROS, "*.wav"))
print(f"Incorporando {len(arquivos_auditoria)} audios corrigidos da Auditoria...")

for arq in arquivos_auditoria:
    nome_arq = os.path.basename(arq)
    try:
        partes = nome_arq.split("_")
        idx_alvo = partes.index("ALVO") + 1
        classe_alvo = partes[idx_alvo].lower()
        
        if classe_alvo in label_map:
            audio, sr = sf.read(arq)
            if len(audio) < TOTAL_SAMPLES:
                audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)), 'constant')
            else:
                audio = audio[:TOTAL_SAMPLES]
            audio_aug = aplicar_data_augmentation(audio)
            X.append(audio_aug)
            y_labels.append(label_map[classe_alvo])
            contagem_por_classe[classe_alvo] += 1
    except ValueError:
        continue

print("\nCONTAGEM METROLOGICA DA BASE DE DADOS:")
for classe, qtd in contagem_por_classe.items():
    print(f"   Classe '{classe.upper()}': {qtd} amostras.")

X = np.array(X)
y = np.array(y_labels)

X_processado = []
for sinal in X:
    std_sinal = np.std(sinal)
    if std_sinal > 1e-6:
        X_processado.append((sinal - np.mean(sinal)) / std_sinal)
    else:
        X_processado.append(sinal - np.mean(sinal))

X_processado = np.expand_dims(np.array(X_processado), axis=-1)
y_categorical = to_categorical(y, num_classes=len(label_map))

X_train, X_test, y_train, y_test = train_test_split(X_processado, y_categorical, test_size=0.2, random_state=42, stratify=y)

entradas = Input(shape=(TOTAL_SAMPLES, 1))
x = Conv1D(64, kernel_size=9, activation='relu', padding='same')(entradas)
x = MaxPooling1D(pool_size=4)(x)
x = Dropout(0.3)(x)

x = Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
x = MaxPooling1D(pool_size=4)(x)
x = Dropout(0.3)(x)

x = LSTM(64, return_sequences=True)(x)
x = AttentionLayer()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
saidas = Dense(len(label_map), activation='softmax')(x)

model_v2 = Model(inputs=entradas, outputs=saidas)
model_v2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nIniciando o refinamento do Modelo V2...")
history = model_v2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32, verbose=1)

model_v2.save("modelo_hibrido_1s5_atencao_v2.h5")
with open("labels_1s5_atencao_v2.json", "w") as f:
    json.dump(label_map, f)

y_pred = model_v2.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\n" + "="*60)
print("             RELATORIO TECNICO DE DESEMPENHO (V2)             ")
print("="*60)
print(classification_report(y_true_classes, y_pred_classes, target_names=list(label_map.keys())))
print("="*60)

plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Acuracia Treino (V2)', color='darkblue', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Acuracia Validacao (V2)', color='crimson', linestyle='--', linewidth=2)
plt.title('Evolucao do Aprendizado Metrologico - Modelo V2')
plt.xlabel('Epocas')
plt.ylabel('Acuracia')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizacao_curva_aprendizado.png')
plt.show()

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_map.keys()), yticklabels=list(label_map.keys()))
plt.title('Matriz de Confusao Computacional - Modelo V2')
plt.ylabel('Classe Real Humana')
plt.xlabel('Classe Predita pela IA')
plt.tight_layout()
plt.savefig('visualizacao_matriz_confusao.png')
plt.show()

print("\nGraficos e matrizes salvos com sucesso no seu diretorio.")