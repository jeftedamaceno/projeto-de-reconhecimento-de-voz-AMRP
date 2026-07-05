import os
import glob
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model

@tf.keras.utils.register_keras_serializable(package="Custom")
class AttentionLayer(tf.keras.layers.Layer):
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

BASE_DIR = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"
PASTA_DADOS_ORIGINAIS = os.path.join(BASE_DIR, "dataset_final")
PASTA_AUDITORIA_ERROS = os.path.join(BASE_DIR, "auditoria_erros_reais")
OUTPUT_DIR = os.path.join(BASE_DIR, "comparacao_1s5_modelos")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 1.5
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)
CLASSES = ["direita", "esquerda", "siga", "pare", "voltar", "ruido"]
label_map = {classe: idx for idx, classe in enumerate(CLASSES)}

X_base = []
y_base = []

for classe in CLASSES:
    caminho_classe = os.path.join(PASTA_DADOS_ORIGINAIS, classe)
    if not os.path.isdir(caminho_classe):
        continue
    arquivos = glob.glob(os.path.join(caminho_classe, "*.wav"))
    for arq in arquivos:
        try:
            audio, sr = sf.read(arq)
            if len(audio) < TOTAL_SAMPLES:
                audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)), 'constant')
            else:
                audio = audio[:TOTAL_SAMPLES]
            X_base.append(audio)
            y_base.append(label_map[classe])
        except Exception:
            continue

X_base = np.array(X_base)
y_base = np.array(y_base)

X_auditoria = []
y_auditoria = []

arquivos_auditoria = glob.glob(os.path.join(PASTA_AUDITORIA_ERROS, "*.wav"))
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
            X_auditoria.append(audio_aug)
            y_auditoria.append(label_map[classe_alvo])
    except ValueError:
        continue

X_auditoria = np.array(X_auditoria)
y_auditoria = np.array(y_auditoria)

def padronizar_sinais(X_input):
    X_proc = []
    for sinal in X_input:
        std_sinal = np.std(sinal)
        if std_sinal > 1e-6:
            X_proc.append((sinal - np.mean(sinal)) / std_sinal)
        else:
            X_proc.append(sinal - np.mean(sinal))
    return np.expand_dims(np.array(X_proc), axis=-1)

X_base_proc = padronizar_sinais(X_base)
y_base_cat = tf.keras.utils.to_categorical(y_base, num_classes=len(CLASSES))

X_tr_b, X_val_b, y_tr_b, y_val_b = train_test_split(X_base_proc, y_base_cat, test_size=0.2, random_state=42, stratify=y_base)

if len(X_auditoria) > 0:
    X_aud_proc = padronizar_sinais(X_auditoria)
    y_aud_cat = tf.keras.utils.to_categorical(y_auditoria, num_classes=len(CLASSES))
    
    X_tr_a, X_val_a, y_tr_a, y_val_a = train_test_split(X_aud_proc, y_aud_cat, test_size=0.2, random_state=42, stratify=y_auditoria)
    
    X_val_v2 = np.concatenate((X_val_b, X_val_a), axis=0)
    y_val_v2 = np.concatenate((y_val_b, y_val_a), axis=0)
else:
    X_val_v2 = X_val_b
    y_val_v2 = y_val_b

custom_scope = {'AttentionLayer': AttentionLayer}
model_v1 = load_model(os.path.join(BASE_DIR, "modelo_hibrido_1s5_atencao.h5"), custom_objects=custom_scope)
model_v2 = load_model(os.path.join(BASE_DIR, "modelo_hibrido_1s5_atencao_v2.h5"), custom_objects=custom_scope)

preds_v1 = model_v1.predict(X_val_b, verbose=0)
preds_v2 = model_v2.predict(X_val_v2, verbose=0)

y_true_v1 = np.argmax(y_val_b, axis=1)
y_pred_v1 = np.argmax(preds_v1, axis=1)

y_true_v2 = np.argmax(y_val_v2, axis=1)
y_pred_v2 = np.argmax(preds_v2, axis=1)

plt.figure(figsize=(15, 10))
for i, classe in enumerate(CLASSES):
    fpr_v1, tpr_v1, _ = roc_curve(y_val_b[:, i], preds_v1[:, i])
    auc_v1 = auc(fpr_v1, tpr_v1)
    
    fpr_v2, tpr_v2, _ = roc_curve(y_val_v2[:, i], preds_v2[:, i])
    auc_v2 = auc(fpr_v2, tpr_v2)
    
    plt.subplot(2, 3, i+1)
    plt.plot(fpr_v1, tpr_v1, label=f'V1 Original (AUC = {auc_v1:.3f})', color='darkorange', lw=2)
    plt.plot(fpr_v2, tpr_v2, label=f'V2 Ampliado (AUC = {auc_v2:.3f})', color='navy', linestyle='--', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'Classe: {classe.upper()}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

plt.suptitle('Curvas ROC Comparativas (Ambientes de Validacao Respectivos)', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparativo_curvas_roc.png"), dpi=300)
plt.show()

conf_v1 = [np.mean(preds_v1[np.where(y_true_v1 == idx)[0], idx]) * 100 if len(np.where(y_true_v1 == idx)[0]) > 0 else 0 for idx in range(len(CLASSES))]
conf_v2 = [np.mean(preds_v2[np.where(y_true_v2 == idx)[0], idx]) * 100 if len(np.where(y_true_v2 == idx)[0]) > 0 else 0 for idx in range(len(CLASSES))]

x = np.arange(len(CLASSES))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, conf_v1, width, label='Modelo V1 (Base Limpa)', color='coral')
ax.bar(x + width/2, conf_v2, width, label='Modelo V2 (Base Expandida)', color='teal')
ax.set_ylabel('Confianca Softmax Media (%)')
ax.set_title('Seguranca de Predicao por Classe', fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c.upper() for c in CLASSES])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparativo_confianca_classes.png"), dpi=300)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(confusion_matrix(y_true_v1, y_pred_v1, labels=range(len(CLASSES))), annot=True, fmt='d', cmap='Oranges', xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
axes[0].set_title('Matriz de Confusao - Modelo V1', fontsize=12, weight='bold')
axes[0].set_xlabel('Predito')
axes[0].set_ylabel('Real')

sns.heatmap(confusion_matrix(y_true_v2, y_pred_v2, labels=range(len(CLASSES))), annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
axes[1].set_title('Matriz de Confusao - Modelo V2', fontsize=12, weight='bold')
axes[1].set_xlabel('Predito')
axes[1].set_ylabel('Real')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparativo_matrizes_confusao.png"), dpi=300)
plt.show()

print("\n--- METRICAS MODELO V1 ---")
print(classification_report(y_true_v1, y_pred_v1, target_names=CLASSES))
print("\n--- METRICAS MODELO V2 ---")
print(classification_report(y_true_v2, y_pred_v2, target_names=CLASSES))