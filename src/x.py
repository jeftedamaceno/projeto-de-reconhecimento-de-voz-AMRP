import os
import json
import numpy as np
import tensorflow as tf
import sounddevice as sd
import keyboard
import time

# Certifique-se de que a classe de atenção está declarada exatamente igual
@tf.keras.utils.register_keras_serializable(package="Custom")
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = inputs * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()

# ==========================================
# 1. PARÂMETROS E CARREGAMENTO ATUALIZADOS
# ==========================================
SAMPLE_RATE = 16000
DURATION = 1.5  # Atualizado para 1.5 segundos conforme o modelo novo
CHANNELS = 1     # Mono

MODEL_PATH = "modelo_hibrido_1s5_atencao_v2.h5"
LABEL_PATH = "labels_1s5_atencao_v2.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    print(f"ERRO: Certifique-se de que '{MODEL_PATH}' e '{LABEL_PATH}' estão na mesma pasta!")
    exit()

print("Carregando o modelo híbrido v2 (1.5s)...")
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={"AttentionLayer": AttentionLayer}
)

print("Carregando mapa de labels atualizado...")
with open(LABEL_PATH, "r") as f:
    label_map = json.load(f)
    
inv_label_map = {v: k for k, v in label_map.items()}

# ==========================================
# 2. LOOP DE EXECUÇÃO INTERATIVA
# ==========================================
print("\n========================================================")
print(" SISTEMA DE VALIDAÇÃO INTERATIVA PRONTO (1.5 SEGUNDOS)")
print(" Comandos esperados:", list(label_map.keys()))
print(" --> Pressione a tecla 'G' para iniciar uma gravação.")
print(" --> Pressione 'ESC' para sair do programa.")
print("========================================================")

while True:
    # Captura a saída do programa caso o usuário queira encerrar
    if keyboard.is_pressed('esc'):
        print("\nFinalizando validador de áudio...")
        break
        
    # Verifica se a tecla 'g' foi profissionalizada
    if keyboard.is_pressed('g'):
        # Aguarda o usuário soltar a tecla para não capturar o barulho físico do clique no teclado
        while keyboard.is_pressed('g'):
            time.sleep(0.05)
            
        print("\n>>> [GRAVANDO] Fale agora por 1.5s...")
        # Captura usando a nova DURATION de 1.5s
        audio_float32 = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        print(">>> Gravação concluída!")
        
        audio_raw = audio_float32.flatten()
        
        # --------------------------------------------------
        # TRATAMENTO E REPRODUÇÃO: OUVIR O AUDIO
        # --------------------------------------------------
        # Converte para a escala int16 
        audio_int16 = (audio_raw * 32767.0).astype(np.float32)
        
        print(">>> Reproduzindo o áudio capturado para checagem...")
        sd.play(audio_raw, SAMPLE_RATE)
        sd.wait()
        
        print(">>> Processando inferência no modelo...")
        
        # --------------------------------------------------
        # PRÉ-PROCESSAMENTO E INFERÊNCIA
        # --------------------------------------------------
        # Aplica o Z-score exatamente igual ao pipeline de treino
        if np.std(audio_int16) > 0:
            audio_preprocessed = (audio_int16 - np.mean(audio_int16)) / np.std(audio_int16)
        else:
            audio_preprocessed = audio_int16
            
        # Formata para a entrada do modelo: (1, 24000, 1) já que 16000 * 1.5 = 24000
        audio_input = np.expand_dims(audio_preprocessed, axis=(0, -1))
        
        # Executa a predição
        predicoes = model.predict(audio_input, verbose=0)
        classe_index = np.argmax(predicoes[0])
        confianca = predicoes[0][classe_index] * 100
        
        # Exibe o resultado final de forma destacada
        print("\n" + "="*45)
        print(f" COMANDO DETECTADO: {inv_label_map[classe_index].upper()}")
        print(f" Grau de Confiança: {confianca:.2f}%")
        print("="*45)
        print("\nPronto para o próximo teste! Pressione 'G' novamente ou 'ESC' para sair...")
        
    time.sleep(0.1)