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


SAMPLE_RATE = 16000
DURATION = 1  # 1 segundo
CHANNELS = 1     

print("Carregando o modelo híbrido...")
model = tf.keras.models.load_model(
    "modelo_hibrido_com_atencao.h5", 
    custom_objects={"AttentionLayer": AttentionLayer}
)

print("Carregando mapa de labels...")
with open("label_map_hibrido.json", "r") as f:
    label_map = json.load(f)
    
inv_label_map = {v: k for k, v in label_map.items()}


print("\n========================================================")
print(" SISTEMA DE VALIDAÇÃO INTERATIVA PRONTO")
print(" Comandos esperados:", list(label_map.keys()))
print(" --> Pressione a tecla 'G' para iniciar uma gravação.")
print(" --> Pressione 'ESC' para sair do programa.")
print("========================================================")

while True:
   
    if keyboard.is_pressed('esc'):
        print("\nFinalizando validador de áudio...")
        break
        
   
    if keyboard.is_pressed('g'):
       
        while keyboard.is_pressed('g'):
            time.sleep(0.05)
            
        print("\n>>> [GRAVANDO] Fale agora...")
   
        audio_float32 = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        print(">>> Gravação concluída!")
        
        audio_raw = audio_float32.flatten()
        
        
    
        audio_int16 = (audio_raw * 32767.0).astype(np.float32)
        
        print(">>> Reproduzindo o áudio capturado para checagem...")

        sd.play(audio_raw, SAMPLE_RATE)
        sd.wait()
        
      
        print(">>> Processando inferência no modelo...")
        
        
      
        
        # Aplica o Z-score exatamente igual ao pipeline de treino
        if np.std(audio_int16) > 0:
            audio_preprocessed = (audio_int16 - np.mean(audio_int16)) / np.std(audio_int16)
        else:
            audio_preprocessed = audio_int16
            
      
        audio_input = audio_preprocessed.reshape(1, SAMPLE_RATE, 1)
      
        predicoes = model.predict(audio_input, verbose=0)
        classe_index = np.argmax(predicoes[0])
        confianca = predicoes[0][classe_index] * 100
        

        print("\n" + "="*45)
        print(f" COMANDO DETECTADO: {inv_label_map[classe_index].upper()}")
        print(f" Grau de Confiança: {confianca:.2f}%")
        print("="*45)
        print("\nPronto para o próximo teste! Pressione 'G' novamente ou 'ESC' para sair...")
 
    time.sleep(0.1)