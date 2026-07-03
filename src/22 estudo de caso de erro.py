import os
import json
import time
import csv
import numpy as np
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import keras
import keyboard
import soundfile as sf

# --- CONFIGURAÇÕES DE INFRAESTRUTURA ---
PASTA_EXPERIMENTO = "experimentos_ia"
MODEL_PATH = os.path.join(PASTA_EXPERIMENTO, "modelo_hibrido_1s5_atencao.h5")
LABELS_PATH = os.path.join(PASTA_EXPERIMENTO, "labels_1s5_atencao.json")

PASTA_AUDITORIA_ERROS = "auditoria_erros_reais"
ARQUIVO_LOG_CSV = os.path.join(PASTA_AUDITORIA_ERROS, "relatorio_auditoria.csv")

os.makedirs(PASTA_AUDITORIA_ERROS, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 1.5         
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)  # 24.000 amostras
THRESHOLD_CONFIAVEL = 0.75  

# =====================================================================
# DEFINIÇÃO DA CAMADA DE ATENÇÃO PERSONALIZADA (CUSTOM OBJECT)
# =====================================================================
@keras.saving.register_keras_serializable(package="Custom")
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = tf.squeeze(tf.tanh(tf.matmul(x, self.W) + self.b), axis=-1)
        at = tf.nn.softmax(et)
        at = tf.expand_dims(at, axis=-1)
        output = x * at
        return tf.reduce_sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionLayer, self).get_config()

def calcular_entropia_vetorial(probabilidades):
    return -np.sum(probabilidades * np.log2(probabilidades + 1e-9))

# =====================================================================
# PIPELINE DE TRATAMENTO DIGITAL
# =====================================================================
def processar_audio_microfone_direto(audio):
    if len(audio) < TOTAL_SAMPLES:
        audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)), 'constant')
    else:
        audio = audio[:TOTAL_SAMPLES]
        
    desvio = np.std(audio)
    if desvio > 1e-6:
        audio_tratado = (audio - np.mean(audio)) / desvio
    else:
        audio_tratado = audio - np.mean(audio)
        
    return np.expand_dims(audio_tratado, axis=-1)

# =====================================================================
# GERENCIADOR DO RELATÓRIO METROLÓGICO (CSV)
# =====================================================================
def inicializar_csv():
    if not os.path.exists(ARQUIVO_LOG_CSV):
        with open(ARQUIVO_LOG_CSV, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Usuario", "Nome_Arquivo", "Classe_Alvo_Humana", 
                "Classe_Predita_IA", "Confianca_IA", "Entropia_Incerteza", 
                "Motivo_Falha", "Energia_Sinal_RMS"
            ])

def registrar_falha_csv(usuario, nome_arq, alvo, predito, conf, entropia, motivo, rms):
    with open(ARQUIVO_LOG_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), usuario, nome_arq, alvo, 
            predito, f"{conf*100:.2f}%", f"{entropia:.4f}", motivo, f"{rms:.5f}"
        ])

# =====================================================================
# INICIALIZAÇÃO DO ECOSSISTEMA
# =====================================================================
print("🔄 Carregando ecossistema Híbrido Temporal com Atenção (.h5)...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Erro: Coloque os arquivos na pasta '{PASTA_EXPERIMENTO}'.")

custom_dict = {"AttentionLayer": AttentionLayer}
model = load_model(MODEL_PATH, custom_objects=custom_dict)

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)
inv_map = {v: k for k, v in label_map.items()}
inicializar_csv()

# Detecta automaticamente o nome da classe de ruído/descarte do seu modelo
CLASSE_RUIDO_MODELO = None
for classe_possivel in ["ruido", "background", "silencio", "desconhecido"]:
    if classe_possivel in label_map:
        CLASSE_RUIDO_MODELO = classe_possivel
        break

if not CLASSE_RUIDO_MODELO:
    # Se não achar nenhuma com esses nomes, assume a última classe do mapeamento como descarte
    CLASSE_RUIDO_MODELO = list(label_map.keys())[-1]

# =====================================================================
# PIPELINE DE INFERÊNCIA E CAPTURA COLETIVA CONTÍNUA
# =====================================================================
def executar_pipeline_coleta(classe_alvo, usuario, modo_ruido_oov=False):
    # Ajusta a exibição do alvo no terminal caso seja palavra fora do vocabulário
    exibicao_alvo = "PALAVRA FORA DO VOCABULÁRIO (RUÍDO)" if modo_ruido_oov else classe_alvo.upper()
    print(f"\n🎤 [PRONTO] Alvo: '{exibicao_alvo}'. Pressione [ G ] para gravar...")
    
    while True:
        if keyboard.is_pressed('g'):
            while keyboard.is_pressed('g'):  
                time.sleep(0.01)
            break
            
    time.sleep(0.2)
            
    print("🔴 [GRAVANDO... Diga uma palavra qualquer!]" if modo_ruido_oov else "🔴 [GRAVANDO... Fale agora!]")
    audio_raw = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio_raw.flatten()
    
    # Calibração de Volume Corrigida (Ganho otimizado para 85%)
    pico = np.max(np.abs(audio))
    audio_salvamento = (audio / pico) * 0.85 if pico > 1e-6 else audio
    
    energia_rms = np.sqrt(np.mean(audio**2))
    vetor_tratado = processar_audio_microfone_direto(audio_salvamento)
    tensor_input = np.expand_dims(vetor_tratado, axis=0)
    
    predicoes = model.predict(tensor_input, verbose=0)[0]
    idx_vencedor = np.argmax(predicoes)
    confianca = predicoes[idx_vencedor]
    entropia = calcular_entropia_vetorial(predicoes)
    classe_predita = inv_map[idx_vencedor]
    
    # --- APLICAÇÃO DA LÓGICA DE DECISÃO DE SALVAMENTO ---
    if modo_ruido_oov:
        # No modo OOV, o 'acerto' esperado é a classe de ruído do modelo
        errou_classe = (classe_predita != CLASSE_RUIDO_MODELO)
        baixa_confianca = (confianca < THRESHOLD_CONFIAVEL)
        log_alvo = f"oov_ruido_{CLASSE_RUIDO_MODELO}"
    else:
        errou_classe = (classe_predita != classe_alvo)
        baixa_confianca = (confianca < THRESHOLD_CONFIAVEL)
        log_alvo = classe_alvo

    print(f"   🔹 Operador testou '{exibicao_alvo}' -> IA predisse '{classe_predita.upper()}' com {confianca*100:.1f}%")
    
    print("🔊 Reproduzindo áudio capturado...")
    sd.play(audio_salvamento, SAMPLE_RATE)
    sd.wait()
    
    # Grava se confundir com outra classe OU se tiver certeza menor que 75%
    if errou_classe or baixa_confianca:
        if modo_ruido_oov:
            motivo = "Confundiu Ruido com Comando" if errou_classe else "Baixa Certeza no Ruido (<75%)"
        else:
            motivo = "Erro de Classificacao" if errou_classe else "Baixa Confianca (<75%)"
            
        print(f"   ⚠️ FALHA DETECTADA ({motivo}) -> Gravando áudio para auditoria...")
        
        timestamp = int(time.time())
        nome_arquivo = f"USER_{usuario}_ALVO_{log_alvo}_PREDITO_{classe_predita}_CONF_{int(confianca*100)}_{timestamp}.wav"
        caminho_wav = os.path.join(PASTA_AUDITORIA_ERROS, nome_arquivo)
        
        sf.write(caminho_wav, audio_salvamento, SAMPLE_RATE)
        registrar_falha_csv(usuario, nome_arquivo, log_alvo, classe_predita, confianca, entropia, motivo, energia_rms)
    else:
        print("   ✅ REJEIÇÃO PERFEITA! A IA identificou o ruído com segurança. (Áudio descartado)")

def main():
    print("==========================================================")
    print("  SISTEMA DE AUDITORIA ATIVA - MODELO HÍBRIDO DE ATENÇÃO  ")
    print("==========================================================")
    print(f"ℹ️ Classe de descarte mapeada pelo sistema: '{CLASSE_RUIDO_MODELO.upper()}'")
    
    usuario = input("👤 Identificação do Operador (Seu nome/Iniciais): ").strip().lower().replace(" ", "_")
    if not usuario:
        usuario = "anonimo"

    while True:
        print("\n" + "═"*50)
        print("📋 CLASSES:", ", ".join([c.upper() for c in label_map.keys()]))
        print("🔥 DIGITE 'ruido' PARA TESTAR PALAVRAS FORA DO VOCABULÁRIO (OOV)")
        print("═"*50)
        
        classe_alvo = input("✍️ Selecione a classe (ou 'ruido', ou 'q' para sair): ").strip().lower()
        
        if classe_alvo == 'q':
            print("Encerrando coletor.")
            break
            
        modo_ruido_oov = (classe_alvo == 'ruido' and 'ruido' not in label_map) or (classe_alvo == 'ruido')
        
        # Se digitou 'ruido' mas a classe do JSON tem outro nome (ex: background), redirecionamos a lógica
        if modo_ruido_oov:
            print(f"\n🎯 Modo Estresse OOV iniciado. Diga palavras aleatórias (ex: carro, café, blabla).")
        else:
            if classe_alvo not in label_map:
                print("❌ Classe inválida!")
                continue
            print(f"\n🎯 Modo contínuo iniciado para a classe '{classe_alvo.upper()}'.")
            
        print("Segure a tecla [ ESC ] logo após um resultado se quiser voltar ao menu.")
        
        while True:
            executar_pipeline_coleta(classe_alvo, usuario, modo_ruido_oov=modo_ruido_oov)
            
            time.sleep(0.2)
            if keyboard.is_pressed('esc'):
                print("\n🔄 Voltando ao menu de seleção de classes...")
                break

if __name__ == "__main__":
    main()