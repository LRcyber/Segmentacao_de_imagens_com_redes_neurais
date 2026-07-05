import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Suprime logs verbose do TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ─────────────────────────────────────────────
# CONFIGURACOES
# ─────────────────────────────────────────────
PASTA_IMAGENS  = Path("C:/Users/ronca/Documents/Pos_processamento/imagens")
CAMINHO_MODELO = Path("C:/Users/ronca/Documents/Pos_processamento/2026-05-18_Unet_visible_best.h5")
PASTA_SAIDA    = Path("C:/Users/ronca/Documents/Pos_processamento/mascaras_unet")

THRESHOLD  = 0.5
BATCH_SIZE = 8   # processa 8 imagens por vez (reduza para 4 se der MemoryError)
EXTENSOES  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
# ─────────────────────────────────────────────


def carregar_modelo(caminho):
    import tensorflow as tf
    print("[INFO] Carregando modelo:", caminho.name)
    modelo = tf.keras.models.load_model(str(caminho), compile=False)
    _, h, w, c = modelo.input_shape
    print(f"[INFO] Entrada do modelo: {h}x{w}, canais: {c}")
    return modelo, (h, w, c)


def preparar_imagem(caminho_img, altura, largura, canais):
    img = Image.open(caminho_img).convert("RGB")
    w_orig, h_orig = img.size
    img = img.resize((largura, altura), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    if canais == 1:
        arr = np.mean(arr, axis=-1, keepdims=True)
    return arr, (h_orig, w_orig)


def salvar_mascara(pred, h_orig, w_orig, caminho_saida):
    pred = pred.squeeze()
    binaria = (pred >= THRESHOLD).astype(np.uint8) * 255
    mascara = Image.fromarray(binaria, mode="L")
    mascara = mascara.resize((w_orig, h_orig), Image.NEAREST)
    mascara.save(str(caminho_saida))


def main():
    if not PASTA_IMAGENS.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {PASTA_IMAGENS}")
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {CAMINHO_MODELO}")

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    print("[INFO] Mascaras serao salvas em:", PASTA_SAIDA)

    modelo, (h_modelo, w_modelo, c_modelo) = carregar_modelo(CAMINHO_MODELO)

    imagens = sorted([
        p for p in PASTA_IMAGENS.iterdir()
        if p.suffix.lower() in EXTENSOES
    ])
    total = len(imagens)
    if total == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada em: {PASTA_IMAGENS}")
    print(f"[INFO] {total} imagens encontradas. Batch size: {BATCH_SIZE}\n")

    erros = []
    processadas = 0

    # Processa em batches
    for i in range(0, total, BATCH_SIZE):
        lote = imagens[i : i + BATCH_SIZE]

        batch_arr = []
        batch_meta = []  # (h_orig, w_orig, caminho_saida)

        for caminho_img in lote:
            try:
                arr, (h_orig, w_orig) = preparar_imagem(
                    caminho_img, h_modelo, w_modelo, c_modelo
                )
                batch_arr.append(arr)
                nome_saida = PASTA_SAIDA / (caminho_img.stem + "_mask.png")
                batch_meta.append((h_orig, w_orig, nome_saida))
            except Exception as e:
                erros.append((caminho_img.name, str(e)))
                print(f"[ERRO leitura] {caminho_img.name}: {e}")

        if not batch_arr:
            continue

        # Inferencia do batch inteiro de uma vez
        batch_np = np.stack(batch_arr, axis=0)  # (B, H, W, C)
        try:
            predicoes = modelo.predict(batch_np, verbose=0)  # (B, H, W, 1)
        except Exception as e:
            for caminho_img in lote:
                erros.append((caminho_img.name, str(e)))
            print(f"[ERRO predict] batch {i}-{i+len(lote)}: {e}")
            continue

        # Salva cada mascara do batch
        for j, (h_orig, w_orig, caminho_saida) in enumerate(batch_meta):
            try:
                salvar_mascara(predicoes[j], h_orig, w_orig, caminho_saida)
                processadas += 1
            except Exception as e:
                erros.append((caminho_saida.name, str(e)))
                print(f"[ERRO salvar] {caminho_saida.name}: {e}")

        # Progresso manual a cada batch
        pct = processadas / total * 100
        print(f"  [{processadas:4d}/{total}] {pct:5.1f}%  — ultimo: {lote[-1].name}",
              flush=True)

    print(f"\n{'='*55}")
    print(f"  Concluido! {processadas}/{total} mascaras geradas.")
    if erros:
        print(f"  {len(erros)} erros:")
        for nome, msg in erros:
            print(f"    - {nome}: {msg}")
    print(f"  Saida: {PASTA_SAIDA}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()