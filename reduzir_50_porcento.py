from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ==================================
# CONFIGURAÇÃO
# ==================================

PASTA_ENTRADA = Path(
    r"C:\Users\ronca\Documents\Pos_processamento\mascaras_unet"
)

PASTA_SAIDA = Path(
    r"C:\Users\ronca\Documents\Pos_processamento\mascaras_unet_25"
)

FATOR = 0.25

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

EXTENSOES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ==================================
# PROCESSAMENTO
# ==================================

arquivos = sorted(
    [p for p in PASTA_ENTRADA.iterdir()
     if p.suffix.lower() in EXTENSOES]
)

print(f"\nEncontradas {len(arquivos)} imagens\n")

for arquivo in tqdm(arquivos):

    with Image.open(arquivo) as img:

        largura, altura = img.size

        nova_largura = int(largura * FATOR)
        nova_altura = int(altura * FATOR)

        # Máscaras binárias
        if img.mode == "L":

            img_red = img.resize(
                (nova_largura, nova_altura),
                Image.NEAREST
            )

        # RGB/RGBA
        else:

            img_red = img.resize(
                (nova_largura, nova_altura),
                Image.LANCZOS
            )

        destino = PASTA_SAIDA / arquivo.name

        img_red.save(destino)

print("\nConcluído!")
print(f"Imagens salvas em:\n{PASTA_SAIDA}")