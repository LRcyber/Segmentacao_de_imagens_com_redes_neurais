"""
aplicar_textura_pele.py  v7
============================
Usa canal alpha das proprias png_transparente como mascara.
Erosao da mascara afasta pixels de borda ruidosa (mistura fundo azul + corpo).

Requisitos:
    pip install numpy trimesh pillow tqdm scipy
"""

import warnings
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt
import trimesh

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
CAMINHO_PLY    = Path("C:/Users/ronca/Documents/Pos_processamento/modelo_3d/model3d.ply")
PASTA_TEXTURAS = Path("C:/Users/ronca/Documents/Pos_processamento/png_transparente")
PASTA_SAIDA    = Path("C:/Users/ronca/Documents/Pos_processamento/modelo_3d")
NOME_SAIDA     = "model3d_texturizado_v7.ply"

SPACING_XY = 0.33
SPACING_Z  = 1.0
JANELA_BLEND  = 1   # +-1 fatia para blend suave

# Erosao da borda: afasta N pixels da borda do corpo para evitar
# pixels de mistura fundo-azul/corpo. Aumente se ainda aparecer azul.
EROSAO_BORDA  = 6

# Espessura da casca de pele amostrada a partir da borda erodida
# Aumente se ficarem buracos sem textura.
ESPESSURA_CASCA = 20

COR_FALLBACK  = np.array([180, 140, 120], dtype=np.uint8)
EXTENSOES     = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# ─────────────────────────────────────────────


def listar_arquivos(pasta):
    return sorted([p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES])


def carregar_rgba(caminho):
    with Image.open(caminho) as im:
        return np.array(im.convert("RGBA"), dtype=np.uint8)


def gerar_mascara_pele(rgba, erosao, espessura):
    """
    A partir do canal alpha da imagem RGBA:
    1. Binariza alpha -> mascara do corpo
    2. Erode EROSAO_BORDA pixels para afastar pixels de borda ruidosa
    3. Calcula distancia da borda erodida
    4. Aceita casca de ESPESSURA_CASCA pixels a partir da borda erodida
    """
    alpha  = rgba[:, :, 3]
    corpo  = (alpha >= 10)

    if not np.any(corpo):
        return np.zeros(alpha.shape, dtype=bool)

    # Erode para afastar pixels de borda com mistura de cor
    struct = np.ones((3, 3), dtype=bool)
    corpo_erodido = corpo.copy()
    for _ in range(erosao):
        corpo_erodido = binary_erosion(corpo_erodido, structure=struct)

    if not np.any(corpo_erodido):
        # Se erosao zerou tudo (fatia fina), usa corpo sem erosao
        corpo_erodido = corpo

    # Distancia de cada pixel ate a borda do corpo erodido
    dist = distance_transform_edt(corpo_erodido)

    # Casca: pixels do corpo erodido proximos da borda
    casca = corpo_erodido & (dist <= espessura)
    return casca


def calcular_indices(verts_mm, n_fatias, H, W):
    z_fatia = np.clip(
        np.round(verts_mm[:, 2] / SPACING_Z), 0, n_fatias - 1
    ).astype(np.int32)
    y_px = np.clip(
        np.round(verts_mm[:, 1] / SPACING_XY), 0, H - 1
    ).astype(np.int32)
    x_px = np.clip(
        np.round(verts_mm[:, 0] / SPACING_XY), 0, W - 1
    ).astype(np.int32)
    return z_fatia, y_px, x_px


def texturizar(verts_mm, texturas):
    n_fatias = len(texturas)
    n_verts  = len(verts_mm)

    with Image.open(texturas[0]) as im:
        W_orig, H_orig = im.size
    print(f"[INFO] Texturas     : {W_orig}x{H_orig} px")
    print(f"[INFO] Erosao borda : {EROSAO_BORDA} px")
    print(f"[INFO] Espessura    : {ESPESSURA_CASCA} px")
    print(f"[INFO] Blend        : +-{JANELA_BLEND} fatias\n")

    z_fatia, y_px, x_px = calcular_indices(verts_mm, n_fatias, H_orig, W_orig)

    cores = np.full((n_verts, 3), COR_FALLBACK, dtype=np.float32)
    pesos = np.zeros(n_verts, dtype=np.float32)

    fatias_necessarias = sorted(set(
        int(np.clip(z + dz, 0, n_fatias - 1))
        for z in np.unique(z_fatia)
        for dz in range(-JANELA_BLEND, JANELA_BLEND + 1)
    ))

    cache_rgba  = {}
    cache_casca = {}

    for zi in tqdm(fatias_necessarias, desc="Processando fatias", unit="fatia"):

        # Carrega RGBA e casca para esta fatia
        if zi not in cache_rgba:
            cache_rgba[zi]  = carregar_rgba(texturas[zi])
            cache_casca[zi] = gerar_mascara_pele(
                cache_rgba[zi], EROSAO_BORDA, ESPESSURA_CASCA
            )
            # Limpa cache antigo
            if len(cache_rgba) > 15:
                k = min(cache_rgba.keys())
                del cache_rgba[k]
                del cache_casca[k]

        img   = cache_rgba[zi]
        casca = cache_casca[zi]

        for dz in range(-JANELA_BLEND, JANELA_BLEND + 1):
            z_alvo = int(np.clip(zi - dz, 0, n_fatias - 1))
            mask   = (z_fatia == z_alvo)
            if not np.any(mask):
                continue

            yy   = y_px[mask]
            xx   = x_px[mask]
            rgba = img[yy, xx]

            # Valido: dentro da casca de pele (ja exclui borda ruidosa e interior)
            valido = casca[yy, xx]

            if np.any(valido):
                idx = np.where(mask)[0][valido]
                cores[idx] += rgba[valido, :3].astype(np.float32)
                pesos[idx] += 1.0

    # Normaliza blend
    tem_cor = pesos > 0
    cores[tem_cor] /= pesos[tem_cor, np.newaxis]

    print(f"\n[INFO] Com cor de pele : {tem_cor.sum():,} ({tem_cor.sum()/n_verts*100:.1f}%)")
    print(f"[INFO] Com fallback    : {(~tem_cor).sum():,} ({(~tem_cor).sum()/n_verts*100:.1f}%)")

    return cores.astype(np.uint8)


def main():
    if not CAMINHO_PLY.exists():
        raise FileNotFoundError(f"PLY nao encontrado: {CAMINHO_PLY}")
    if not PASTA_TEXTURAS.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {PASTA_TEXTURAS}")

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    texturas = listar_arquivos(PASTA_TEXTURAS)
    print(f"[INFO] {len(texturas)} texturas\n")

    print("[INFO] Carregando PLY...")
    mesh  = trimesh.load(str(CAMINHO_PLY), process=False)
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces)
    print(f"[INFO] {len(verts):,} vertices | {len(faces):,} faces\n")

    cores = texturizar(verts, texturas)

    print("\n[INFO] Salvando PLY...")
    mesh_tex = trimesh.Trimesh(
        vertices=verts, faces=faces,
        vertex_colors=cores, process=False
    )
    caminho_saida = PASTA_SAIDA / NOME_SAIDA
    mesh_tex.export(str(caminho_saida))

    print(f"\n{'='*55}")
    print(f"  CONCLUIDO! -> {caminho_saida}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
