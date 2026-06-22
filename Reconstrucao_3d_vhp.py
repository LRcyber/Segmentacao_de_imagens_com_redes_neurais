"""
reconstrucao_3d_vhp.py  v3
===========================
Correcoes v3:
  - Textura: amostragem em janela ±2 fatias + blend para suavizar listras
  - Chunks: overlap aumentado para 16, zona de filtro mais agressiva
  - Orientacao: eixo Z flipado (modelo nao fica de cabeca para baixo)

Requisitos:
    pip install numpy pillow scipy scikit-image trimesh tqdm
"""

import os
import gc
import warnings
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing, label as nd_label
from skimage.measure import marching_cubes
import trimesh

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ─────────────────────────────────────────────
# CONFIGURACOES
# ─────────────────────────────────────────────
PASTA_MASCARAS  = Path("C:/Users/ronca/Documents/Pos_processamento/mascaras_unet")
PASTA_TEXTURAS  = Path("C:/Users/ronca/Documents/Pos_processamento/png_transparente")
PASTA_SAIDA     = Path("C:/Users/ronca/Documents/Pos_processamento/modelo_3d")

SPACING_XY = 0.33   # mm/pixel original
SPACING_Z  = 1.0    # mm entre fatias
FATOR      = 0.5    # downscale XY (0.5 = metade, economiza RAM)

# Limpeza morfologica
ITER_ABERTURA   = 2
ITER_FECHAMENTO = 3
MIN_AREA_PX     = 500

# Marching Cubes
NIVEL_ISO  = 0.5
CHUNK_SIZE = 64
OVERLAP    = 16    # FIX: era 8, aumentado para reduzir artefatos de juncao

# Textura: janela de fatias para blend de cor (reduz listras)
JANELA_TEXTURA = 2   # amostra ±2 fatias vizinhas e faz media

# Suavizacao Laplaciana
SUAVIZACAO_ITER = 8  # ligeiramente mais suavizacao

EXTENSOES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# ─────────────────────────────────────────────


def listar_arquivos(pasta, extensoes):
    return sorted([p for p in pasta.iterdir() if p.suffix.lower() in extensoes])


def limpar_fatia(mascara_bin):
    arr = mascara_bin.astype(bool)
    arr = binary_opening(arr, iterations=ITER_ABERTURA)

    rotulada, n = nd_label(arr)
    if n > 0:
        tamanhos = ndimage.sum(arr, rotulada, range(1, n + 1))
        mascara_valida = np.zeros_like(arr)
        for idx, tam in enumerate(tamanhos):
            if tam >= MIN_AREA_PX:
                mascara_valida |= (rotulada == idx + 1)
        arr = mascara_valida

    arr = binary_closing(arr, iterations=ITER_FECHAMENTO)

    rotulada, n = nd_label(arr)
    if n > 1:
        tamanhos = ndimage.sum(arr, rotulada, range(1, n + 1))
        maior = int(np.argmax(tamanhos)) + 1
        arr = (rotulada == maior)

    return arr.astype(np.uint8)


def carregar_volume(mascaras, fator):
    print(f"\n[1/5] Carregando {len(mascaras)} mascaras...")
    with Image.open(mascaras[0]) as im:
        w0, h0 = im.size
    w_work = int(w0 * fator)
    h_work = int(h0 * fator)
    print(f"      Resolucao de trabalho: {w_work}x{h_work} (fator={fator})")

    volume = np.zeros((len(mascaras), h_work, w_work), dtype=np.uint8)
    for i, caminho in enumerate(tqdm(mascaras, desc="  Lendo mascaras", unit="fatia")):
        with Image.open(caminho) as im:
            if im.mode != "L":
                im = im.convert("L")
            im = im.resize((w_work, h_work), Image.NEAREST)
            arr = np.array(im)
        volume[i] = limpar_fatia((arr > 127).astype(np.uint8))

    print(f"      Volume montado: {volume.shape} ({volume.nbytes / 1e6:.1f} MB)")
    return volume, (h_work, w_work)


def interpolar_isotropico(volume, spacing_xy_scaled, spacing_z):
    print("\n[2/5] Interpolando para voxels isotropicos...")
    zoom_z = spacing_z / spacing_xy_scaled
    print(f"      Fator zoom Z: {zoom_z:.3f}")
    volume_iso = ndimage.zoom(volume.astype(np.float32), zoom=(zoom_z, 1.0, 1.0), order=1)
    volume_iso = (volume_iso > 0.5).astype(np.uint8)
    print(f"      Volume isotropico: {volume_iso.shape}")
    return volume_iso


def marching_cubes_chunked(volume):
    """
    Marching Cubes por chunks com overlap=16.
    Faces dentro da zona de overlap sao descartadas para evitar artefatos.
    """
    print("\n[3/5] Executando Marching Cubes...")
    D, H, W = volume.shape
    todos_verts = []
    todas_faces = []
    offset_verts = 0

    z_starts = list(range(0, D, CHUNK_SIZE - OVERLAP))

    for z0 in tqdm(z_starts, desc="  Chunks", unit="chunk"):
        z1 = min(z0 + CHUNK_SIZE, D)
        chunk = volume[z0:z1]

        if chunk.max() == 0 or chunk.min() == 1:
            continue

        try:
            verts, faces, _, _ = marching_cubes(
                chunk.astype(np.float32),
                level=NIVEL_ISO,
                spacing=(1.0, 1.0, 1.0)
            )
        except Exception:
            continue

        # Descarta faces na zona de overlap do inicio do chunk
        if z0 > 0:
            zona = OVERLAP // 2
            mascara_faces = ~np.any(verts[faces][:, :, 0] < zona, axis=1)
            faces = faces[mascara_faces]

        # Descarta faces na zona de overlap do fim do chunk (exceto ultimo)
        if z1 < D:
            zona = (z1 - z0) - OVERLAP // 2
            mascara_faces = ~np.any(verts[faces][:, :, 0] > zona, axis=1)
            faces = faces[mascara_faces]

        if len(faces) == 0:
            continue

        verts[:, 0] += z0
        todos_verts.append(verts)
        todas_faces.append(faces + offset_verts)
        offset_verts += len(verts)

    if not todos_verts:
        raise RuntimeError("Marching Cubes nao gerou nenhuma face!")

    verts_final = np.concatenate(todos_verts, axis=0)
    faces_final = np.concatenate(todas_faces, axis=0)
    print(f"      Malha: {len(verts_final):,} vertices, {len(faces_final):,} faces")
    return verts_final, faces_final


def escalar_vertices(verts, spacing_xy_scaled, n_fatias_iso):
    """
    Escala voxels -> mm e corrige orientacao (flip Z para nao ficar invertido).
    Apos interpolacao isotrópica todos os eixos usam spacing_xy_scaled.
    """
    verts_mm = verts.copy().astype(np.float64)
    # FIX orientacao: inverte eixo Z (cabeca para cima)
    verts_mm[:, 0] = (n_fatias_iso - 1 - verts_mm[:, 0]) * spacing_xy_scaled
    verts_mm[:, 1] *= spacing_xy_scaled
    verts_mm[:, 2] *= spacing_xy_scaled
    return verts_mm


def carregar_textura_fatia(caminho, fator):
    """Carrega uma fatia de textura RGBA redimensionada."""
    with Image.open(caminho) as im:
        w0, h0 = im.size
        w_work = int(w0 * fator)
        h_work = int(h0 * fator)
        im_rgba = im.convert("RGBA").resize((w_work, h_work), Image.BILINEAR)
        return np.array(im_rgba)


def texturizar_vertices(verts, volume_shape, texturas, fator, escala_z):
    """
    Amostra cor RGB com blend em janela de ±JANELA_TEXTURA fatias vizinhas.
    Isso suaviza as listras causadas por variacoes de cor entre fatias.
    """
    print("\n[4/5] Texturizando vertices...")
    D, H, W = volume_shape
    n_tex = len(texturas)
    cores = np.zeros((len(verts), 3), dtype=np.float32)
    cache = {}

    for i, (z_iso, y, x) in enumerate(tqdm(verts, desc="  Amostrando cores", unit="vert")):
        z_centro = z_iso / escala_z
        y_px = int(np.clip(round(y), 0, H - 1))
        x_px = int(np.clip(round(x), 0, W - 1))

        # Coleta fatias na janela ±JANELA_TEXTURA
        amostras = []
        for dz in range(-JANELA_TEXTURA, JANELA_TEXTURA + 1):
            z_idx = int(np.clip(round(z_centro + dz), 0, n_tex - 1))
            if z_idx not in cache:
                cache[z_idx] = carregar_textura_fatia(texturas[z_idx], fator)
                if len(cache) > 30:
                    del cache[min(cache.keys())]
            img = cache[z_idx]
            r, g, b, a = img[y_px, x_px]
            if a >= 10:
                amostras.append([r, g, b])

        if amostras:
            cores[i] = np.mean(amostras, axis=0)
        else:
            cores[i] = [180, 140, 120]  # tom de pele neutro para pixels transparentes

    return cores.astype(np.uint8)


def exportar(verts_mm, faces, cores):
    print("\n[5/5] Exportando modelo 3D...")
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.Trimesh(
        vertices=verts_mm,
        faces=faces,
        vertex_colors=cores,
        process=False
    )

    print(f"      Suavizando malha ({SUAVIZACAO_ITER} iteracoes Laplacian)...")
    trimesh.smoothing.filter_laplacian(mesh, iterations=SUAVIZACAO_ITER)

    caminho_ply = PASTA_SAIDA / "vhp_modelo.ply"
    mesh.export(str(caminho_ply))
    print(f"  PLY salvo: {caminho_ply}")

    caminho_obj = PASTA_SAIDA / "vhp_modelo.obj"
    mesh.export(str(caminho_obj))
    print(f"  OBJ salvo: {caminho_obj}")

    return mesh


def main():
    for pasta in [PASTA_MASCARAS, PASTA_TEXTURAS]:
        if not pasta.exists():
            raise FileNotFoundError(f"Pasta nao encontrada: {pasta}")
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    mascaras = listar_arquivos(PASTA_MASCARAS, EXTENSOES)
    texturas = listar_arquivos(PASTA_TEXTURAS, EXTENSOES)

    print(f"[INFO] {len(mascaras)} mascaras | {len(texturas)} texturas")
    print(f"[INFO] FATOR={FATOR} | SPACING_XY={SPACING_XY} mm | SPACING_Z={SPACING_Z} mm")
    print(f"[INFO] OVERLAP={OVERLAP} | JANELA_TEXTURA=+-{JANELA_TEXTURA} fatias")

    if len(mascaras) == 0:
        raise RuntimeError("Nenhuma mascara encontrada!")
    if len(texturas) == 0:
        raise RuntimeError("Nenhuma imagem de textura encontrada!")
    if len(mascaras) != len(texturas):
        print(f"[AVISO] Mascaras ({len(mascaras)}) != Texturas ({len(texturas)})")
        n = min(len(mascaras), len(texturas))
        mascaras, texturas = mascaras[:n], texturas[:n]
        print(f"         Usando primeiras {n}.")

    spacing_xy_scaled = SPACING_XY / FATOR
    print(f"[INFO] spacing_xy_scaled = {spacing_xy_scaled:.4f} mm/px\n")

    volume, (H, W) = carregar_volume(mascaras, FATOR)
    volume_iso     = interpolar_isotropico(volume, spacing_xy_scaled, SPACING_Z)
    D_iso          = volume_iso.shape[0]
    del volume; gc.collect()

    verts, faces = marching_cubes_chunked(volume_iso)
    del volume_iso; gc.collect()

    escala_z = D_iso / len(mascaras)
    cores    = texturizar_vertices(verts, (len(mascaras), H, W), texturas, FATOR, escala_z)
    verts_mm = escalar_vertices(verts, spacing_xy_scaled, D_iso)

    mesh = exportar(verts_mm, faces, cores)

    print(f"\n{'='*55}")
    print(f"  CONCLUIDO!")
    print(f"  Vertices : {len(mesh.vertices):,}")
    print(f"  Faces    : {len(mesh.faces):,}")
    print(f"  Saida    : {PASTA_SAIDA}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()