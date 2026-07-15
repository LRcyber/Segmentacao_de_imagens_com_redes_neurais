# Reconstrução 3D do Visible Human Project via Segmentação por Deep Learning

Pipeline completo para reconstrução tridimensional de estruturas anatômicas a partir do dataset **Visible Human Project (VHP)**, utilizando segmentação semântica com **U-Net**, reconstrução de malha 3D e reprojeção de textura de pele.

Projeto desenvolvido como Trabalho de Conclusão de Curso (TCC) em Ciência da Computação — Universidade Estadual de Santa Cruz (UESC).

---

## 📋 Visão geral

O pipeline transforma uma sequência de **1.865 imagens 2D** do dataset VHP em um **modelo 3D texturizado**, passando por três grandes etapas:

1. **Segmentação semântica** das fatias com uma U-Net treinada, gerando máscaras binárias.
2. **Reconstrução da malha 3D** a partir das máscaras via Marching Cubes.
3. **Reprojeção de textura RGB** (pele) sobre a malha reconstruída.

Resultado: um modelo 3D com aproximadamente **2,27 milhões de vértices** e **4,54 milhões de faces**.

---

## 🧠 Segmentação (U-Net)

- Modelo treinado: `2026-05-18_Unet_visible_best.h5`
- Aplicado sobre as 1.865 imagens PNG do VHP
- Avaliação com **coeficiente de Dice** e **IoU**
- Máscaras binárias (não as imagens coloridas segmentadas) foram usadas como entrada para a reconstrução — uso das imagens coloridas diretamente causava vértices internos indesejados (segmentando órgãos internos) e estouro de memória, inviabilizando o fechamento da malha.

---

## 🧊 Reconstrução 3D — `reconstrucao_3d_vhp.py`

Reconstrói a malha 3D a partir das máscaras binárias segmentadas.

**Parâmetros principais:**

| Parâmetro | Valor | Descrição |
|---|---|---|
| `FATOR` | 0.25 | Fator de downscaling das imagens |
| `SPACING_XY` | 0.33 mm/px | Espaçamento entre pixels no plano XY |
| `SPACING_Z` | 1.0 mm | Espaçamento entre fatias no eixo Z |
| `CHUNK_SIZE` | 64 | Tamanho do bloco para Marching Cubes em chunks |
| `OVERLAP` | 16 | Sobreposição entre chunks |

**Etapas:**
- Interpolação isotrópica via `scipy.ndimage.zoom`
- Marching Cubes em chunks (processamento por blocos, evitando estouro de memória)
- Suavização por Laplacian smoothing
- Exportação: `model3d.ply`

---

## 🎨 Texturização — `aplicar_textura_pele.py` (v7)

Aplica a textura RGB de pele sobre a malha reconstruída, usando o canal alfa das imagens PNG transparentes como máscara.

**Parâmetros principais:**

| Parâmetro | Valor | Descrição |
|---|---|---|
| `EROSAO_BORDA` | 6 px | Erosão morfológica nas bordas do canal alfa |
| `ESPESSURA_CASCA` | 20 px | Espessura da "casca" de pele via distance transform |
| `JANELA_BLEND` | ±1 fatia | Janela de blending entre fatias adjacentes |

**Saída:** `model3d_texturizado_v7.ply` — modelo com detalhes de textura preservados (ex.: tatuagem de bandeira americana).

> Observação: o modelo sólido foi exportado sem coloração de vértices; o modelo oco carrega a textura de pele e os detalhes visuais.

---

## 🔧 Problemas técnicos resolvidos

Durante o desenvolvimento, diversos obstáculos de engenharia foram superados:

- Limitação de *long path* do Windows quebrando o TensorFlow
- Erros de Unicode em caminhos de arquivo
- Barras de progresso congelando durante processamento longo
- Bugs de mapeamento de eixos na reprojeção de textura
- Falhas do OpenCV em caminhos com acentos (contornado com Pillow e caminhos curtos)
- Detecção de arquivos duplicados no Windows
- Espaçamento XY incorreto do VHP (corrigido para 0.33 mm/px)
- Mudanças na API do `trimesh`
- *File-locking* do `numpy.memmap` no Windows

---

## 📈 Evolução do projeto

O pipeline atual é resultado de diversas iterações:

1. **3D Slicer** (Threshold → Islands → Smoothing → Create Closed Surface → Wrap Solidify → exportação PLY → visualização no MeshLab), com script Python de *color-baking* mapeando coordenadas de vértices OBJ para pixels de imagens PNG.
2. **Regra Delta (RGBDZ)** — subtração pixel a pixel entre fatias consecutivas do VHP para extrair informação de superfície, reproduzindo abordagem de trabalho anterior da UESC (Bomfim, 2017).
3. Protótipo em **Google Colab** antes da migração para execução local em Windows.
4. Pipeline atual, totalmente em Python, com segmentação por U-Net e reconstrução via Marching Cubes.

---

## 📚 Trabalhos relacionados

Comparado a trabalhos anteriores de reconstrução 3D do VHP (Bonfim, 2017; Bastos, 2020; Campos, 2022), este pipeline:

- Utiliza o **conjunto completo de imagens**, sem necessidade de redução de dados
- Produz modelos **mais detalhados**
- Evita as limitações de qualidade observadas nos trabalhos anteriores

---

## 🚀 Direções futuras

- Arquiteturas de segmentação mais recentes: **UNet++**, **UNet 3D** (Zhou et al., 2018)
- Aplicações relacionadas em sensoriamento remoto (LeCun; Bengio; Hinton, 2015)
- Exploração de **Attention U-Net** e **SAM (Segment Anything Model)** como estado da arte em segmentação

---

## 🛠️ Stack técnica

`Python` · `TensorFlow/Keras` · `OpenCV` · `scipy` · `trimesh` · `3D Slicer` · `MeshLab`

---

## 📄 Licença / Contexto acadêmico

Projeto desenvolvido no âmbito de um Trabalho de Conclusão de Curso em Ciência da Computação (UESC), com documentação formatada segundo as normas ABNT.
