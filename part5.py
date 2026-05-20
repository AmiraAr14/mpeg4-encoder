import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from part1 import load_frames, bgr_to_ycbcr
from part2 import get_quant_matrix, encode_channel, decode_channel, dct2, idct2
from part3 import encode_pframe, decode_pframe
from part4 import encode_video, decode_video

QUANTIZATION_FACTOR = 10

# ── Charger les frames ─────────────────────────────────────
frames = load_frames("frames")

print("Encodage en cours...")
bitstream = encode_video(frames)
frames_recon = decode_video(bitstream)
print("Done !")

# ── Compression ratio vs Quantization Factor ───────────────
print("Calcul compression vs QF...")
qf_range = [1, 5, 10, 20, 50, 100]
ratios = []
h, w, c = frames[0].shape
original_bytes = len(frames) * h * w * c

for qf in qf_range:
    bs = encode_video(frames, qf=qf)
    r = original_bytes / len(bs)
    ratios.append(r)
    print(f"  QF={qf:3d} → ratio={r:.1f}x")

# ── Analyse GOP size ────────────────────────────────────────
print("Calcul compression vs GOP size...")
gop_range = [1, 2, 4, 8, len(frames)]
gop_ratios = []

import part4 as _part4
old_gop = _part4.GOP_SIZE

for gop in gop_range:
    _part4.GOP_SIZE = gop
    bs = _part4.encode_video(frames)
    r = original_bytes / len(bs)
    gop_ratios.append(r)
    print(f"  GOP={gop:3d} → ratio={r:.1f}x")

_part4.GOP_SIZE = old_gop

# ── Préparer les données de visualisation ──────────────────
qm = get_quant_matrix(QUANTIZATION_FACTOR)
Y, Cb, Cr = bgr_to_ycbcr(frames[0])
best_var, best_by, best_bx = 0, 32, 32
for by in range(0, Y.shape[0] - 8, 8):
    for bx in range(0, Y.shape[1] - 8, 8):
        v = np.var(Y[by:by+8, bx:bx+8])
        if v > best_var:
            best_var, best_by, best_bx = v, by, bx
block_orig  = Y[best_by:best_by+8, best_bx:best_bx+8] - 128
block_dct   = dct2(block_orig)
block_quant = np.round(block_dct / qm).astype(np.int16)
block_recon = idct2(block_quant.astype(np.float32) * qm) + 128

Y0, _, _ = bgr_to_ycbcr(frames[0])
Y1, _, _ = bgr_to_ycbcr(frames[1])
mvs, residuals_q = encode_pframe(Y1, Y0, qm)
residual_map = np.abs(Y1 - Y0)
residual_map_signed = Y1.astype(np.float32) - Y0.astype(np.float32)


# ══════════════════════════════════════════════════════════════
# FIGURE 1 — Visualisation des images
# ══════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(20, 22))
fig1.suptitle("Pipeline MPEG-4 — Visualisation des images", fontsize=16,
              fontweight='bold', y=0.97)

gs1 = gridspec.GridSpec(
    5, 4,
    figure=fig1,
    hspace=0.45,
    wspace=0.40,
    top=0.90,
    bottom=0.03,
    left=0.05,
    right=0.97
)

# ── Ligne 1 : frames originales ────────────────────────────
frame_indices = [0, 4, 8, 12]
frame_types   = ["I", "P", "P", "P"]
for col, (fi, ft) in enumerate(zip(frame_indices, frame_types)):
    ax = fig1.add_subplot(gs1[0, col])
    ax.imshow(cv2.cvtColor(frames[fi], cv2.COLOR_BGR2RGB))
    ax.set_title(f"Frame {fi} ({ft})", fontsize=11, pad=5)
    ax.axis('off')

# ── Ligne 2 : canaux Y / Cb / Cr ──────────────────────────
items_l2 = [
    (cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB), "Original",          None),
    (Y,                                           f"Y  {Y.shape}",    'gray'),
    (Cb,                                          f"Cb {Cb.shape}",   'Blues'),
    (Cr,                                          f"Cr {Cr.shape}",   'Reds'),
]
for col, (data, title, cmap) in enumerate(items_l2):
    ax = fig1.add_subplot(gs1[1, col])
    ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=11, pad=5)
    ax.axis('off')

# ── Ligne 3 : DCT sur un bloc 8x8 ─────────────────────────
items_l3 = [
    (block_orig + 128,                  "Bloc original",        'hot'),
    (np.abs(block_dct),                 "Après DCT",            'hot'),
    (np.abs(block_quant.astype(float)), "Après quantization",   'hot'),
    (block_recon,                       "Bloc reconstruit",     'hot'),
]
for col, (data, title, cmap) in enumerate(items_l3):
    ax = fig1.add_subplot(gs1[2, col])
    ax.imshow(data, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=11, pad=5)
    ax.axis('off')

# ── Ligne 4 : vecteurs de mouvement + frames orig/recon ────
ax = fig1.add_subplot(gs1[3, 0])
ax.imshow(cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB))
for (by, bx), (dy, dx) in mvs:
    if dy != 0 or dx != 0:
        ax.annotate("", xy=(bx+dx+8, by+dy+8), xytext=(bx+8, by+8),
                    arrowprops=dict(arrowstyle="->", color='yellow', lw=1.2))
ax.set_title("Vecteurs de mouvement", fontsize=11, pad=5)
ax.axis('off')

ax = fig1.add_subplot(gs1[3, 1])
ax.imshow(residual_map, cmap='hot')
ax.set_title("Résidu (différence)", fontsize=11, pad=5)
ax.axis('off')

ax = fig1.add_subplot(gs1[3, 2])
ax.imshow(cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB))
ax.set_title("Frame 1 originale", fontsize=11, pad=5)
ax.axis('off')

ax = fig1.add_subplot(gs1[3, 3])
ax.imshow(cv2.cvtColor(frames_recon[1], cv2.COLOR_BGR2RGB))
ax.set_title("Frame 1 reconstruite", fontsize=11, pad=5)
ax.axis('off')

fig1.savefig("part5_images.png", dpi=150, bbox_inches='tight')
print("Figure 1 sauvegardée : part5_images.png")

# ── Ligne 5 : résidu signé rouge/bleu ─────────────────────
ax = fig1.add_subplot(gs1[4, 1])
im = ax.imshow(residual_map_signed, cmap='RdBu_r', vmin=-50, vmax=50)
ax.set_title("Résidu (P-frame 1 vs prev)", fontsize=11, pad=5)
ax.axis('off')
fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ══════════════════════════════════════════════════════════════
# FIGURE 2 — Courbes d'analyse
# ══════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Pipeline MPEG-4 — Analyse quantitative", fontsize=16,
              fontweight='bold')
fig2.subplots_adjust(wspace=0.35, top=0.85, bottom=0.15, left=0.08, right=0.97)

# Courbe 1 : Compression vs QF
ax = axes[0]
ax.plot(qf_range, ratios, marker='o', color='coral', linewidth=1.8, markersize=5)
ax.set_xlabel("Quantization Factor", fontsize=10)
ax.set_ylabel("Compression ratio", fontsize=10)
ax.set_title("Compression vs Quantization Factor", fontsize=12, pad=8)
ax.grid(True, alpha=0.4)
ax.tick_params(labelsize=9)

# Courbe 2 : Compression vs GOP Size
ax = axes[1]
ax.plot(gop_range, gop_ratios, marker='s', color='green', linewidth=1.8, markersize=5)
ax.set_xlabel("GOP Size", fontsize=10)
ax.set_ylabel("Compression ratio", fontsize=10)
ax.set_title("Compression vs GOP Size", fontsize=12, pad=8)
ax.grid(True, alpha=0.4)
ax.tick_params(labelsize=9)

fig2.savefig("part5_courbes.png", dpi=150, bbox_inches='tight')
print("Figure 2 sauvegardée : part5_courbes.png")

plt.show()

# ── Résultats console ──────────────────────────────────────
print(f"\n=== RÉSULTATS FINAUX ===")
print(f"I-frames : {sum(1 for i in range(len(frames)) if i % 4 == 0)}")
print(f"P-frames : {sum(1 for i in range(len(frames)) if i % 4 != 0)}")