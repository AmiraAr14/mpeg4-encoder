import cv2
import numpy as np
import matplotlib.pyplot as plt
from part1 import load_frames, bgr_to_ycbcr, ycbcr_to_bgr
from part2 import encode_channel, decode_channel, get_quant_matrix, dct2, idct2

BLOCK_SIZE = 16
SEARCH_WINDOW = 16
QUANTIZATION_FACTOR = 10

def motion_estimate(current_block, ref_frame, by, bx):
    h, w = ref_frame.shape
    best_sad = float('inf')
    best_mv = (0, 0)
    for dy in range(-SEARCH_WINDOW, SEARCH_WINDOW + 1):
        for dx in range(-SEARCH_WINDOW, SEARCH_WINDOW + 1):
            ry, rx = by + dy, bx + dx
            if ry < 0 or rx < 0 or ry + BLOCK_SIZE > h or rx + BLOCK_SIZE > w:
                continue
            ref_block = ref_frame[ry:ry+BLOCK_SIZE, rx:rx+BLOCK_SIZE]
            sad = np.sum(np.abs(current_block.astype(np.float32) - ref_block.astype(np.float32)))
            if sad < best_sad:
                best_sad = sad
                best_mv = (dy, dx)
    return best_mv

def encode_pframe(current_Y, ref_Y, quant_matrix):
    h, w = current_Y.shape
    motion_vectors = []
    residuals_q = np.zeros((h, w), dtype=np.int16)
    for by in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        for bx in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            curr_block = current_Y[by:by+BLOCK_SIZE, bx:bx+BLOCK_SIZE]
            mv = motion_estimate(curr_block, ref_Y, by, bx)
            motion_vectors.append(((by, bx), mv))
            ry, rx = by + mv[0], bx + mv[1]
            pred_block = ref_Y[ry:ry+BLOCK_SIZE, rx:rx+BLOCK_SIZE]
            residual = curr_block.astype(np.float32) - pred_block.astype(np.float32)
            for si in range(0, BLOCK_SIZE, 8):
                for sj in range(0, BLOCK_SIZE, 8):
                    sub = residual[si:si+8, sj:sj+8]
                    dct_r = dct2(sub)
                    q_r = np.round(dct_r / quant_matrix).astype(np.int16)
                    residuals_q[by+si:by+si+8, bx+sj:bx+sj+8] = q_r
    return motion_vectors, residuals_q

def decode_pframe(ref_Y, motion_vectors, residuals_q, quant_matrix):
    h, w = ref_Y.shape
    recon = np.zeros((h, w), dtype=np.float32)
    for (by, bx), (dy, dx) in motion_vectors:
        ry, rx = by + dy, bx + dx
        pred_block = ref_Y[ry:ry+BLOCK_SIZE, rx:rx+BLOCK_SIZE].astype(np.float32)
        res_block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
        for si in range(0, BLOCK_SIZE, 8):
            for sj in range(0, BLOCK_SIZE, 8):
                q = residuals_q[by+si:by+si+8, bx+sj:bx+sj+8].astype(np.float32)
                res_block[si:si+8, sj:sj+8] = idct2(q * quant_matrix)
        recon[by:by+BLOCK_SIZE, bx:bx+BLOCK_SIZE] = np.clip(pred_block + res_block, 0, 255)
    return recon

if __name__ == "__main__":
    frames = load_frames("frames")
    qm = get_quant_matrix(QUANTIZATION_FACTOR)

    Y0, Cb0, Cr0 = bgr_to_ycbcr(frames[0])
    Y1, Cb1, Cr1 = bgr_to_ycbcr(frames[1])

    print("Calcul des vecteurs de mouvement...")
    mvs, residuals_q = encode_pframe(Y1, Y0, qm)
    Y1_recon = decode_pframe(Y0, mvs, residuals_q, qm)
    print("Done !")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    axes[0].set_title("Frame 0 (référence)")
    axes[1].imshow(cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB))
    axes[1].set_title("Frame 1 (originale)")
    axes[2].imshow(cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB))
    for (by, bx), (dy, dx) in mvs:
        if dy != 0 or dx != 0:
            axes[2].annotate("", xy=(bx+dx+8, by+dy+8), xytext=(bx+8, by+8),
                             arrowprops=dict(arrowstyle="->", color='yellow', lw=1.5))
    axes[2].set_title("Vecteurs de mouvement")
    axes[3].imshow(Y1_recon, cmap='gray')
    axes[3].set_title("Frame 1 reconstruite")
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("part3_result.png")
    plt.show()

    print(f"Nombre de macroblocks   : {len(mvs)}")
    print(f"Blocs avec mouvement    : {len([(p,m) for p,m in mvs if m != (0,0)])}")