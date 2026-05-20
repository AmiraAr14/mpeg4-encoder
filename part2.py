import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from part1 import load_frames, bgr_to_ycbcr, ycbcr_to_bgr

DCT_BLOCK = 8
QUANTIZATION_FACTOR = 10

BASE_QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float32)

def get_quant_matrix(factor):
    return np.clip(BASE_QUANT_MATRIX * factor / 10, 1, 255).astype(np.float32)

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def encode_channel(channel, quant_matrix):
    h, w = channel.shape
    ph = (DCT_BLOCK - h % DCT_BLOCK) % DCT_BLOCK
    pw = (DCT_BLOCK - w % DCT_BLOCK) % DCT_BLOCK
    padded = np.pad(channel, ((0, ph), (0, pw)), mode='edge')
    out = np.zeros_like(padded, dtype=np.int16)
    for i in range(0, padded.shape[0], DCT_BLOCK):
        for j in range(0, padded.shape[1], DCT_BLOCK):
            block = padded[i:i+DCT_BLOCK, j:j+DCT_BLOCK] - 128.0
            coeffs = dct2(block)
            out[i:i+DCT_BLOCK, j:j+DCT_BLOCK] = np.round(coeffs / quant_matrix).astype(np.int16)
    return out[:h, :w]

def decode_channel(coeffs, quant_matrix):
    h, w = coeffs.shape
    ph = (DCT_BLOCK - h % DCT_BLOCK) % DCT_BLOCK
    pw = (DCT_BLOCK - w % DCT_BLOCK) % DCT_BLOCK
    padded = np.pad(coeffs, ((0, ph), (0, pw)), mode='constant')
    out = np.zeros_like(padded, dtype=np.float32)
    for i in range(0, padded.shape[0], DCT_BLOCK):
        for j in range(0, padded.shape[1], DCT_BLOCK):
            block = padded[i:i+DCT_BLOCK, j:j+DCT_BLOCK].astype(np.float32)
            dequant = block * quant_matrix
            out[i:i+DCT_BLOCK, j:j+DCT_BLOCK] = idct2(dequant) + 128.0
    return np.clip(out[:h, :w], 0, 255)

if __name__ == "__main__":
    frames = load_frames("frames")
    frame  = frames[0]
    qm     = get_quant_matrix(QUANTIZATION_FACTOR)

    Y, Cb, Cr = bgr_to_ycbcr(frame)

    Y_enc  = encode_channel(Y,  qm)
    Cb_enc = encode_channel(Cb, qm)
    Cr_enc = encode_channel(Cr, qm)

    Y_dec  = decode_channel(Y_enc,  qm)
    Cb_dec = decode_channel(Cb_enc, qm)
    Cr_dec = decode_channel(Cr_enc, qm)

    reconstructed = ycbcr_to_bgr(Y_dec, Cb_dec, Cr_dec)

    best_var, best_by, best_bx = 0, 32, 32
    for by in range(0, Y.shape[0] - 8, 8):
        for bx in range(0, Y.shape[1] - 8, 8):
            v = np.var(Y[by:by+8, bx:bx+8])
            if v > best_var:
                best_var, best_by, best_bx = v, by, bx
    block_orig   = Y[best_by:best_by+8, best_bx:best_bx+8] - 128
    block_dct    = dct2(block_orig)
    block_quant  = np.round(block_dct / qm).astype(np.int16)
    block_recon  = idct2(block_quant.astype(np.float32) * qm) + 128

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0][0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0][0].set_title("Original")
    axes[0][1].imshow(Y, cmap='gray')
    axes[0][1].set_title("Y original")
    axes[0][2].imshow(Y_dec, cmap='gray')
    axes[0][2].set_title("Y reconstruit")
    axes[0][3].imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
    axes[0][3].set_title("Image reconstruite")
    axes[1][0].imshow(block_orig + 128, cmap='gray', interpolation='nearest')
    axes[1][0].set_title("Bloc 8x8 original")
    axes[1][1].imshow(np.abs(block_dct), cmap='hot', interpolation='nearest')
    axes[1][1].set_title("Après DCT")
    axes[1][2].imshow(np.abs(block_quant.astype(float)), cmap='hot', interpolation='nearest')
    axes[1][2].set_title("Après quantization")
    axes[1][3].imshow(block_recon, cmap='gray', interpolation='nearest')
    axes[1][3].set_title("Bloc reconstruit")
    for row in axes:
        for ax in row:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("part2_result.png")
    plt.show()

    psnr = 10 * np.log10(255**2 / np.mean((Y - Y_dec)**2))
    print(f"PSNR : {psnr:.1f} dB")