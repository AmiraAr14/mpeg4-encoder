import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_frames(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    frames = [cv2.imread(os.path.join(folder, f)) for f in files]
    print(f"Chargé {len(frames)} frames")
    return frames

def bgr_to_ycbcr(frame_bgr):
    frame_ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    Y  = frame_ycrcb[:, :, 0].astype(np.float32)
    Cr = frame_ycrcb[:, :, 1].astype(np.float32)
    Cb = frame_ycrcb[:, :, 2].astype(np.float32)
    Cb_sub = Cb[::2, ::2]
    Cr_sub = Cr[::2, ::2]
    return Y, Cb_sub, Cr_sub

def ycbcr_to_bgr(Y, Cb_sub, Cr_sub):
    h, w = Y.shape
    Cb = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    ycrcb = np.stack([Y, Cr, Cb], axis=2).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

if __name__ == "__main__":
    frames = load_frames("frames")
    frame  = frames[0]

    Y, Cb, Cr = bgr_to_ycbcr(frame)
    reconstructed = ycbcr_to_bgr(Y, Cb, Cr)

    print(f"Frame originale : {frame.shape}")
    print(f"Y               : {Y.shape}")
    print(f"Cb              : {Cb.shape}")
    print(f"Cr              : {Cr.shape}")

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[1].imshow(Y, cmap='gray')
    axes[1].set_title(f"Y (luminosité)\n{Y.shape}")
    axes[2].imshow(Cb, cmap='Blues')
    axes[2].set_title(f"Cb (chroma bleu)\n{Cb.shape}")
    axes[3].imshow(Cr, cmap='Reds')
    axes[3].set_title(f"Cr (chroma rouge)\n{Cr.shape}")
    axes[4].imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
    axes[4].set_title("Reconstruit")
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("part1_result.png")
    plt.show()
    print("Image sauvegardée : part1_result.png")