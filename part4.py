import struct
import zlib
import pickle
import numpy as np
from part1 import load_frames, bgr_to_ycbcr, ycbcr_to_bgr
from part2 import encode_channel, decode_channel, get_quant_matrix
from part3 import encode_pframe, decode_pframe

GOP_SIZE = 4
QUANTIZATION_FACTOR = 10

def encode_video(frames, qf=QUANTIZATION_FACTOR):
    qm = get_quant_matrix(qf)
    frames_data = []
    ref_Y = None

    for idx, frame in enumerate(frames):
        Y, Cb, Cr = bgr_to_ycbcr(frame)
        is_iframe = (idx % GOP_SIZE == 0)

        if is_iframe:
            Y_enc  = encode_channel(Y,  qm)
            Cb_enc = encode_channel(Cb, qm)
            Cr_enc = encode_channel(Cr, qm)
            ref_Y  = decode_channel(Y_enc, qm)
            mvs    = None
            ftype  = 'I'
        else:
            qm_residual = get_quant_matrix(qf * 2)
            mvs, Y_enc = encode_pframe(Y, ref_Y, qm_residual)
            Cb_enc = encode_channel(Cb, qm)
            Cr_enc = encode_channel(Cr, qm)
            ref_Y  = decode_pframe(ref_Y, mvs, Y_enc, qm_residual)
            ftype  = 'P'

        payload = pickle.dumps({
            'type': ftype,
            'Y':   Y_enc,
            'Cb':  Cb_enc,
            'Cr':  Cr_enc,
            'mv':  mvs,
        })
        frames_data.append(payload)

    header = struct.pack('>I', len(frames_data))
    body = b''
    for raw in frames_data:
        compressed = zlib.compress(raw, level=6)
        body += struct.pack('>I', len(compressed)) + compressed

    return header + body


def decode_video(bitstream, qf=QUANTIZATION_FACTOR):
    qm = get_quant_matrix(qf)
    num_frames = struct.unpack('>I', bitstream[:4])[0]
    offset = 4
    frames_recon = []
    ref_Y = None

    for idx in range(num_frames):
        size = struct.unpack('>I', bitstream[offset:offset+4])[0]
        offset += 4
        compressed = bitstream[offset:offset+size]
        offset += size

        d = pickle.loads(zlib.decompress(compressed))

        if d['type'] == 'I':
            Y_dec  = decode_channel(d['Y'],  qm)
            Cb_dec = decode_channel(d['Cb'], qm)
            Cr_dec = decode_channel(d['Cr'], qm)
            ref_Y  = Y_dec
        else:
            qm_residual = get_quant_matrix(qf * 2)
            Y_dec  = decode_pframe(ref_Y, d['mv'], d['Y'], qm_residual)
            Cb_dec = decode_channel(d['Cb'], qm)
            Cr_dec = decode_channel(d['Cr'], qm)
            ref_Y  = Y_dec

        frame_recon = ycbcr_to_bgr(Y_dec, Cb_dec, Cr_dec)
        frames_recon.append(frame_recon)

    return frames_recon


if __name__ == "__main__":
    frames = load_frames("frames")

    print("=== ENCODAGE ===")
    bitstream = encode_video(frames)

    with open("encoded.bin", "wb") as f:
        f.write(bitstream)

    h, w, c = frames[0].shape
    original_bytes   = len(frames) * h * w * c
    compressed_bytes = len(bitstream)
    ratio            = original_bytes / compressed_bytes

    print(f"\n=== RÉSULTATS ===")
    print(f"Taille originale   : {original_bytes:,} bytes")
    print(f"Taille compressée  : {compressed_bytes:,} bytes")
    print(f"Taux de compression: {ratio:.2f}x")

    print("\n=== DÉCODAGE ===")
    with open("encoded.bin", "rb") as f:
        bitstream_lu = f.read()

    frames_recon = decode_video(bitstream_lu)
    print(f"\n{len(frames_recon)} frames reconstruites avec succès !")