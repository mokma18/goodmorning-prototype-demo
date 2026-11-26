"""Make a flat-backed GLB from a single photo and its SAM 2 silhouette."""

import argparse
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def mask_photo(photo_path, output, checkpoint, box):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_num_threads(min(4, torch.get_num_threads()))
    with Image.open(photo_path) as source:
        photo = np.asarray(source.convert("RGB"))
    h, w = photo.shape[:2]
    predictor = SAM2ImagePredictor(build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml", str(checkpoint), device="cpu",
        apply_postprocessing=False,
    ))
    with torch.inference_mode():
        predictor.set_image(photo)
        masks, scores, _ = predictor.predict(
            box=np.asarray(box, np.float32) * [w, h, w, h], multimask_output=False
        )
    Image.fromarray(masks[scores.argmax()].astype(np.uint8) * 255).save(output)


def relief(photo_path, mask_path, output):
    from scipy import ndimage

    with Image.open(photo_path) as source:
        photo = np.asarray(source.convert("RGB"))
    with Image.open(mask_path) as source:
        mask = np.asarray(source.convert("L")) > 127
    labels, count = ndimage.label(mask)
    if not count or mask.sum() < 32:
        raise ValueError("No usable collectible mask.")
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    mask = labels == areas.argmax()
    ys, xs = np.nonzero(mask)
    long_side = max(np.ptp(xs), np.ptp(ys)) + 1
    border = max(2, round(long_side * .035))
    pad = border + 6
    bounds = (slice(max(0, ys.min() - pad), ys.max() + pad + 1),
              slice(max(0, xs.min() - pad), xs.max() + pad + 1))
    mask = np.pad(mask[bounds], pad)
    photo = np.pad(photo[bounds], ((pad, pad), (pad, pad), (0, 0)))
    cut = ndimage.distance_transform_edt(~mask) <= border
    alpha = ndimage.gaussian_filter(cut.astype(float), .65)
    texture = np.dstack((np.where(mask[..., None], photo, [251, 249, 243]),
                         np.round(alpha * 255))).astype(np.uint8)
    png = io.BytesIO()
    Image.fromarray(texture).save(png, format="PNG")

    h, w = cut.shape
    scale = .06 / (long_side + 2 * border)
    distance = ndimage.distance_transform_edt(cut)
    height = .004 * np.sqrt(np.clip((distance - 1.5) / (long_side * .15), 0, 1))
    dy, dx = np.gradient(height, scale)
    stride = max(2, long_side // 130)
    y, x = np.mgrid[:h:stride, :w:stride]
    pos = np.stack(((x - (w - 1) / 2) * scale, ((h - 1) / 2 - y) * scale, height[y, x]), -1)
    normal = np.stack((-dx[y, x], dy[y, x], np.ones_like(x)), -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    uv = np.stack(((x + .5) / w, (y + .5) / h), -1)
    adjacent = ndimage.maximum_filter(cut, size=2 * stride + 1)[y, x]
    iy, ix = np.nonzero(adjacent[:-1, :-1])
    columns = x.shape[1]
    a = iy * columns + ix
    b = a + columns
    front = np.concatenate((np.stack((a, b, a + 1), 1), np.stack((a + 1, b, b + 1), 1)))
    pos = pos.reshape(-1, 3)
    back = pos.copy()
    back[:, 2] = 0
    positions = np.concatenate((pos, back)).astype("<f4")
    normals = np.concatenate((normal.reshape(-1, 3), np.tile([0, 0, -1], (len(back), 1)))).astype("<f4")
    texcoords = np.tile(uv.reshape(-1, 2), (2, 1)).astype("<f4")
    triangles = np.concatenate((front, front[:, ::-1] + len(back))).astype("<u4").ravel()
    write_glb(output, positions, normals, texcoords, triangles, png.getvalue())


def write_glb(output, positions, normals, uv, triangles, texture):
    data = bytearray()
    views = []
    for content in (positions.tobytes(), normals.tobytes(), uv.tobytes(), triangles.tobytes(), texture):
        data.extend(b"\0" * (-len(data) % 4))
        views.append({"buffer": 0, "byteOffset": len(data), "byteLength": len(content)})
        data.extend(content)
    accessors = [
        {"bufferView": i, "componentType": component, "count": len(array), "type": shape}
        for i, (component, array, shape) in enumerate(
            ((5126, positions, "VEC3"), (5126, normals, "VEC3"),
             (5126, uv, "VEC2"), (5125, triangles, "SCALAR"))
        )
    ]
    accessors[0].update(min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist())
    model = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                                     "indices": 3, "material": 0}]}],
        "extensionsUsed": ["KHR_materials_clearcoat"],
        "materials": [{
            "alphaMode": "MASK",
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": .42},
            "extensions": {"KHR_materials_clearcoat": {"clearcoatFactor": 1, "clearcoatRoughnessFactor": .09}},
        }],
        "textures": [{"source": 0}], "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "buffers": [{"byteLength": len(data)}], "bufferViews": views, "accessors": accessors,
    }
    document = json.dumps(model, separators=(",", ":")).encode()
    document += b" " * (-len(document) % 4)
    data.extend(b"\0" * (-len(data) % 4))
    with output.open("wb") as result:
        result.write(struct.pack("<4sII", b"glTF", 2, 28 + len(document) + len(data)))
        result.write(struct.pack("<I4s", len(document), b"JSON"))
        result.write(document)
        result.write(struct.pack("<I4s", len(data), b"BIN\0"))
        result.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--box", type=float, nargs=4, required=True)
    args = parser.parse_args()
    mask_photo(args.photo, args.mask, args.checkpoint, args.box)
