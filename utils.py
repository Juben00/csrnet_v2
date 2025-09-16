# utils.py
import os
import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

def load_json_annotations(json_path):
    """
    Supports common formats:
    - VIA style: { "filename.jpg": { "filename": "filename.jpg", "regions": [ {"shape_attributes":{"name":"point","cx":..,"cy":..}} ] } }
    - Simple dict: { "filename": {"regions": [...] } }
    - COCO-like: {"images":[{"file_name":"...","id":id}], "annotations":[{"image_id":id,"points":[ [x,y],... ]}]}
    Returns: dict mapping filename -> list of (x,y) points (ints or floats)
    """
    with open(json_path,'r') as f:
        data = json.load(f)
    points_dict = {}

    # VIA-like detection
    if isinstance(data, dict) and any('.jpg' in k or '.png' in k for k in data.keys()):
        for key, val in data.items():
            # some VIA exports use key as filename or id; try robust approach
            if isinstance(val, dict) and 'filename' in val:
                fname = val['filename']
            else:
                fname = key
            regs = val.get('regions', [])
            pts = []
            for r in regs:
                sa = r.get('shape_attributes', {})
                if sa.get('name') == 'point':
                    cx = sa.get('cx')
                    cy = sa.get('cy')
                    if cx is not None and cy is not None:
                        pts.append((float(cx), float(cy)))
                # support circle/ellipse? skip for now
            points_dict[fname] = pts
        return points_dict

    # COCO-like
    if isinstance(data, dict) and 'images' in data and 'annotations' in data:
        id_to_fname = {img['id']: img['file_name'] for img in data['images']}
        for ann in data['annotations']:
            img_id = ann['image_id']
            fname = id_to_fname.get(img_id)
            if fname is None:
                continue
            pts = []
            # support if points stored as list of [x,y] in ann.get('points') or 'keypoints' or 'segmentation'
            if 'points' in ann:
                pts = [(float(x), float(y)) for [x,y] in ann['points']]
            elif 'keypoints' in ann:
                kp = ann['keypoints']
                # COCO keypoints come as [x,y,v,...]. Try to parse every 3.
                pts = [(float(kp[i]), float(kp[i+1])) for i in range(0,len(kp),3) if kp[i] != 0 or kp[i+1] != 0]
            else:
                # fallback if segmentation with single-point polygons
                seg = ann.get('segmentation', [])
                if seg and isinstance(seg, list):
                    # try first polygon
                    flat = seg[0]
                    pts = [(float(flat[i]), float(flat[i+1])) for i in range(0,len(flat),2)]
            points_dict.setdefault(fname, []).extend(pts)
        return points_dict

    # otherwise, assume simple dict mapping
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict) and 'regions' in v:
                regs = v['regions']
                pts = []
                for r in regs:
                    sa = r.get('shape_attributes', {})
                    if sa.get('name') == 'point':
                        cx, cy = sa.get('cx'), sa.get('cy')
                        if cx is not None and cy is not None:
                            pts.append((float(cx), float(cy)))
                points_dict[k] = pts
    return points_dict


def generate_density_map(image_shape, points, adaptive=True, fixed_sigma=4):
    """
    image_shape: (H, W)
    points: list of (x, y) pixel coordinates (x across width, y across height)
    returns density map (H,W) float32 where sum equals number of points
    Methods:
    - Adaptive sigma: sigma = max(1.0, mean distance to k nearest neighbors) * 0.3
    - Fixed sigma: use fixed_sigma
    """
    H, W = image_shape
    density = np.zeros((H, W), dtype=np.float32)
    if len(points) == 0:
        return density

    pts = np.array(points, dtype=np.float32)
    # Ensure points inside bounds
    pts[:,0] = np.clip(pts[:,0], 0, W-1)
    pts[:,1] = np.clip(pts[:,1], 0, H-1)

    if adaptive and len(pts) > 1:
        tree = KDTree(pts)
        # query 4 nearest (including itself)
        distances, _ = tree.query(pts, k=4)
        # distances[:,1:] are distances to neighbors; take mean of first neighbor distances
        # to avoid zeros, fallback to fixed_sigma
        sigmas = []
        for i in range(len(pts)):
            # skip self at distances[i,0]==0
            dists = distances[i,1:]
            mean_dist = np.mean(dists)
            sigma = max(1.0, mean_dist*0.3)
            sigmas.append(sigma)
    else:
        sigmas = [fixed_sigma for _ in range(len(pts))]

    # Place gaussian for each point
    for (x, y), sigma in zip(pts, sigmas):
        # create small gaussian patch to add
        x = int(round(x))
        y = int(round(y))
        # radius = 3*sigma
        radius = int(3 * sigma)
        x1 = max(0, x - radius)
        x2 = min(W, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(H, y + radius + 1)

        patch_w = x2 - x1
        patch_h = y2 - y1
        if patch_w <=0 or patch_h <= 0:
            continue

        # grid
        xx = np.arange(x1, x2)
        yy = np.arange(y1, y2)
        xv, yv = np.meshgrid(xx, yy)
        gaussian = np.exp(-((xv - x)**2 + (yv - y)**2) / (2 * sigma * sigma))
        # normalize so patch sums to 1
        gaussian = gaussian / (gaussian.sum() + 1e-12)
        density[y1:y2, x1:x2] += gaussian

    return density


def save_density_map_as_h5(density, path):
    import h5py
    with h5py.File(path, 'w') as f:
        f.create_dataset('density', data=density, compression='gzip')
