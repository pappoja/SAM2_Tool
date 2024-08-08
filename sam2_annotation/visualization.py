import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    if mask is not None and np.any(mask):
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        x_min, y_min, x_max, y_max = mask_to_bb(mask)
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def mask_to_bb(mask):
    if mask is not None and np.any(mask):
        rows, cols = np.where(mask.squeeze())
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        xyxy = (x_min, y_min, x_max, y_max)
        return xyxy
