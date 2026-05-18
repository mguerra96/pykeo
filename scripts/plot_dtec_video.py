"""
Generate a video of dTEC (vtec_detrended) over a geographic map.
Each satellite arc is drawn as a continuous colored line (lon/lat path),
colored by dtec value. One frame per epoch, 06:00–12:00 UTC.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PARQUET = r"C:\Users\MarcoGuerra\Documents\PYTHON\pykeo\results\tec_data\tec_2015-09-11.parquet"
OUTPUT  = r"C:\Users\MarcoGuerra\Documents\PYTHON\pykeo\results\dtec_2015-09-11_06-12.mp4"

VMIN, VMAX = -0.15, 0.15
FPS = 22

# ── load & filter ─────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET)
mask = (df["epoch"].dt.hour >= 6) & (df["epoch"].dt.hour < 12)
df = df[mask].sort_values(["id_arc_valid", "epoch"]).reset_index(drop=True)

epochs = sorted(df["epoch"].unique())
n_frames = len(epochs)
print(f"Frames: {n_frames}  |  FPS: {FPS}  |  Duration: {n_frames/FPS:.1f} s")

# build per-arc arrays sorted by time for fast slicing
arc_data = {}   # arc_id -> (epoch_array, lon_array, lat_array, dtec_array)
for arc_id, g in df.groupby("id_arc_valid", sort=False):
    g = g.sort_values("epoch")
    arc_data[arc_id] = (
        g["epoch"].values,
        g["lon_ipp"].values,
        g["lat_ipp"].values,
        g["vtec_detrended"].values,
    )

# epoch index lookup
epoch_index = {ep: i for i, ep in enumerate(epochs)}

# ── figure setup ──────────────────────────────────────────────────────────────
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": proj})

ax.set_extent([-15, 35, 25, 70], crs=proj)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=":")
ax.add_feature(cfeature.LAND,      facecolor="whitesmoke")
ax.add_feature(cfeature.OCEAN,     facecolor="lightcyan")
ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

norm = Normalize(vmin=VMIN, vmax=VMAX)
cmap = plt.get_cmap("RdBu_r")
sm   = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, fraction=0.03)
cbar.set_label("dTEC (TECU)")

title = ax.set_title("")

# single LineCollection that we replace each frame
lc_container = [None]

# ── animation ─────────────────────────────────────────────────────────────────
def update(frame_idx):
    current_epoch = epochs[frame_idx]
    cur_ts = pd.Timestamp(current_epoch)

    # remove previous LineCollection
    if lc_container[0] is not None:
        lc_container[0].remove()

    segments = []
    colors   = []

    for arc_eps, lons, lats, dtec in arc_data.values():
        # keep only points up to current epoch
        mask = arc_eps <= current_epoch
        if mask.sum() < 2:
            continue
        x = lons[mask]
        y = lats[mask]
        v = dtec[mask]

        # build segments: list of (N-1) pairs of consecutive points
        pts = np.column_stack([x, y])          # (N, 2)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (N-1, 2, 2)
        seg_vals = (v[:-1] + v[1:]) / 2        # color per segment = midpoint value

        segments.append(segs)
        colors.append(seg_vals)

    if segments:
        all_segs   = np.concatenate(segments, axis=0)
        all_colors = np.concatenate(colors,   axis=0)
        lc = LineCollection(all_segs, cmap=cmap, norm=norm,
                            linewidth=1.2, transform=proj, zorder=5)
        lc.set_array(all_colors)
        ax.add_collection(lc)
        lc_container[0] = lc
    else:
        lc_container[0] = None

    title.set_text(f"dTEC — {cur_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    return []

ani = animation.FuncAnimation(fig, update, frames=n_frames,
                               blit=False, interval=1000 / FPS)

writer = animation.FFMpegWriter(fps=FPS, bitrate=3000,
                                extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
ani.save(OUTPUT, writer=writer, dpi=120)
print(f"Saved: {OUTPUT}")
