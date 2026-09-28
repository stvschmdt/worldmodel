"""Turn an episode into an annotated, upscaled GIF you can watch in VS Code.

Each frame shows the 64x64 observation upscaled, the path so far, the action about
to be taken (arrow), and a red border on steps where that action hit a wall.
Reused later to watch model rollouts next to the real episode.
"""

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

TRACE = (40, 40, 40)
ARROW = (30, 110, 200)
BUMP = (210, 40, 40)
HEADER_H = 22


def episode_frames(obs, actions=None, pos=None, collided=None, scale=6, title=""):
    """obs (T+1, H, W, 3) uint8; actions (T, 2); pos (T+1, 2) in [0, 1]; collided (T,).
    Everything but obs is optional. Returns a list of annotated RGB uint8 frames."""
    T1, H, W, _ = obs.shape
    S = H * scale
    frames = []
    for t in range(T1):
        img = Image.fromarray(obs[t]).resize((W * scale, S), Image.NEAREST)
        canvas = Image.new("RGB", (W * scale, S + HEADER_H), "white")
        canvas.paste(img, (0, HEADER_H))
        draw = ImageDraw.Draw(canvas)

        def px(p):  # world coords -> canvas pixels
            return (float(p[0]) * W * scale, float(p[1]) * S + HEADER_H)

        if pos is not None and t > 0:
            draw.line([px(p) for p in pos[: t + 1]], fill=TRACE, width=2)
        a = actions[t] if actions is not None and t < len(actions) else None
        if a is not None and pos is not None:
            # Arrow length: the full commanded displacement, drawn 3x so it's visible.
            x0, y0 = px(pos[t])
            x1, y1 = x0 + a[0] * 0.05 * 3 * W * scale, y0 + a[1] * 0.05 * 3 * S
            draw.line([(x0, y0), (x1, y1)], fill=ARROW, width=3)
            draw.ellipse([x1 - 3, y1 - 3, x1 + 3, y1 + 3], fill=ARROW)
        hit = collided is not None and t < len(collided) and bool(collided[t])
        if hit:
            draw.rectangle([0, HEADER_H, W * scale - 1, S + HEADER_H - 1], outline=BUMP, width=4)

        text = f"{title}  t={t:3d}/{T1 - 1}"
        if a is not None:
            text += f"  a=({a[0]:+.2f}, {a[1]:+.2f})"
        if hit:
            text += "  BUMP"
        draw.text((6, 5), text, fill=BUMP if hit else (0, 0, 0))
        frames.append(np.asarray(canvas))
    return frames


def save_gif(frames, path, fps=10):
    # Hold the last frame so the loop point is obvious.
    imageio.mimsave(path, frames + [frames[-1]] * fps, duration=1000 / fps, loop=0)
