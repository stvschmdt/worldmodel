"""PointMaze-lite: a tiny 2D navigation world rendered to 64x64 RGB images.

The world is the unit square [0, 1]^2. An agent (a disk) moves through rooms
separated by walls with doorways. Actions are 2D displacement commands in
[-1, 1]^2, scaled by `max_step`. Collisions slide along walls (x then y).

Because we render it ourselves, we always know the true state (x, y), which is
what lets us probe what a world model has learned.
"""

from dataclasses import dataclass

import numpy as np

# Walls are axis-aligned rectangles (x0, y0, x1, y1) in world coordinates.
# y points *down* in images, so y=0 is the top row.
_T = 0.04  # wall thickness
_BORDER = [
    (0.0, 0.0, 1.0, _T),
    (0.0, 1.0 - _T, 1.0, 1.0),
    (0.0, 0.0, _T, 1.0),
    (1.0 - _T, 0.0, 1.0, 1.0),
]
_C0, _C1 = 0.5 - _T / 2, 0.5 + _T / 2  # center wall span
_DOOR = 0.16

LAYOUTS = {
    "open": _BORDER,
    "two_rooms": _BORDER + [
        (_C0, 0.0, _C1, 0.5 - _DOOR / 2),
        (_C0, 0.5 + _DOOR / 2, _C1, 1.0),
    ],
    "four_rooms": _BORDER + [
        # vertical wall, doors at y=0.25 and y=0.75
        (_C0, 0.0, _C1, 0.25 - _DOOR / 2),
        (_C0, 0.25 + _DOOR / 2, _C1, 0.75 - _DOOR / 2),
        (_C0, 0.75 + _DOOR / 2, _C1, 1.0),
        # horizontal wall, doors at x=0.25 and x=0.75
        (0.0, _C0, 0.25 - _DOOR / 2, _C1),
        (0.25 + _DOOR / 2, _C0, 0.75 - _DOOR / 2, _C1),
        (0.75 + _DOOR / 2, _C0, 1.0, _C1),
    ],
}

BG = np.array([236, 234, 226], dtype=np.float32)
WALL = np.array([58, 60, 72], dtype=np.float32)
AGENT = np.array([222, 72, 44], dtype=np.float32)
GOAL = np.array([48, 132, 214], dtype=np.float32)


@dataclass
class PointMazeConfig:
    layout: str = "four_rooms"
    size: int = 64  # image is size x size
    radius: float = 0.045  # agent radius (world units)
    max_step: float = 0.05  # displacement per step at |a| = 1


class PointMaze:
    def __init__(self, cfg: PointMazeConfig = PointMazeConfig(), seed=None):
        self.cfg = cfg
        self.walls = np.array(LAYOUTS[cfg.layout], dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(2, dtype=np.float32)
        self.goal = None

        # Pixel-center coordinates, shape (size, size), used for rendering.
        c = (np.arange(cfg.size, dtype=np.float32) + 0.5) / cfg.size
        self._px, self._py = np.meshgrid(c, c)  # x varies along columns
        wall_mask = np.zeros((cfg.size, cfg.size), dtype=bool)
        for x0, y0, x1, y1 in self.walls:
            wall_mask |= (self._px >= x0) & (self._px < x1) & (self._py >= y0) & (self._py < y1)
        self._background = np.where(wall_mask[..., None], WALL, BG)

    # ---- geometry -------------------------------------------------------
    def collides(self, p) -> bool:
        """True if a disk of the agent's radius at p overlaps any wall."""
        x0, y0, x1, y1 = self.walls.T
        dx = np.maximum(np.maximum(x0 - p[0], 0.0), p[0] - x1)
        dy = np.maximum(np.maximum(y0 - p[1], 0.0), p[1] - y1)
        return bool(np.any(dx * dx + dy * dy < self.cfg.radius ** 2))

    def sample_free(self):
        while True:
            p = self.rng.uniform(0.0, 1.0, size=2).astype(np.float32)
            if not self.collides(p):
                return p

    # ---- env API --------------------------------------------------------
    def reset(self, pos=None, goal=None):
        self.pos = np.asarray(pos, dtype=np.float32) if pos is not None else self.sample_free()
        self.goal = None if goal is None else np.asarray(goal, dtype=np.float32)
        return self.render()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        delta = a * self.cfg.max_step
        new = self.pos + delta
        hit = self.collides(new)
        if hit:  # slide: try moving along one axis at a time
            new = self.pos.copy()
            for axis in (0, 1):
                trial = new.copy()
                trial[axis] += delta[axis]
                if not self.collides(trial):
                    new = trial
        self.pos = new
        return self.render(), {"pos": self.pos.copy(), "collided": hit}

    def render(self, pos=None):
        """Return a (size, size, 3) uint8 image. Soft disk edges encode sub-pixel position."""
        img = self._background.copy()
        if self.goal is not None:
            img = self._draw_disk(img, self.goal, GOAL, ring=True)
        img = self._draw_disk(img, self.pos if pos is None else pos, AGENT)
        return img.astype(np.uint8)

    def _draw_disk(self, img, center, color, ring=False):
        d = np.hypot(self._px - center[0], self._py - center[1])
        r = self.cfg.radius
        alpha = np.clip((r - d) * self.cfg.size + 0.5, 0.0, 1.0)  # ~1px anti-aliased edge
        if ring:
            inner = np.clip((r * 0.45 - d) * self.cfg.size + 0.5, 0.0, 1.0)
            alpha = alpha - inner
        return img * (1 - alpha[..., None]) + color * alpha[..., None]


def random_actions(n, rng, smooth=0.8, scale=0.6):
    """Temporally correlated random actions, so random walks actually travel."""
    a = np.zeros((n, 2), dtype=np.float32)
    prev = rng.normal(0, 1, 2)
    for t in range(n):
        prev = smooth * prev + scale * rng.normal(0, 1, 2)
        a[t] = np.clip(prev, -1, 1)
    return a


def collect_trajectory(env: PointMaze, T: int, rng=None):
    """Roll out a random policy. Returns obs (T+1, H, W, 3), actions (T, 2), pos (T+1, 2)."""
    rng = env.rng if rng is None else rng
    obs, pos = [env.reset()], [env.pos.copy()]
    actions = random_actions(T, rng)
    for a in actions:
        o, info = env.step(a)
        obs.append(o)
        pos.append(info["pos"])
    return np.stack(obs), actions, np.stack(pos)
