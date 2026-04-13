# Self Driving Car - Improved

import os
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque

# Config must be set before any other Kivy imports
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'width',  '1200')
Config.set('graphics', 'height', '700')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Ellipse, Line, Rectangle, InstructionGroup
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty, ObjectProperty
from kivy.clock import Clock
from kivy.core.window import Window

from ai import Dqn
from environment import Environment

Window.clearcolor = (0.07, 0.08, 0.13, 1)

# ── Globals ────────────────────────────────────────────────────────────────────

last_x        = 0
last_y        = 0
brush_size    = 10
draw_mode     = True    # True = paint sand, False = erase sand
current_shape = 'freehand'   # 'freehand' | 'rectangle' | 'circle' | 'line'
show_sensor_display = True    # toggle sensor ray display (sensors always active)
undo_stack    = []             # list of sand array snapshots for undo
MAX_UNDO      = 20
goals_reached = 0       # how many times the goal has been reached

MAP_GRID = 16   # sand map is downsampled to MAP_GRID × MAP_GRID for the network
# 12 local inputs (10 + car x/y) + MAP_GRID² map cells
brain           = Dqn(12 + MAP_GRID * MAP_GRID, 3, 0.99)
action2rotation = [0, 20, -20]
last_reward     = 0
scores          = []

first_update  = True
sand          = np.zeros((1, 1))   # placeholder until init() runs with real dimensions

def _push_undo():
    if sand.shape[0] > 1:   # skip the placeholder
        undo_stack.append(sand.copy())
        if len(undo_stack) > MAX_UNDO:
            undo_stack.pop(0)

def init():
    global sand, first_update
    sand         = np.zeros((longueur, largeur))
    first_update = False

def _get_map_input():
    """Downsample the full sand array to MAP_GRID×MAP_GRID and return as a flat list."""
    if sand.shape[0] < MAP_GRID or sand.shape[1] < MAP_GRID:
        return [0.0] * (MAP_GRID * MAP_GRID)
    bh  = sand.shape[0] // MAP_GRID
    bw  = sand.shape[1] // MAP_GRID
    arr = sand[:bh * MAP_GRID, :bw * MAP_GRID]
    ds  = arr.reshape(MAP_GRID, bh, MAP_GRID, bw).mean(axis=(1, 3))
    return ds.flatten().tolist()

# ── Car widget — display only, all logic lives in Environment ──────────────────

class Car(Widget):
    angle      = NumericProperty(0)
    is_on_sand = NumericProperty(0)


# ── Sensor ball widgets ────────────────────────────────────────────────────────

class Ball1(Widget): pass
class Ball2(Widget): pass
class Ball3(Widget): pass
class Ball4(Widget): pass
class Ball5(Widget): pass


# ── Score graph widget ─────────────────────────────────────────────────────────

class ScoreGraph(Widget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scores = deque(maxlen=200)
        with self.canvas:
            Color(0.05, 0.06, 0.10, 1)
            self._bg     = Rectangle(pos=self.pos, size=self.size)
            Color(0.20, 0.32, 0.20, 0.70)
            self._border = Line(rectangle=(self.x, self.y, self.width, self.height), width=1)
            Color(0.40, 0.40, 0.40, 0.40)
            self._zero   = Line(points=[0, 0, 0, 0], width=0.8)
            self._lc     = Color(0.25, 1, 0.55, 0.90)
            self._line   = Line(points=[0, 0], width=1.5)
            self._dc     = Color(1, 1, 0.30, 0.90)
            self._dot    = Ellipse(pos=(0, 0), size=(7, 7))
        self.bind(size=self._redraw, pos=self._redraw)

    def push(self, score):
        self._scores.append(score)
        self._redraw()

    def clear(self):
        self._scores.clear()
        self._redraw()

    def _redraw(self, *args):
        px, py, pw, ph = self.x + 3, self.y + 3, self.width - 6, self.height - 6
        self._bg.pos            = (self.x, self.y)
        self._bg.size           = (self.width, self.height)
        self._border.rectangle  = (self.x, self.y, self.width, self.height)
        zy = py + max(0.0, min(1.0, 1.0 / 1.5)) * ph
        self._zero.points = [px, zy, px + pw, zy]

        n = len(self._scores)
        if n < 2:
            return
        pts = []
        for i, s in enumerate(self._scores):
            gx = px + (i / (n - 1)) * pw
            gy = py + max(0.0, min(1.0, (s + 1.0) / 1.5)) * ph
            pts.extend([gx, gy])
        self._line.points = pts

        cur   = self._scores[-1]
        cur_y = py + max(0.0, min(1.0, (cur + 1.0) / 1.5)) * ph
        self._dot.pos = (px + pw - 5, cur_y - 3)
        self._lc.r = 0.25 if cur >= 0 else min(1.0, 0.25 + abs(cur) * 0.75)
        self._lc.g = min(1.0, 0.35 + max(0, cur + 0.4) * 1.6)
        self._dc.r = 0.3 if cur >= 0 else 1.0
        self._dc.g = 1.0 if cur >= 0 else 0.4


# ── Q-value bar widget ─────────────────────────────────────────────────────────

class QValueWidget(Widget):
    """Live vertical bar chart showing the Q-value for each action."""

    _LABELS = ['Str', 'L', 'R']           # action2rotation = [0, 20, -20]
    _COLORS = [
        (0.28, 1.00, 0.55),               # straight — green
        (0.30, 0.65, 1.00),               # left     — blue
        (1.00, 0.55, 0.20),               # right    — orange
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._q = [0.0, 0.0, 0.0]
        with self.canvas:
            Color(0.05, 0.06, 0.10, 1)
            self._bg     = Rectangle(pos=self.pos, size=self.size)
            Color(0.18, 0.20, 0.32, 1)
            self._border = Line(rectangle=(self.x, self.y, self.width, self.height),
                                width=1)
        self._bar_instrs = []   # list of (Color, Rectangle) pairs per action
        self._lbl_instrs = []
        self.bind(size=self._rebuild, pos=self._rebuild)

    def _rebuild(self, *_):
        # Remove old bar/label instructions
        for c, r in self._bar_instrs:
            self.canvas.remove(c)
            self.canvas.remove(r)
        for lbl_grp in self._lbl_instrs:
            self.canvas.remove(lbl_grp)
        self._bar_instrs.clear()
        self._lbl_instrs.clear()

        self._bg.pos            = self.pos
        self._bg.size           = self.size
        self._border.rectangle  = (self.x, self.y, self.width, self.height)
        self._redraw()

    def update(self, q_values):
        self._q = list(q_values)
        self._redraw()

    def _redraw(self, *_):
        for c, r in self._bar_instrs:
            self.canvas.remove(c)
            self.canvas.remove(r)
        self._bar_instrs.clear()

        pad   = 6
        n     = len(self._q)
        w     = (self.width  - pad * (n + 1)) / n
        mid_y = self.y + self.height / 2
        scale = (self.height / 2 - pad * 2) / max(1.0, max(abs(v) for v in self._q))

        for i, (q, rgb) in enumerate(zip(self._q, self._COLORS)):
            bx  = self.x + pad + i * (w + pad)
            bh  = q * scale
            by  = mid_y if bh >= 0 else mid_y + bh
            bh  = abs(bh)
            c   = Color(*rgb, 0.85)
            r   = Rectangle(pos=(bx, by), size=(w, max(1, bh)))
            self.canvas.add(c)
            self.canvas.add(r)
            self._bar_instrs.append((c, r))


# ── Status indicator (coloured dot + label row) ────────────────────────────────

class StatusDot(Widget):
    """Small canvas-drawn circle used as a reliable mode indicator."""

    def __init__(self, color=(0.28, 1, 0.55, 1), **kwargs):
        super().__init__(size_hint=(None, None), size=(10, 10), **kwargs)
        with self.canvas:
            self._c = Color(*color)
            self._e = Ellipse(pos=self.pos, size=self.size)
        self.bind(pos=self._sync, size=self._sync)

    def set_color(self, rgba):
        self._c.rgba = rgba

    def _sync(self, *args):
        self._e.pos  = self.pos
        self._e.size = self.size


# ── Panel helpers ──────────────────────────────────────────────────────────────

class _Divider(Widget):
    def __init__(self, **kwargs):
        super().__init__(size_hint_y=None, height=1, **kwargs)
        with self.canvas:
            Color(0.18, 0.20, 0.32, 1)
            self._rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(size=self._sync, pos=self._sync)

    def _sync(self, *args):
        self._rect.pos  = self.pos
        self._rect.size = self.size


def _section_header(text):
    lbl = Label(
        text=text, font_size=10, bold=True,
        color=(0.45, 0.55, 0.45, 0.75),
        size_hint_y=None, height=20,
        halign='left',
    )
    lbl.bind(size=lambda inst, v: setattr(inst, 'text_size', (v[0], None)))
    return lbl


def _stat_label(text, color):
    lbl = Label(
        text=text, color=color,
        size_hint_y=None, height=22, font_size=13, bold=True, halign='left',
    )
    lbl.bind(size=lambda inst, v: setattr(inst, 'text_size', (v[0], None)))
    return lbl


# ── Control panel ──────────────────────────────────────────────────────────────

class ControlPanel(BoxLayout):

    def __init__(self, **kwargs):
        kwargs.setdefault('orientation', 'vertical')
        kwargs.setdefault('padding',     [16, 14, 16, 10])
        kwargs.setdefault('spacing',     5)
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(0.08, 0.09, 0.15, 1)
            self._bg     = Rectangle(pos=self.pos, size=self.size)
            Color(0.18, 0.20, 0.32, 1)
            self._border = Line(rectangle=(self.x, self.y, self.width, self.height), width=1.5)
        self.bind(size=self._sync, pos=self._sync)

    def _sync(self, *args):
        self._bg.pos           = self.pos
        self._bg.size          = self.size
        self._border.rectangle = (self.x, self.y, self.width, self.height)


# ── Game widget ────────────────────────────────────────────────────────────────

class Game(Widget):

    car   = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    ball4 = ObjectProperty(None)
    ball5 = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pulse       = 0
        self._flash_alpha = 0.0
        self.env          = None    # created lazily on first update

        with self.canvas.before:
            Color(0.10, 0.11, 0.18, 1)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)

            self._gc_glow = Color(0.25, 1, 0.55, 0.2)
            self._g_glow  = Ellipse(pos=(0, 0), size=(50, 50))
            Color(0.25, 1, 0.55, 0.85)
            self._g_ring  = Line(circle=(0, 0, 5), width=2)
            Color(0.25, 1, 0.55, 0.95)
            self._g_dot   = Ellipse(pos=(0, 0), size=(10, 10))
            Color(0.25, 1, 0.55, 0.65)
            self._g_h     = Line(points=[0, 0, 0, 0], width=1.5)
            self._g_v     = Line(points=[0, 0, 0, 0], width=1.5)

            self._arrow_color = Color(0.45, 0.90, 0.45, 0.0)
            self._arrow_line  = Line(points=[0, 0, 0, 0], width=1.5)

            self._sc1 = Color(0, 1, 0.5, 0.8); self._sl1 = Line(points=[0,0,0,0], width=1.3)
            self._sc2 = Color(0, 1, 0.5, 0.8); self._sl2 = Line(points=[0,0,0,0], width=1.3)
            self._sc3 = Color(0, 1, 0.5, 0.8); self._sl3 = Line(points=[0,0,0,0], width=1.3)
            self._sc4 = Color(0, 1, 0.5, 0.8); self._sl4 = Line(points=[0,0,0,0], width=1.3)
            self._sc5 = Color(0, 1, 0.5, 0.8); self._sl5 = Line(points=[0,0,0,0], width=1.3)

        with self.canvas.after:
            self._flash_color = Color(1, 0.92, 0.55, 0)
            self._flash_rect  = Rectangle(pos=self.pos, size=self.size)

        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, *args):
        self.bg_rect.pos      = self.pos
        self.bg_rect.size     = self.size
        self._flash_rect.pos  = self.pos
        self._flash_rect.size = self.size

    def serve_car(self):
        """Place car at centre; sync both env state and the Kivy widget."""
        if self.env is not None:
            self.env.reset(self.width / 2, self.height / 2)
        self.car.center = self.center

    def update(self, dt):
        global brain, last_reward, scores, goals_reached, longueur, largeur

        longueur = int(self.width)
        largeur  = int(self.height)
        if longueur < 100 or largeur < 100:
            return
        if first_update or sand.shape != (longueur, largeur):
            prev_obstacle_mode = self.env.obstacle_mode if self.env is not None else False
            init()
            self.env = Environment(longueur, largeur, sand)
            self.env.obstacle_mode = prev_obstacle_mode

        # ── Teleport flash decay ───────────────────────────────────────────
        if self._flash_alpha > 0:
            self._flash_alpha   = max(0.0, self._flash_alpha - 0.05)
            self._flash_color.a = self._flash_alpha

        # ── Animate goal marker ────────────────────────────────────────────
        gx, gy = self.env.goal_x, self.env.goal_y
        self._pulse  = (self._pulse + 4) % 360
        p            = abs(math.sin(math.radians(self._pulse)))
        gs           = 28 + p * 24
        self._g_glow.pos    = (gx - gs / 2, gy - gs / 2)
        self._g_glow.size   = (gs, gs)
        self._gc_glow.a     = 0.10 + p * 0.25
        self._g_ring.circle = (gx, gy, 10 + p * 5)
        self._g_dot.pos     = (gx - 5, gy - 5)
        self._g_h.points    = [gx - 20, gy,      gx + 20, gy    ]
        self._g_v.points    = [gx,      gy - 20, gx,      gy + 20]

        # ── AI step ────────────────────────────────────────────────────────
        state  = self.env.get_state(_get_map_input())
        action = brain.update(last_reward, state)
        scores.append(brain.score())

        needs_teleport = self.env.step(action2rotation[action])

        # ── Redraw canvas if obstacles were auto-generated ─────────────────
        if self.env.obstacles_dirty:
            self.env.obstacles_dirty = False
            if hasattr(self, 'painter') and self.painter is not None:
                self.painter.redraw_from_sand()

        # ── Sync Kivy car widget ───────────────────────────────────────────
        self.car.x          = self.env.x
        self.car.y          = self.env.y
        self.car.angle      = self.env.angle
        self.car.is_on_sand = float(self.env.is_on_sand)

        for ball, sx, sy in zip(
            (self.ball1, self.ball2, self.ball3, self.ball4, self.ball5),
            self.env.sensor_x, self.env.sensor_y,
        ):
            ball.pos = (sx, sy)

        # ── Sync globals from env (used by HUD) ───────────────────────────
        last_reward   = self.env.reward
        goals_reached = self.env.goals_reached

        # ── Sensor rays ────────────────────────────────────────────────────
        cx, cy  = self.car.center
        vis_a   = 0.8 if show_sensor_display else 0.0
        ball_op = 1.0 if show_sensor_display else 0.0
        for sc, sl, sx, sy, sig in zip(
            (self._sc1, self._sc2, self._sc3, self._sc4, self._sc5),
            (self._sl1, self._sl2, self._sl3, self._sl4, self._sl5),
            self.env.sensor_x, self.env.sensor_y, self.env.signals,
        ):
            sc.r = sig;  sc.g = 1.0 - sig * 0.85;  sc.b = 0.0;  sc.a = vis_a
            sl.points = [cx, cy, sx, sy]
        for ball in (self.ball1, self.ball2, self.ball3, self.ball4, self.ball5):
            ball.opacity = ball_op

        # ── Direction arrow ────────────────────────────────────────────────
        dist = self.env.distance
        if dist > 40:
            inv  = 1.0 / dist
            alen = min(50, dist * 0.25)
            self._arrow_color.a = min(0.55, dist / 350.0) * 0.75
            self._arrow_line.points = [
                cx, cy,
                cx + (gx - cx) * inv * alen,
                cy + (gy - cy) * inv * alen,
            ]
        else:
            self._arrow_color.a = 0.0

        # ── Teleport if stuck ──────────────────────────────────────────────
        if needs_teleport:
            self._teleport_car()

    def _teleport_car(self):
        x, y, _ = self.env.teleport()
        self.car.x         = x
        self.car.y         = y
        self._flash_alpha  = 0.35
        brain.random_steps = 300


# ── Paint widget ───────────────────────────────────────────────────────────────

class MyPaintWidget(Widget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shape_start    = None
        self._preview_group  = None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _sand_color(self):
        return (0.88, 0.52, 0.08, 1) if draw_mode else (0.10, 0.11, 0.18, 1)

    def _sand_val(self):
        return 1 if draw_mode else 0

    def _clear_preview(self):
        if self._preview_group is not None:
            self.canvas.remove(self._preview_group)
            self._preview_group = None

    def _draw_preview(self, sx, sy, ex, ey):
        self._clear_preview()
        grp = InstructionGroup()
        r, g, b, _ = self._sand_color()
        grp.add(Color(r, g, b, 0.45))
        if current_shape == 'rectangle':
            x = min(sx, ex);  y = min(sy, ey)
            w = abs(ex - sx); h = abs(ey - sy)
            grp.add(Line(rectangle=(x, y, w, h), width=1.5))
        elif current_shape == 'circle':
            cx = (sx + ex) / 2;  cy = (sy + ey) / 2
            rad = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2) / 2
            grp.add(Line(circle=(cx, cy, max(1, rad)), width=1.5))
        elif current_shape == 'line':
            grp.add(Line(points=[sx, sy, ex, ey], width=max(2, brush_size)))
        self.canvas.add(grp)
        self._preview_group = grp

    def _commit_shape(self, sx, sy, ex, ey):
        val   = self._sand_val()
        color = self._sand_color()
        sx, sy, ex, ey = int(sx), int(sy), int(ex), int(ey)

        with self.canvas:
            Color(*color)
            if current_shape == 'rectangle':
                x = min(sx, ex);  y = min(sy, ey)
                w = abs(ex - sx); h = abs(ey - sy)
                Rectangle(pos=(x, y), size=(w, h))
                x1 = max(0, x);     x2 = min(longueur, x + w)
                y1 = max(0, y);     y2 = min(largeur,  y + h)
                if x2 > x1 and y2 > y1:
                    sand[x1:x2, y1:y2] = val

            elif current_shape == 'circle':
                cx = (sx + ex) / 2;  cy = (sy + ey) / 2
                rad = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2) / 2
                Ellipse(pos=(cx - rad, cy - rad), size=(rad * 2, rad * 2))
                ri  = int(rad)
                cxi = int(cx);  cyi = int(cy)
                xi  = np.arange(max(0, cxi - ri), min(longueur, cxi + ri + 1))
                yi  = np.arange(max(0, cyi - ri), min(largeur,  cyi + ri + 1))
                if xi.size and yi.size:
                    xx, yy = np.meshgrid(xi, yi, indexing='ij')
                    mask   = (xx - cxi) ** 2 + (yy - cyi) ** 2 <= ri * ri
                    sand[xx[mask], yy[mask]] = val

            elif current_shape == 'line':
                Line(points=[sx, sy, ex, ey], width=max(2, brush_size))
                dist  = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
                steps = max(1, int(dist))
                bs    = brush_size
                for i in range(steps + 1):
                    t  = i / steps
                    px = int(sx + t * (ex - sx))
                    py = int(sy + t * (ey - sy))
                    x1 = max(0, px - bs);  x2 = min(longueur, px + bs)
                    y1 = max(0, py - bs);  y2 = min(largeur,  py + bs)
                    sand[x1:x2, y1:y2] = val

    def redraw_from_sand(self):
        """Rebuild the painter canvas entirely from the sand array (used by undo)."""
        self.canvas.clear()
        w, h = int(longueur), int(largeur)
        if w < 2 or h < 2:
            return
        tex  = Texture.create(size=(w, h), colorfmt='rgba')
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        # sand[x, y] — transpose to get row-major (y, x) layout expected by texture
        mask = sand[:w, :h].T > 0
        rgba[mask] = [224, 133, 20, 255]
        tex.blit_buffer(rgba.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        with self.canvas:
            Color(1, 1, 1, 1)
            Rectangle(texture=tex, pos=(0, 0), size=(w, h))

    # ── Events ─────────────────────────────────────────────────────────────────

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return
        global last_x, last_y
        _push_undo()
        last_x = int(touch.x);  last_y = int(touch.y)
        x, y   = last_x, last_y

        if current_shape == 'freehand':
            with self.canvas:
                if draw_mode:
                    Color(0.88, 0.52, 0.08, 1)
                    touch.ud['line'] = Line(points=(touch.x, touch.y),
                                           width=brush_size * 2)
                    sand[x - brush_size:x + brush_size,
                         y - brush_size:y + brush_size] = 1
                else:
                    Color(0.10, 0.11, 0.18, 1)
                    sz = brush_size * 2
                    Rectangle(pos=(x - sz, y - sz), size=(sz * 2, sz * 2))
                    sand[x - brush_size:x + brush_size,
                         y - brush_size:y + brush_size] = 0
        else:
            self._shape_start = (touch.x, touch.y)

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):
            return
        global last_x, last_y
        x, y = int(touch.x), int(touch.y)

        if current_shape == 'freehand':
            if draw_mode:
                if touch.button == 'left' and 'line' in touch.ud:
                    touch.ud['line'].points += [touch.x, touch.y]
                    touch.ud['line'].width   = brush_size * 2
                    sand[x - brush_size:x + brush_size,
                         y - brush_size:y + brush_size] = 1
            else:
                with self.canvas:
                    Color(0.10, 0.11, 0.18, 1)
                    sz = brush_size * 2
                    Rectangle(pos=(x - sz, y - sz), size=(sz * 2, sz * 2))
                sand[x - brush_size:x + brush_size,
                     y - brush_size:y + brush_size] = 0
        elif self._shape_start:
            self._draw_preview(*self._shape_start, touch.x, touch.y)

        last_x = x;  last_y = y

    def on_touch_up(self, touch):
        if not self.collide_point(*touch.pos):
            return
        if current_shape != 'freehand' and self._shape_start:
            self._clear_preview()
            self._commit_shape(*self._shape_start, touch.x, touch.y)
            self._shape_start = None


# ── App ────────────────────────────────────────────────────────────────────────

class CarApp(App):
    title = 'Self-Driving Car'

    def build(self):
        self._paused = False
        self._speed  = 1.0

        root = BoxLayout(orientation='horizontal')

        # ── Left: agent environment ────────────────────────────────────────
        game = Game()
        game.size_hint = (None, 1)
        game.width     = 860
        game.serve_car()
        self._game       = game
        self._game_event = Clock.schedule_interval(game.update, 1.0 / 60.0)

        self.painter = MyPaintWidget()
        self.painter.size_hint = (None, None)
        def _sync_painter(*_):
            self.painter.size = game.size
            self.painter.pos  = game.pos
        game.bind(size=_sync_painter, pos=_sync_painter)
        game.add_widget(self.painter)
        game.painter = self.painter   # give Game a reference for obstacle redraws

        # ── Right: control panel ───────────────────────────────────────────
        panel = ControlPanel()

        # Title
        panel.add_widget(Label(
            text='SELF-DRIVING CAR', font_size=15, bold=True,
            color=(0.28, 1, 0.55, 1), size_hint_y=None, height=28,
        ))
        panel.add_widget(Label(
            text='Deep Q-Network Agent', font_size=11,
            color=(0.50, 0.62, 0.50, 0.80), size_hint_y=None, height=16,
        ))
        panel.add_widget(_Divider())

        # ── Statistics ────────────────────────────────────────────────────
        panel.add_widget(_section_header('STATISTICS'))
        self.score_label  = _stat_label('Score:   0.000', (0.28, 1,    0.55, 1))
        self.reward_label = _stat_label('Reward:  0.000', (1,    0.80, 0.25, 1))
        self.steps_label  = _stat_label('Steps:   0',     (0.60, 0.60, 1,   1))
        self.goals_label  = _stat_label('Goals:   0',     (1,    0.60, 0.28, 1))
        for lbl in (self.score_label, self.reward_label,
                    self.steps_label, self.goals_label):
            panel.add_widget(lbl)

        # Mode row: canvas dot + text (fixes the ● font rendering issue)
        mode_row = BoxLayout(orientation='horizontal', size_hint_y=None,
                             height=22, spacing=6)
        self.status_dot  = StatusDot(color=(0.28, 1, 0.55, 1),
                                     pos_hint={'center_y': 0.5})
        self.mode_label  = _stat_label('LEARNING', (0.28, 1, 0.55, 1))
        mode_row.add_widget(self.status_dot)
        mode_row.add_widget(self.mode_label)
        panel.add_widget(mode_row)
        panel.add_widget(_Divider())

        # ── Q-values ──────────────────────────────────────────────────────
        panel.add_widget(_section_header('Q-VALUES  (Straight · Left · Right)'))
        self.qvalue_widget = QValueWidget(size_hint=(1, None), height=60)
        panel.add_widget(self.qvalue_widget)
        panel.add_widget(_Divider())

        # ── Simulation ────────────────────────────────────────────────────
        panel.add_widget(_section_header('SIMULATION'))
        self.pause_btn = Button(
            text='Pause', size_hint_y=None, height=34,
            background_color=(0.42, 0.22, 0.18, 1),
        )
        self.pause_btn.bind(on_release=lambda _: self._toggle_pause())
        panel.add_widget(self.pause_btn)

        speed_row = BoxLayout(orientation='horizontal', size_hint_y=None,
                              height=28, spacing=6)
        speed_lbl = Label(text='Speed', size_hint_x=None, width=44,
                          font_size=12, bold=True, color=(0.75, 0.75, 0.75, 1))
        self.speed_slider = Slider(min=0.5, max=4.0, value=1.0, step=0.5)
        self.speed_label  = Label(text='1.0x', size_hint_x=None, width=38,
                                  font_size=12, bold=True, color=(1, 0.88, 0.50, 1))
        self.speed_slider.bind(value=self._set_speed)
        speed_row.add_widget(speed_lbl)
        speed_row.add_widget(self.speed_slider)
        speed_row.add_widget(self.speed_label)
        panel.add_widget(speed_row)
        panel.add_widget(_Divider())

        # ── Brain ─────────────────────────────────────────────────────────
        panel.add_widget(_section_header('BRAIN'))
        brain_row = BoxLayout(orientation='horizontal', size_hint_y=None,
                              height=34, spacing=6)
        savebtn  = Button(text='Save',  background_color=(0.18, 0.42, 0.22, 1))
        loadbtn  = Button(text='Load',  background_color=(0.42, 0.18, 0.18, 1))
        resetbtn = Button(text='Reset', background_color=(0.45, 0.32, 0.10, 1))
        savebtn .bind(on_release=self.save)
        loadbtn .bind(on_release=self.load)
        resetbtn.bind(on_release=lambda _: self._reset_brain())
        for b in (savebtn, loadbtn, resetbtn):
            brain_row.add_widget(b)
        panel.add_widget(brain_row)
        map_row = BoxLayout(orientation='horizontal', size_hint_y=None,
                            height=34, spacing=6)
        savemapbtn = Button(text='Save Map', background_color=(0.18, 0.32, 0.42, 1))
        loadmapbtn = Button(text='Load Map', background_color=(0.32, 0.18, 0.42, 1))
        savemapbtn.bind(on_release=lambda _: self._save_map())
        loadmapbtn.bind(on_release=lambda _: self._load_map())
        map_row.add_widget(savemapbtn)
        map_row.add_widget(loadmapbtn)
        panel.add_widget(map_row)
        panel.add_widget(_Divider())

        # ── Drawing tools ─────────────────────────────────────────────────
        panel.add_widget(_section_header('DRAWING TOOLS'))
        draw_row = BoxLayout(orientation='horizontal', size_hint_y=None,
                             height=34, spacing=6)
        self.draw_btn  = Button(text='Draw',  background_color=(0.18, 0.48, 0.22, 1))
        self.erase_btn = Button(text='Erase', background_color=(0.22, 0.22, 0.32, 1))
        clearbtn       = Button(text='Clear', background_color=(0.22, 0.22, 0.42, 1))
        self.undo_btn  = Button(text='Undo',  background_color=(0.38, 0.28, 0.12, 1))
        self.draw_btn .bind(on_release=lambda _: self._set_draw_mode(True))
        self.erase_btn.bind(on_release=lambda _: self._set_draw_mode(False))
        clearbtn      .bind(on_release=self.clear_canvas)
        self.undo_btn .bind(on_release=lambda _: self._undo())
        for b in (self.draw_btn, self.erase_btn, clearbtn, self.undo_btn):
            draw_row.add_widget(b)
        panel.add_widget(draw_row)

        # Shape selector
        panel.add_widget(_section_header('SHAPE'))
        shape_row1 = BoxLayout(orientation='horizontal', size_hint_y=None,
                               height=34, spacing=6)
        shape_row2 = BoxLayout(orientation='horizontal', size_hint_y=None,
                               height=34, spacing=6)
        _active_shape   = (0.18, 0.48, 0.55, 1)
        _inactive_shape = (0.22, 0.22, 0.32, 1)
        self.shape_btns = {}
        for name, label, row in [
            ('freehand',  'Freehand',  shape_row1),
            ('line',      'Line',      shape_row1),
            ('rectangle', 'Rectangle', shape_row2),
            ('circle',    'Circle',    shape_row2),
        ]:
            btn = Button(
                text=label,
                background_color=_active_shape if name == 'freehand' else _inactive_shape,
            )
            btn.bind(on_release=lambda _, n=name: self._set_shape(n))
            self.shape_btns[name] = btn
            row.add_widget(btn)
        panel.add_widget(shape_row1)
        panel.add_widget(shape_row2)

        brush_row = BoxLayout(orientation='horizontal', size_hint_y=None,
                              height=28, spacing=6)
        brush_hdr  = Label(text='Brush', size_hint_x=None, width=44,
                           font_size=12, bold=True, color=(0.75, 0.75, 0.75, 1))
        self.brush_slider = Slider(min=5, max=60, value=brush_size, step=1)
        self.brush_label  = Label(text=f'{brush_size}px', size_hint_x=None, width=38,
                                  font_size=12, bold=True, color=(1, 0.88, 0.50, 1))
        self.brush_slider.bind(value=self._on_brush_size)
        brush_row.add_widget(brush_hdr)
        brush_row.add_widget(self.brush_slider)
        brush_row.add_widget(self.brush_label)
        panel.add_widget(brush_row)
        panel.add_widget(_Divider())

        # ── View ──────────────────────────────────────────────────────────
        panel.add_widget(_section_header('VIEW'))
        self.sensor_btn = Button(
            text='Hide Sensors', size_hint_y=None, height=34,
            background_color=(0.18, 0.42, 0.22, 1),
        )
        self.sensor_btn.bind(on_release=lambda _: self._toggle_sensors())
        panel.add_widget(self.sensor_btn)

        self.obstacle_btn = Button(
            text='Obstacle Mode: OFF', size_hint_y=None, height=34,
            background_color=(0.22, 0.22, 0.32, 1),
        )
        self.obstacle_btn.bind(on_release=lambda _: self._toggle_obstacle_mode())
        panel.add_widget(self.obstacle_btn)
        panel.add_widget(_Divider())

        # ── Score graph ───────────────────────────────────────────────────
        panel.add_widget(_section_header('SCORE HISTORY'))
        self.graph = ScoreGraph(size_hint=(1, 1))
        panel.add_widget(self.graph)

        root.add_widget(game)
        root.add_widget(panel)

        Clock.schedule_interval(self.update_hud, 0.1)
        return root

    # ── HUD update ─────────────────────────────────────────────────────────────

    def update_hud(self, dt):
        self.score_label.text  = f'Score:   {brain.score():.3f}'
        self.reward_label.text = f'Reward:  {last_reward:.3f}'
        self.steps_label.text  = f'Steps:   {brain.steps}'
        self.goals_label.text  = f'Goals:   {goals_reached}'

        if self._paused:
            color = (0.55, 0.55, 0.55, 1)
            self.status_dot.set_color(color)
            self.mode_label.text  = 'PAUSED'
            self.mode_label.color = color
        elif brain.random_steps > 0:
            color = (1, 0.90, 0.15, 1)
            self.status_dot.set_color(color)
            self.mode_label.text  = f'RANDOM ({brain.random_steps})'
            self.mode_label.color = color
        else:
            color = (0.28, 1, 0.55, 1)
            self.status_dot.set_color(color)
            self.mode_label.text  = 'LEARNING'
            self.mode_label.color = color

        self.graph.push(brain.score())
        self.qvalue_widget.update(brain.get_q_values())

    # ── Controls ───────────────────────────────────────────────────────────────

    def _toggle_pause(self):
        if self._paused:
            self._game_event = Clock.schedule_interval(
                self._game.update, 1.0 / (60.0 * self._speed)
            )
            self._paused = False
            self.pause_btn.text             = 'Pause'
            self.pause_btn.background_color = (0.42, 0.22, 0.18, 1)
        else:
            self._game_event.cancel()
            self._paused = True
            self.pause_btn.text             = 'Resume'
            self.pause_btn.background_color = (0.18, 0.42, 0.22, 1)

    def _set_speed(self, slider, value):
        self._speed = value
        self.speed_label.text = f'{value:.1f}x'
        if not self._paused:
            self._game_event.cancel()
            self._game_event = Clock.schedule_interval(
                self._game.update, 1.0 / (60.0 * value)
            )

    def _reset_brain(self):
        global brain, scores, last_reward, goals_reached
        brain         = Dqn(12 + MAP_GRID * MAP_GRID, 3, 0.99)
        scores        = []
        last_reward   = 0
        goals_reached = 0
        if self._game.env is not None:
            self._game.env.goals_reached  = 0
            self._game.env._steps_to_goal = 0
            self._game.env._stuck_steps   = 0
        self._game.serve_car()
        self.graph.clear()

    def _undo(self):
        global sand
        if not undo_stack:
            return
        sand[:] = undo_stack.pop()
        self.painter.redraw_from_sand()

    def _save_map(self):
        np.save('last_map.npy', sand)
        print("Map saved to last_map.npy")

    def _load_map(self):
        global sand
        if os.path.isfile('last_map.npy'):
            loaded = np.load('last_map.npy')
            if loaded.shape == sand.shape:
                _push_undo()
                sand[:] = loaded
                self.painter.redraw_from_sand()
                print("Map loaded.")
            else:
                print(f"Map size mismatch: saved {loaded.shape}, current {sand.shape}")
        else:
            print("No saved map found (last_map.npy).")

    def _set_shape(self, name):
        global current_shape
        current_shape = name
        _active   = (0.18, 0.48, 0.55, 1)
        _inactive = (0.22, 0.22, 0.32, 1)
        for n, btn in self.shape_btns.items():
            btn.background_color = _active if n == name else _inactive

    def _set_draw_mode(self, is_draw):
        global draw_mode
        draw_mode = is_draw
        self.draw_btn.background_color  = (0.18, 0.48, 0.22, 1) if is_draw  else (0.22, 0.22, 0.32, 1)
        self.erase_btn.background_color = (0.52, 0.22, 0.18, 1) if not is_draw else (0.22, 0.22, 0.32, 1)

    def _toggle_obstacle_mode(self):
        env = self._game.env
        if env is None:
            return
        env.obstacle_mode = not env.obstacle_mode
        if env.obstacle_mode:
            self.obstacle_btn.text             = 'Obstacle Mode: ON'
            self.obstacle_btn.background_color = (0.18, 0.48, 0.22, 1)
        else:
            # Clear auto-obstacles when mode is turned off
            env.clear_auto_obstacles()
            self.painter.redraw_from_sand()
            self.obstacle_btn.text             = 'Obstacle Mode: OFF'
            self.obstacle_btn.background_color = (0.22, 0.22, 0.32, 1)

    def _toggle_sensors(self):
        global show_sensor_display
        show_sensor_display = not show_sensor_display
        if show_sensor_display:
            self.sensor_btn.text             = 'Hide Sensors'
            self.sensor_btn.background_color = (0.18, 0.42, 0.22, 1)
        else:
            self.sensor_btn.text             = 'Show Sensors'
            self.sensor_btn.background_color = (0.32, 0.32, 0.32, 1)

    def _on_brush_size(self, slider, value):
        global brush_size
        brush_size = int(value)
        self.brush_label.text = f'{brush_size}px'

    def clear_canvas(self, obj):
        _push_undo()
        self.painter.canvas.clear()
        sand.fill(0)   # in-place: keeps env.sand reference valid

    def save(self, obj):
        print("Saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("Loading last saved brain...")
        brain.load()


if __name__ == '__main__':
    CarApp().run()
