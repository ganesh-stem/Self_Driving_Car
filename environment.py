# Pure Python environment — no Kivy, fully testable independently.

import math
import numpy as np

SENSOR_DIST   = 30          # px from car centre to each sensor tip
SENSOR_ANGLES = [0, 30, -30, 60, -60]   # degrees relative to heading
SENSOR_AREA   = 10          # half-side of reading square (20×20 = 400 cells)
SPEED_FAST    = 6.0         # px/step on road
SPEED_SLOW    = 1.0         # px/step on sand
STUCK_LIMIT    = 480         # consecutive penalty steps before teleport (~8 s at 60 fps)
SPEED_REF      = 1500        # steps at which speed bonus becomes 0
CIRCLING_LIMIT = 480         # steps without closing distance before teleport (~8 s at 60 fps)
PROGRESS_MIN   = 20.0        # px improvement needed to reset circling counter
SPIN_LIMIT     = 25          # consecutive same-direction turns before spin penalty
SPIN_PENALTY   = 0.3         # reward subtracted while spinning
OBSTACLE_COUNT    = 5        # auto-generated rectangles per update
OBSTACLE_MIN_SIZE = 40       # px — minimum rectangle side
OBSTACLE_MAX_SIZE = 100      # px — maximum rectangle side
OBSTACLE_GOAL_GAP = 150      # px — exclusion radius around each goal position


class Environment:
    """
    Owns all game logic: car physics, sensor reading, reward shaping,
    goal management, stuck detection, and teleportation.

    Kivy widgets read from this object each frame but never write back.
    """

    def __init__(self, width, height, sand):
        self.width  = int(width)
        self.height = int(height)
        self.sand   = sand          # reference to the shared numpy array

        # ── Car state ──────────────────────────────────────────────────────
        self.x     = self.width  / 2.0
        self.y     = self.height / 2.0
        self.angle = 0.0
        self.speed = SPEED_FAST

        # ── Sensor output (5 sensors) ──────────────────────────────────────
        self.sensor_x = [0.0] * 5
        self.sensor_y = [0.0] * 5
        self.signals  = [0.0] * 5

        # ── Goal ───────────────────────────────────────────────────────────
        self.goal_x = 20.0
        self.goal_y = self.height - 20.0

        # ── Observable outputs ─────────────────────────────────────────────
        self.is_on_sand  = False
        self.reward      = 0.0
        self.distance    = 0.0
        self.orientation = 0.0
        self.norm_dist   = 0.0

        # ── Internal counters ──────────────────────────────────────────────
        self.goals_reached    = 0
        self._steps_to_goal   = 0
        self._stuck_steps     = 0
        self._best_dist         = float('inf')
        self._no_progress_steps = 0
        self._last_rotation     = 0.0
        self._spin_steps        = 0

        # ── Obstacle mode ──────────────────────────────────────────────────
        self.obstacle_mode   = False
        self.obstacles_dirty = False   # set True whenever sand is auto-modified
        self._auto_obstacles = []      # list of (x, y, w, h) auto-placed rects

        self._update_sensors()
        self._update_nav()

    # ── Public API ──────────────────────────────────────────────────────────────

    def reset(self, x=None, y=None, angle=0.0):
        """Place the car at (x, y) with heading angle (degrees)."""
        self.x     = float(x) if x is not None else self.width  / 2.0
        self.y     = float(y) if y is not None else self.height / 2.0
        self.angle = float(angle)
        self.speed = SPEED_FAST
        self._best_dist         = float('inf')
        self._no_progress_steps = 0
        self._last_rotation     = 0.0
        self._spin_steps        = 0
        self._update_sensors()
        self._update_nav()

    def step(self, rotation):
        """
        Rotate by `rotation` degrees, advance one step, read sensors,
        compute reward.

        Returns True when the car is stuck and needs to be teleported.
        """
        prev_dist = self.distance

        # ── Spin tracking ──────────────────────────────────────────────────
        if rotation != 0 and self._last_rotation != 0 and \
                (rotation > 0) == (self._last_rotation > 0):
            self._spin_steps += 1
        else:
            self._spin_steps = 0
        self._last_rotation = rotation

        # ── Physics ────────────────────────────────────────────────────────
        self.angle += rotation
        self.x     += self.speed * math.cos(math.radians(self.angle))
        self.y     += self.speed * math.sin(math.radians(self.angle))

        # ── Sensors & navigation ───────────────────────────────────────────
        self._update_sensors()
        self._update_nav()

        # ── Reward ─────────────────────────────────────────────────────────
        return self._compute_reward(prev_dist)

    def teleport(self):
        """
        Find a clear spot far from the goal and place the car there.
        Returns (x, y, angle) of the new position.
        """
        for _ in range(500):
            x = int(np.random.uniform(60, self.width  - 60))
            y = int(np.random.uniform(60, self.height - 60))
            dist_goal = math.sqrt((x - self.goal_x) ** 2 + (y - self.goal_y) ** 2)
            if self.sand[x, y] == 0 and dist_goal > 200:
                angle = float(np.random.uniform(0, 360))
                self.reset(x, y, angle)
                return x, y, angle
        # Fallback: centre
        self.reset()
        return self.width / 2.0, self.height / 2.0, 0.0

    def get_state(self, map_input):
        """
        Build the flat state vector fed to the neural network.
        `map_input` is the pre-computed 256-value downsampled sand map.
        """
        return [
            *self.signals,                          # 5  — local sensor readings
            self.orientation, -self.orientation,    # 2  — signed angle to goal + its negation
            self.norm_dist,                         # 1  — normalised distance to goal
            self.goal_x / self.width,               # 1  — goal x (normalised)
            self.goal_y / self.height,              # 1  — goal y (normalised)
            self.x / self.width,                    # 1  — car x (anchors map input)
            self.y / self.height,                   # 1  — car y
            *map_input,                             # 256 — downsampled obstacle map
        ]                                           # total: 268

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _update_sensors(self):
        """Compute positions and sand-density readings for all 5 sensors."""
        for i, a_offset in enumerate(SENSOR_ANGLES):
            a  = self.angle + a_offset
            sx = self.x + SENSOR_DIST * math.cos(math.radians(a))
            sy = self.y + SENSOR_DIST * math.sin(math.radians(a))
            self.sensor_x[i] = sx
            self.sensor_y[i] = sy
            self.signals[i]  = self._read_sensor(sx, sy)

    def _read_sensor(self, sx, sy):
        """Return sand density [0, 1] at the given sensor position."""
        x, y = int(sx), int(sy)
        if x > self.width - 10 or x < 10 or y > self.height - 10 or y < 10:
            return 1.0
        region = self.sand[x - SENSOR_AREA : x + SENSOR_AREA,
                           y - SENSOR_AREA : y + SENSOR_AREA]
        return float(np.sum(region)) / (4.0 * SENSOR_AREA ** 2)

    def _update_nav(self):
        """Recompute distance, norm_dist, and orientation to goal."""
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        d  = math.sqrt(dx * dx + dy * dy)
        self.distance  = d
        self.norm_dist = min(1.0, d / math.sqrt(self.width ** 2 + self.height ** 2))

        # Orientation: signed angle in [-1, 1] between heading and goal direction.
        # Positive = goal is to the left, negative = goal is to the right.
        # atan2(cross, dot) gives the full signed angle so the agent knows
        # which way to turn instead of only knowing the magnitude.
        if d > 1e-8:
            vx    = math.cos(math.radians(self.angle))
            vy    = math.sin(math.radians(self.angle))
            dot   = max(-1.0, min(1.0, vx * dx / d + vy * dy / d))
            cross = vx * dy / d - vy * dx / d   # z-component of heading × goal
            self.orientation = math.degrees(math.atan2(cross, dot)) / 180.0
        else:
            self.orientation = 0.0

    def _compute_reward(self, prev_dist):
        """
        Set self.reward based on current state.
        Returns True if the car has been stuck long enough to need a teleport.
        """
        # ── Sand / speed ───────────────────────────────────────────────────
        ix, iy = int(self.x), int(self.y)
        in_bounds = 0 <= ix < self.width and 0 <= iy < self.height
        self.is_on_sand = in_bounds and self.sand[ix, iy] > 0

        if self.is_on_sand:
            self.speed  = SPEED_SLOW
            self.reward = -1.0
        else:
            self.speed  = SPEED_FAST
            delta       = prev_dist - self.distance
            self.reward = -0.1 + float(np.clip(delta * 0.1, -0.5, 0.5))

        # ── Boundary clamp ─────────────────────────────────────────────────
        if self.x < 10:
            self.x = 10.0;                    self.reward = -1.0
        if self.x > self.width - 10:
            self.x = float(self.width  - 10); self.reward = -1.0
        if self.y < 10:
            self.y = 10.0;                    self.reward = -1.0
        if self.y > self.height - 10:
            self.y = float(self.height - 10); self.reward = -1.0

        # ── Goal ───────────────────────────────────────────────────────────
        self._steps_to_goal += 1
        if self.distance < 100:
            self.goals_reached      += 1
            speed_bonus              = max(0.0, 1.0 - self._steps_to_goal / SPEED_REF)
            self.reward              = 1.0 + speed_bonus
            self._steps_to_goal      = 0
            self._best_dist          = float('inf')
            self._no_progress_steps  = 0
            self.goal_x              = self.width  - self.goal_x
            self.goal_y              = self.height - self.goal_y
            if self.obstacle_mode and self.goals_reached % 3 == 0:
                self._generate_obstacles()

        # ── Spin penalty ───────────────────────────────────────────────────
        if self._spin_steps >= SPIN_LIMIT:
            self.reward -= SPIN_PENALTY

        # ── Stuck detection ────────────────────────────────────────────────
        if self.reward == -1.0:
            self._stuck_steps += 1
            if self._stuck_steps >= STUCK_LIMIT:
                self._stuck_steps = 0
                return True       # caller should teleport
        else:
            self._stuck_steps = 0

        # ── Circling detection ─────────────────────────────────────────────
        if self.distance < self._best_dist - PROGRESS_MIN:
            self._best_dist         = self.distance
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1
            if self._no_progress_steps >= CIRCLING_LIMIT:
                self._no_progress_steps = 0
                self._best_dist         = float('inf')
                return True       # caller should teleport

        return False

    def _generate_obstacles(self):
        """
        Clear old auto-obstacles, then place OBSTACLE_COUNT new random rectangles.
        Excludes a OBSTACLE_GOAL_GAP px radius around both goal positions and the car.
        Sets obstacles_dirty so the caller can redraw the canvas.
        """
        # Erase previous auto-obstacles from sand
        for (ox, oy, ow, oh) in self._auto_obstacles:
            self.sand[ox:ox + ow, oy:oy + oh] = 0
        self._auto_obstacles = []

        # Both alternating goal positions (they are always mirror images)
        g1x, g1y = self.goal_x, self.goal_y
        g2x, g2y = self.width - g1x, self.height - g1y

        placed = 0
        for _ in range(500):
            if placed >= OBSTACLE_COUNT:
                break
            w = int(np.random.randint(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE))
            h = int(np.random.randint(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE))
            x = int(np.random.randint(20, self.width  - w - 20))
            y = int(np.random.randint(20, self.height - h - 20))
            cx = x + w / 2.0
            cy = y + h / 2.0
            # Skip if centre is too close to either goal
            if math.sqrt((cx - g1x) ** 2 + (cy - g1y) ** 2) < OBSTACLE_GOAL_GAP:
                continue
            if math.sqrt((cx - g2x) ** 2 + (cy - g2y) ** 2) < OBSTACLE_GOAL_GAP:
                continue
            # Skip if centre is too close to the car
            if math.sqrt((cx - self.x) ** 2 + (cy - self.y) ** 2) < 80:
                continue
            self.sand[x:x + w, y:y + h] = 1
            self._auto_obstacles.append((x, y, w, h))
            placed += 1

        self.obstacles_dirty = True

    def clear_auto_obstacles(self):
        """Remove all auto-generated obstacles from the sand map."""
        for (ox, oy, ow, oh) in self._auto_obstacles:
            self.sand[ox:ox + ow, oy:oy + oh] = 0
        self._auto_obstacles = []
        self.obstacles_dirty = True
