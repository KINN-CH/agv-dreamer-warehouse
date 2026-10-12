"""AGV 내비게이션 — 격자 지도 위 A* 경로 + pure pursuit 경로 추종 + 센서 대응, 월드모델 입력(BEV). 순수 Python.

    python nav.py      # demo(): 작은 격자에서 경로 계획·추종·재계획·BEV 자체 검사
"""
import heapq
import math

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label


class GridMap:
    """occ[y, x] = 점유. (x0, y0) = 칸 [0, 0]의 모서리, cell = 칸 크기(m)."""

    def __init__(self, occ: np.ndarray, cell: float, x0: float, y0: float):
        self.occ, self.cell, self.x0, self.y0 = occ.astype(bool), cell, x0, y0
        self.dist = distance_transform_edt(~self.occ) * cell        # 칸마다 가장 가까운 장애물까지 거리(m)

    def ij(self, x, y):
        return int((y - self.y0) / self.cell), int((x - self.x0) / self.cell)

    def xy(self, i, j):
        return self.x0 + (j + 0.5) * self.cell, self.y0 + (i + 0.5) * self.cell

    def clearance(self, x, y) -> float:
        i, j = self.ij(x, y)
        if not (0 <= i < self.occ.shape[0] and 0 <= j < self.occ.shape[1]):
            return 0.0
        return float(self.dist[i, j])

    def coarse(self, cell: float) -> "GridMap":
        """계획용 저해상도 지도 — 칸 안에 점유가 하나라도 있으면 점유."""
        k = max(1, round(cell / self.cell))
        h, w = self.occ.shape[0] // k * k, self.occ.shape[1] // k * k
        occ = self.occ[:h, :w].reshape(h // k, k, w // k, k).any(axis=(1, 3))
        return GridMap(occ, self.cell * k, self.x0, self.y0)


def plan(m: GridMap, start, goal, robot_r=0.5, soft_r=1.2, soft_w=4.0):
    """A* (8방향). 장애물에서 robot_r 안은 금지, soft_r 안은 비용을 더해 통로 가운데로 다니게 한다.
    반환: 월드 좌표 경유점 [(x, y), ...] (시작·끝 포함). 길이 없으면 None."""
    free = m.dist >= robot_r
    extra = soft_w * np.clip((soft_r - m.dist) / soft_r, 0, None)
    s, g = m.ij(*start), m.ij(*goal)
    if not all(0 <= q[0] < free.shape[0] and 0 <= q[1] < free.shape[1] for q in (s, g)):
        return None                                                 # 지도 밖
    if not free[s] or not free[g]:
        return None
    h = lambda a: math.hypot(a[0] - g[0], a[1] - g[1])          # noqa: E731
    best, came, pq = {s: 0.0}, {}, [(h(s), 0.0, s)]
    steps = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    while pq:
        _, cost, cur = heapq.heappop(pq)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            pts = [m.xy(*c) for c in path[::-1]]
            pts[0], pts[-1] = tuple(start), tuple(goal)                 # 칸 중심이 아니라 요청한 점에서 시작·끝
            return shortcut(m, pts, robot_r)
        if cost > best[cur]:
            continue
        for di, dj, d in steps:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] < free.shape[0] and 0 <= nb[1] < free.shape[1]) or not free[nb]:
                continue
            nc = cost + d * (1.0 + extra[nb])
            if nc < best.get(nb, math.inf):
                best[nb], came[nb] = nc, cur
                heapq.heappush(pq, (nc + h(nb), nc, nb))
    return None


def shortcut(m: GridMap, pts, robot_r):
    """격자 계단을 펴는 가시선 단축 — 두 점 사이 직선이 안전하면 중간 점을 뺀다."""
    def safe(a, b):
        n = max(2, int(math.dist(a, b) / (m.cell / 2)))
        return all(m.clearance(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) >= robot_r for t in np.linspace(0, 1, n))
    out, k = [pts[0]], 0
    while k < len(pts) - 1:
        j = len(pts) - 1
        while j > k + 1 and not safe(pts[k], pts[j]):
            j -= 1
        out.append(pts[j])
        k = j
    return out


def densify(pts, step=0.1):
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(math.dist(a, b) / step))
        out += [(a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n) for t in range(1, n + 1)]
    return np.array(out)


class PurePursuit:
    """경로 위 lookahead 앞 점을 향해 곡률로 회전. 방향이 크게 틀리면 제자리 회전부터."""

    def __init__(self, path, lookahead=0.8, v_max=1.0, w_max=1.2, goal_tol=0.3):
        self.path, self.L, self.v_max, self.w_max, self.tol = densify(path), lookahead, v_max, w_max, goal_tol
        self.k = 0                                                  # 지나온 경로 위치 (뒤로 안 감)

    def step(self, x, y, yaw):
        """(선속도 m/s, 각속도 rad/s, 도착 여부)."""
        p = self.path
        win = p[self.k:self.k + 50]
        self.k += int(np.argmin(np.hypot(win[:, 0] - x, win[:, 1] - y)))
        to_goal = math.hypot(p[-1, 0] - x, p[-1, 1] - y)
        if to_goal < self.tol:
            return 0.0, 0.0, True
        ahead = np.hypot(p[self.k:, 0] - x, p[self.k:, 1] - y)
        far = np.nonzero(ahead >= self.L)[0]
        tx, ty = p[self.k + far[0]] if len(far) else p[-1]
        dx, dy = tx - x, ty - y
        xr, yr = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        alpha = math.atan2(yr, xr)
        if abs(alpha) > 0.8:                                        # 먼저 방향을 돌린다
            return 0.0, math.copysign(0.6 * self.w_max, alpha), False
        curv = 2 * yr / max(xr * xr + yr * yr, 1e-6)
        v = min(self.v_max / (1 + 2 * abs(curv)), max(0.2, to_goal))
        return v, float(np.clip(v * curv, -self.w_max, self.w_max)), False

    def heading_err(self, x, y, yaw):
        """lookahead 목표점까지 방향 오차 (rad, 왼쪽 +) — 제자리 회전량."""
        p = self.path
        ahead = np.hypot(p[self.k:, 0] - x, p[self.k:, 1] - y)
        far = np.nonzero(ahead >= self.L)[0]
        tx, ty = p[self.k + far[0]] if len(far) else p[-1]
        dx, dy = tx - x, ty - y
        return math.atan2(-math.sin(yaw) * dx + math.cos(yaw) * dy, math.cos(yaw) * dx + math.sin(yaw) * dy)

    def reverse_cmd(self, x, y, yaw, speed=0.3):
        """경로 목표점이 뒤쪽(뒤 ±0.8rad 안)이면 경로를 뒤로 따라가는 (선속도<0, 각속도) — 좁아서 못 도는 곳의 U턴 대신. 아니면 None.
        뒤 방향 φ = yaw + π를 앞으로 보고 같은 순수 추종: 곡률 κ, 각속도 = 속도 × κ (φ' = yaw')."""
        p = self.path
        ahead = np.hypot(p[self.k:, 0] - x, p[self.k:, 1] - y)
        far = np.nonzero(ahead >= self.L)[0]
        tx, ty = p[self.k + far[0]] if len(far) else p[-1]
        dx, dy = tx - x, ty - y
        xr, yr = -(math.cos(yaw) * dx + math.sin(yaw) * dy), -(-math.sin(yaw) * dx + math.cos(yaw) * dy)
        if abs(math.atan2(yr, xr)) > 0.8:
            return None
        curv = 2 * yr / max(xr * xr + yr * yr, 1e-6)
        return -speed, float(np.clip(speed * curv, -self.w_max, self.w_max))

    def cross_track(self, x, y) -> float:
        return float(np.min(np.hypot(self.path[:, 0] - x, self.path[:, 1] - y)))


class Navigator:
    """정적 지도로 계획한 경로를 따라가며 앞 센서로 대응한다.
    - 같은 칸을 confirm번 연속 장애물로 보면 지도에 확정, 확정 칸이 남은 경로를 막으면 다시 계획 (상자·서 있는 사람)
      → 지나가는 사람·지게차는 칸마다 잠깐만 보이므로 확정되지 않고, 멈춰 기다린다
    - 다시 바닥이 보이는 칸은 지운다. 지도가 바뀌었으면 replan_every번 관측마다 다시 계획 (지름길로 복귀)
    - 확정 안 된(움직이는·처음 보는) 장애물이 차체 앞 stop m·옆 stop_side m 안이면 정지, slow m·옆 slow_side m 안이면 감속
      (가로지르는 사람·지게차). 확정된 정지 장애물은 경로가 피해 가므로 정지 규칙에서 뺀다 (옆을 지날 때 서 버리지 않게)
    step에 센서 점을 한 번도 안 주면 정적 지도만 믿고 가는 '대응 없음'과 같다."""

    def __init__(self, static: GridMap, start, goal, robot_r=0.7, plan_cell=0.1, front=0.4, half_w=0.33,
                 stop=1.5, slow=3.0, stop_side=0.8, slow_side=1.2, v_slow=0.35, confirm=15, replan_every=20,
                 backout=False, back=-1.033, sweep=1.15, safe_reverse=False, swept=False, **pp_kw):
        self.static, self.goal, self.r0, self.pp_kw = static, tuple(goal), robot_r, pp_kw
        self.backout, self.back, self.sweep = backout, back, sweep     # 좁은 곳 제자리 회전 대신 똑바로 후진 (차체 뒤 1.03m가 랙을 쓸지 않게)
        self.rear_block, self.backed, self.crept = False, 0.0, 0.0
        self.smap = None                                            # 회전 공간 판단용 지도 (backout)
        self.last_obs = None                                        # 마지막으로 본 장애물 점 (돌며 갈 때 쓸림 확인)
        self.safe_reverse = safe_reverse                            # 앞보기가 고른 후진도 뒤 1m·랙을 확인
        self.swept = swept                                          # 회전 가능 = 차체가 실제로 쓸고 지나갈 자리가 비었나 (원 대신)
        us = np.linspace(back, front, 12)
        self.body_pts = [(u, s * half_w) for u in us for s in (-1, 1)] + [(u, v) for u in (back, front) for v in np.linspace(-half_w, half_w, 5)]
        self.front, self.half_w, self.stop, self.slow, self.v_slow = front, half_w, stop, slow, v_slow
        self.stop_side, self.slow_side = stop_side, slow_side
        self.confirm, self.replan_every = confirm, replan_every
        self.base = static.coarse(plan_cell)
        self.hits = np.zeros(self.base.occ.shape, np.int16)        # 칸마다 연속으로 장애물로 본 횟수 (계획 해상도)
        self.seen = np.zeros_like(self.base.occ)                    # 확정된 장애물 칸
        self.cmap, self.limit, self.dirty, self.since = self.base, None, False, 0
        self.near_seen = self.seen
        self.cap = None                                             # 앞보기(B′·월드모델)가 정한 속도 (음수면 후진)
        self.replans, self.pp, self.r = -1, None, robot_r
        self.replan(*start)

    def _cells(self, xy, margin=0.0):
        """점 → 계획 지도 칸. margin > 0이면 정적 장애물 margin m 안 점은 뺀다 (이미 지도에 있음)."""
        m, xy = self.base, np.asarray(xy, float).reshape(-1, 2)
        i, j = ((xy[:, 1] - m.y0) / m.cell).astype(int), ((xy[:, 0] - m.x0) / m.cell).astype(int)
        ok = (i >= 0) & (i < m.occ.shape[0]) & (j >= 0) & (j < m.occ.shape[1])
        if margin > 0:
            si = ((xy[:, 1] - self.static.y0) / self.static.cell).astype(int)
            sj = ((xy[:, 0] - self.static.x0) / self.static.cell).astype(int)
            ok &= (si >= 0) & (si < self.static.occ.shape[0]) & (sj >= 0) & (sj < self.static.occ.shape[1])
            ok[ok] = self.static.dist[si[ok], sj[ok]] >= margin
        return i[ok], j[ok]

    def observe(self, obs_xy, free_xy=(), static_margin=0.15) -> bool:
        """장애물 점·바닥 점(월드 x, y). 장애물 칸은 +1(확정 + 10까지), 바닥만 보인 칸은 −5. 새로 확정된 칸이 생기면 True."""
        before = self.seen
        ij = np.unique(np.c_[self._cells(obs_xy, static_margin)], axis=0) if len(obs_xy) else np.zeros((0, 2), int)
        if len(free_xy):                                            # 장애물 점도 있는 칸(다리 사이로 바닥이 보이는 등)은 안 지운다
            keep = self.hits[ij[:, 0], ij[:, 1]].copy()
            fi, fj = self._cells(free_xy)
            self.hits[fi, fj] = np.maximum(self.hits[fi, fj] - 5, 0)
            self.hits[ij[:, 0], ij[:, 1]] = keep
        if len(ij):
            self.hits[ij[:, 0], ij[:, 1]] = np.minimum(self.hits[ij[:, 0], ij[:, 1]] + 1, self.confirm + 10)
        self.seen = self.hits >= self.confirm
        if (self.seen != before).any():
            self.cmap = GridMap(self.base.occ | self.seen, self.base.cell, self.base.x0, self.base.y0)
            self.near_seen = binary_dilation(self.seen, iterations=3)   # 확정 칸 0.3m 둘레 — 가장자리 깜빡임은 정지 규칙에서 뺀다
            if self.backout:                                        # 회전 공간용: 랙 0.4m 안 확정 칸은 위치 오차로 생긴 가짜일 수 있다
                self.smap = GridMap(self.base.occ | (self.seen & ~binary_dilation(self.base.occ, iterations=4)),
                                    self.base.cell, self.base.x0, self.base.y0)
            self.dirty = True
        return bool((self.seen & ~before).any())

    def speed_limit(self, x, y, yaw, obs_xy) -> float:
        """차체 앞 넓은 띠 안 장애물 점으로 최고 속도를 정한다. 앞면 조금 뒤(−0.3m)부터 — 앞 모서리 옆으로 오는 것도 본다."""
        if not len(obs_xy):
            return math.inf
        d = np.asarray(obs_xy) - (x, y)
        u = math.cos(yaw) * d[:, 0] + math.sin(yaw) * d[:, 1] - self.front      # 차체 앞면에서 앞으로
        v = -math.sin(yaw) * d[:, 0] + math.cos(yaw) * d[:, 1]
        if ((u > -0.3) & (u < self.stop) & (np.abs(v) < self.half_w + self.stop_side)).any():
            return 0.0
        if ((u > -0.3) & (u < self.slow) & (np.abs(v) < self.half_w + self.slow_side)).any():
            return self.v_slow
        return math.inf

    def unconfirmed(self, obs_xy, static_margin=0.15):
        """정적 지도 근처도 아니고 확정 칸 0.3m 둘레도 아닌 점 — 움직이는 것이거나 아직 확정 전."""
        if not len(obs_xy):
            return obs_xy
        xy = np.asarray(obs_xy, float)
        m, b = self.static, self.base
        si = np.clip(((xy[:, 1] - m.y0) / m.cell).astype(int), 0, m.occ.shape[0] - 1)
        sj = np.clip(((xy[:, 0] - m.x0) / m.cell).astype(int), 0, m.occ.shape[1] - 1)
        bi = np.clip(((xy[:, 1] - b.y0) / b.cell).astype(int), 0, b.occ.shape[0] - 1)
        bj = np.clip(((xy[:, 0] - b.x0) / b.cell).astype(int), 0, b.occ.shape[1] - 1)
        return xy[(m.dist[si, sj] >= static_margin) & ~self.near_seen[bi, bj]]

    def blocked(self, ahead=80) -> bool:
        """앞으로 갈 경로(0.1m 간격 ahead개)가 지금 지도에서 안전 거리보다 장애물에 가까워졌나."""
        rest = self.pp.path[self.pp.k:self.pp.k + ahead]
        return min(self.cmap.clearance(x, y) for x, y in rest) < self.r - 0.05

    def rot_free(self, x, y, yaw, delta, margin=0.15):
        """yaw에서 delta만큼 제자리 회전할 때 차체 둘레가 지나가는 자리(10°마다)가 모두 장애물에서 margin 이상인가.
        margin 0.15m = 위치 추정 오차(최대 ~10cm) + 여유."""
        m = self.smap or self.cmap
        for th in np.linspace(yaw, yaw + delta, max(2, int(abs(delta) / math.radians(10)) + 2)):
            c, s = math.cos(th), math.sin(th)
            for u, v in self.body_pts:
                if m.clearance(x + c * u - s * v, y + s * u + c * v) < margin:
                    return False
        return True

    def move_free(self, x, y, yaw, v, w, T=0.6, margin=0.05):
        """(v, w)로 T초 갈 때 차체(직사각형)가 지금 보이는 장애물 점에 margin보다 가까워지지 않나 — 이미 그보다 가까우면
        더 가까워지지만 않으면 된다 (교착 방지). 돌며 갈 때 긴 뒤쪽(1.03m)이 옆에 선 사람·물건을 쓰는 것을 막는다."""
        P = self.last_obs
        if P is None or not len(P):
            return True
        P = P[np.hypot(P[:, 0] - x, P[:, 1] - y) < 2.5]
        if not len(P):
            return True

        def gap(px, py, th):
            d = P - (px, py)
            u = math.cos(th) * d[:, 0] + math.sin(th) * d[:, 1]
            q = -math.sin(th) * d[:, 0] + math.cos(th) * d[:, 1]
            return float(np.min(np.hypot(np.maximum(np.maximum(self.back - u, u - self.front), 0), np.maximum(np.abs(q) - self.half_w, 0))))
        now, (px, py, th) = gap(x, y, yaw), (x, y, yaw)
        for _ in range(3):
            th += w * T / 3
            px, py = px + v * T / 3 * math.cos(th), py + v * T / 3 * math.sin(th)
            if gap(px, py, th) < min(margin, now - 0.01):
                return False
        return True

    def replan(self, x, y) -> bool:
        """지금 위치에서 다시 계획. 좁으면 안전 거리를 조금씩 줄여 본다."""
        self.replans += 1
        self.dirty, self.since = False, 0
        for r in (self.r0, self.r0 - 0.15, self.r0 - 0.25):
            p = plan(self.cmap, (x, y), self.goal, robot_r=r, soft_r=1.5)
            if p is not None:
                self.pp, self.r = PurePursuit(p, **self.pp_kw), r
                return True
        if self.backout:                                            # 목적지가 사람·물건에 가렸으면 1m 안 가장 가까운 빈 곳까지 가서 기다림
            m, r = self.cmap, self.r0 - 0.25                        # (비키면 지도가 바뀌어 다시 계획 → 원래 목적지)
            gi, gj = m.ij(*self.goal)
            ii, jj = np.mgrid[max(gi - 10, 0):min(gi + 11, m.occ.shape[0]), max(gj - 10, 0):min(gj + 11, m.occ.shape[1])]
            ok = (m.dist[ii, jj] >= r) & (np.hypot(ii - gi, jj - gj) * m.cell <= 1.0)
            for k in np.argsort(np.hypot(ii - gi, jj - gj)[ok]):
                q = (m.x0 + (jj[ok][k] + 0.5) * m.cell, m.y0 + (ii[ok][k] + 0.5) * m.cell)
                p = plan(m, (x, y), q, robot_r=r, soft_r=1.5)
                if p is not None:
                    self.pp, self.r = PurePursuit(p, **self.pp_kw), r
                    self.dirty = True                               # 곧 다시 원래 목적지를 시도
                    return True
                break
        self.pp = None
        return False

    def step(self, x, y, yaw, obs_xy=None, free_xy=()):
        """(선속도, 각속도, 도착). obs_xy·free_xy = 이번에 센서가 본 장애물·바닥 점 (센서 없는 스텝은 None)."""
        if obs_xy is not None:
            self.last_obs = np.asarray(obs_xy, float).reshape(-1, 2)
            new = self.observe(obs_xy, free_xy)
            self.since += 1
            if self.pp is None or (new and self.blocked()) or (self.dirty and self.since >= self.replan_every):
                self.replan(x, y)                                   # 막힘 / 길이 없던 중 / 지도가 바뀐 지 오래
            self.limit = self.speed_limit(x, y, yaw, self.unconfirmed(obs_xy))
            if self.backout and len(obs_xy):                        # 뒤 1m·차폭+0.3m 안에 뭔가 있으면 후진하지 않는다
                d = np.asarray(obs_xy) - (x, y)
                u = math.cos(yaw) * d[:, 0] + math.sin(yaw) * d[:, 1]
                v = -math.sin(yaw) * d[:, 0] + math.cos(yaw) * d[:, 1]
                self.rear_block = bool(((u < self.back) & (u > self.back - 1.0) & (np.abs(v) < self.half_w + 0.3)).any())
        if self.pp is None:
            return 0.0, 0.0, False
        v, w, done = self.pp.step(x, y, yaw)
        if self.cap is not None and not done:
            if self.cap < 0:                                        # 후진은 곧게 (앞이 막혀 있어도 뒤로는 간다)
                if self.safe_reverse:                               # 뒤 1m 안 물체나 랙·확정 장애물이 가까우면 후진 대신 정지
                    bx, by = x + math.cos(yaw) * (self.back - 0.3), y + math.sin(yaw) * (self.back - 0.3)
                    if self.rear_block or self.cmap.clearance(bx, by) < self.half_w + 0.05:
                        return 0.0, 0.0, False
                return self.cap, 0.0, False
            if v > self.cap:
                w, v = (w * self.cap / v if v > 0 else w), self.cap
        if self.limit is not None and v > self.limit:
            if self.limit == 0.0:                                   # 앞이 막혀 정지 — 경로가 옆으로 꺾이면 제자리에서 돈다
                v, w = 0.0, (math.copysign(0.6 * self.pp.w_max, w) if abs(w) > 0.05 else 0.0)
            else:
                w, v = w * self.limit / v, self.limit
        if self.swept and v > 0.05 and abs(w) > 0.05 and not self.move_free(x, y, yaw, v, w):
            if self.move_free(x, y, yaw, v, 0.0):                   # 돌며 가면 긴 뒤쪽·옆이 물체로 쓸림 → 곧게 (되면)
                w = 0.0
            else:
                v = 0.0                                             # 아니면 아래 제자리 회전 판단(쓸림 확인·반대 방향·후진)으로
        if self.swept and abs(v) < 0.05 and abs(w) > 0.05:           # 제자리 회전 요청 — 쓸고 지나갈 자리를 정확히 본다
            a = self.pp.heading_err(x, y, yaw)
            if self.rot_free(x, y, yaw, a):
                self.backed = self.crept = 0.0
                return v, w, done
            other = a - math.copysign(2 * math.pi, a)               # 반대 방향으로 돌면 되나
            if self.rot_free(x, y, yaw, other):
                self.backed = self.crept = 0.0
                return 0.0, math.copysign(0.6 * self.pp.w_max, other), done
            room = 0.0                                              # 양쪽 다 막힘 → 아래 후진 추종·후진·전진·대기
        else:
            room = (self.smap or self.cmap).clearance(x, y) if self.backout else 9.0
        if self.backout and abs(v) < 0.05 and abs(w) > 0.05 and room < self.sweep:
            rv = self.pp.reverse_cmd(x, y, yaw)                     # 좁아서 못 도는데 경로가 뒤쪽 → 후진으로 경로 추종 (U턴 대신)
            if rv is not None:
                return (0.0, 0.0, done) if self.rear_block else (*rv, done)
        if self.backout and abs(v) < 0.05 and abs(w) > 0.05 and self.backed < 4.0 and room < self.sweep:
            if not self.rear_block:
                self.backed += 0.25 / 30                            # 0.25m/s 후진 (한 스텝 = 1/30초) — 최대 4m (통로 입구까지)
                return -0.25, 0.0, done
            fx, fy = x + math.cos(yaw) * (self.front + 0.3), y + math.sin(yaw) * (self.front + 0.3)
            if self.crept < 1.5 and self.limit != 0.0 and self.cmap.clearance(fx, fy) > self.half_w + 0.1:
                self.crept += 0.2 / 30                              # 뒤가 막혔으면 앞이 빈 만큼 똑바로 조금 전진 (최대 1.5m) — 공간을 만들고 돈다
                return 0.2, 0.0, done
            return 0.0, 0.0, done                                   # 돌 자리도, 앞뒤로 뺄 자리도 없으면 기다린다
        if abs(w) <= 0.05 or abs(v) >= 0.05:
            self.backed = self.crept = 0.0
        return v, w, done


SPEEDS = (1.0, 0.6, 0.3, 0.0, -0.3)        # 앞보기 후보 속도 (m/s). 음수 = 곧게 후진


def choose_speed(risks, tau=0.1):
    """후보별 (충돌 위험 0~1, 충돌까지 시간 s)에서: 위험 < tau인 것 중 가장 빠른 속도.
    모두 위험하면 충돌까지 가장 오래 버티는 것 (같으면 위험이 낮은 것)."""
    safe = [v for v, (p, _) in zip(SPEEDS, risks) if p < tau]
    if safe:
        return max(safe)
    return max(zip(SPEEDS, risks), key=lambda c: (c[1][1], -c[1][0]))[0]


def choose_speed_score(risks, lam=12.0, horizon=2.0):
    """속도 점수 = horizon초 동안 나아가는 거리 − lam × 충돌 확률 (lam = 충돌 한 번을 몇 m 손해로 치나).
    얻는 속도와 늘어나는 위험을 저울질 — 1.0이 0.6보다 2초에 0.8m 더 가므로, 위험이 0.8/lam(=6.7%p) 넘게 높지 않으면 빠른 쪽."""
    return max(zip(SPEEDS, risks), key=lambda c: (horizon * c[0] - lam * c[1][0], c[0]))[0]


class Tracker:
    """비교 기준 B′ — 움직이는 장애물을 덩어리로 묶어 등속으로 이어 간다.
    정적 지도 밖 장애물 점을 cell 격자에서 묶고(한 칸 팽창으로 두 다리를 한 덩어리로), 이전 덩어리 중 gate m 안 가장
    가까운 것과 이어 속도를 잰다(지수 평활 alpha). 안 보이게 된 덩어리도 keep 초 동안은 같은 속도로 계속 간다고 본다."""

    def __init__(self, cell=0.15, gate=1.0, alpha=0.5, keep=1.0):
        self.cell, self.gate, self.alpha, self.keep = cell, gate, alpha, keep
        self.tracks, self.t = [], None                              # 덩어리: c 중심, v 속도, pts 점, seen 마지막 본 시각

    def update(self, t, pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        dt = 0.0 if self.t is None else t - self.t
        self.t = t
        for tr in self.tracks:                                     # 지난 덩어리는 등속으로 지금까지 옮겨 둔다
            tr["c"], tr["pts"] = tr["c"] + tr["v"] * dt, tr["pts"] + tr["v"] * dt
        new = []
        if len(pts):
            q = np.floor(pts / self.cell).astype(int)
            lo = q.min(0)
            grid = np.zeros(tuple(q.max(0) - lo + 1), bool)
            grid[tuple((q - lo).T)] = True
            lab, _ = label(binary_dilation(grid, iterations=1), structure=np.ones((3, 3)))
            ids = lab[tuple((q - lo).T)]
            for k in np.unique(ids):
                cp = pts[ids == k]
                new.append({"c": cp.mean(0), "v": np.zeros(2), "pts": cp, "seen": t, "n": 1})
        used = set()
        for nt in new:                                             # 가까운 지난 덩어리와 잇기
            best, bd = None, self.gate
            for i, tr in enumerate(self.tracks):
                d = float(np.hypot(*(nt["c"] - tr["c"])))
                if i not in used and d < bd:
                    best, bd = i, d
            if best is not None and dt > 0:
                tr = self.tracks[best]
                used.add(best)
                last = tr["c"] - tr["v"] * (t - tr["seen"])                   # 마지막으로 본 중심 (그 뒤로 등속으로 옮겨 뒀다)
                vel = (nt["c"] - last) / (t - tr["seen"])
                nt["v"] = vel if tr["n"] == 1 else self.alpha * vel + (1 - self.alpha) * tr["v"]   # 첫 연결은 잰 속도 그대로
                nt["n"] = tr["n"] + 1
        keep = [tr for i, tr in enumerate(self.tracks) if i not in used and t - tr["seen"] <= self.keep]
        self.tracks = new + keep

    def risks(self, path_pts, v_now_path, front=0.4, back=1.03, half_w=0.33, horizon=2.0, dt=0.2, margin=0.15):
        """후보 속도마다 (충돌 위험 0/1, 충돌까지 시간). 앞으로는 경로를 따라(v·τ만큼), 후진은 지금 방향 반대로 곧게 간다고 본다.
        path_pts: 지금 위치부터 앞으로의 경로 점 (0.1m 간격) — [0]이 지금 위치, 방향은 path_pts[1]−[0]."""
        out = []
        P = np.asarray(path_pts, float)
        seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        h0 = math.atan2(*(P[min(5, len(P) - 1)] - P[0])[::-1]) if len(P) > 1 else 0.0
        movers = [tr for tr in self.tracks if np.hypot(*tr["v"]) > 0.2]   # 거의 안 움직이는 것은 경로 계획이 맡는다
        for v in SPEEDS:
            hit = horizon + 1.0
            for tau in np.arange(dt, horizon + 1e-6, dt):
                s = v * tau
                if s >= 0:
                    k = min(int(np.searchsorted(seg, s)), len(P) - 1)
                    x, y = P[k]
                    k2 = min(k + 5, len(P) - 1)
                    hd = math.atan2(*(P[k2] - P[k])[::-1]) if k2 > k else h0
                else:
                    x, y, hd = P[0][0] + s * math.cos(h0), P[0][1] + s * math.sin(h0), h0
                c, sn = math.cos(hd), math.sin(hd)
                for tr in movers:
                    q = tr["pts"] + tr["v"] * tau - (x, y)
                    u, w = c * q[:, 0] + sn * q[:, 1], -sn * q[:, 0] + c * q[:, 1]
                    if ((u > -back - margin) & (u < front + margin) & (np.abs(w) < half_w + margin)).any():
                        hit = tau
                        break
                if hit <= horizon:
                    break
            out.append((1.0 if hit <= horizon else 0.0, hit))
        return out


BEV_N, BEV_CELL, BEV_ROW0 = 64, 0.15, 44    # 64×64칸 × 0.15m = 9.6m. AGV 기준점이 위에서 44번째 줄 가운데 → 앞 6.6m, 뒤 3m, 좌우 4.8m
_I, _J = np.mgrid[0:BEV_N, 0:BEV_N]
BEV_FWD, BEV_LEFT = (BEV_ROW0 - _I - 0.5) * BEV_CELL, (BEV_N / 2 - _J - 0.5) * BEV_CELL   # 칸 중심의 AGV 기준 (앞, 왼쪽) m


def bev(static: GridMap, x, y, yaw, obs_xy=(), free_xy=(), static_margin=0.15) -> np.ndarray:
    """AGV 중심·진행 방향이 위인 주변 지도 (64, 64, 3) uint8 — 월드모델 입력.
    R = 정적 지도(랙·벽), G = 이번에 본 정적 지도 밖 장애물 점, B = 이번에 보인 바닥 점 (보이는 곳·빈 곳).
    AGV 센서로 본 것만 다시 그린 것이다 (천장 카메라 아님)."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.zeros((BEV_N, BEV_N, 3), np.uint8)
    wx, wy = x + c * BEV_FWD - s * BEV_LEFT, y + s * BEV_FWD + c * BEV_LEFT
    i, j = ((wy - static.y0) / static.cell).astype(int), ((wx - static.x0) / static.cell).astype(int)
    inside = (i >= 0) & (i < static.occ.shape[0]) & (j >= 0) & (j < static.occ.shape[1])
    out[..., 0] = np.where(inside, static.occ[i.clip(0, static.occ.shape[0] - 1), j.clip(0, static.occ.shape[1] - 1)], True) * 255

    def paint(xy, ch, keep=None):
        xy = np.asarray(xy, float).reshape(-1, 2)
        if keep is not None:
            xy = xy[keep(xy)]
        d = xy - (x, y)
        fwd, left = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        r, q = np.floor(BEV_ROW0 - fwd / BEV_CELL).astype(int), np.floor(BEV_N / 2 - left / BEV_CELL).astype(int)
        ok = (r >= 0) & (r < BEV_N) & (q >= 0) & (q < BEV_N)
        out[r[ok], q[ok], ch] = 255

    def off_static(xy):
        si = np.clip(((xy[:, 1] - static.y0) / static.cell).astype(int), 0, static.occ.shape[0] - 1)
        sj = np.clip(((xy[:, 0] - static.x0) / static.cell).astype(int), 0, static.occ.shape[1] - 1)
        return static.dist[si, sj] >= static_margin
    if len(free_xy):
        paint(free_xy, 2)
    if len(obs_xy):
        paint(obs_xy, 1, off_static)
    return out


def demo():
    occ = np.zeros((60, 60), bool)
    occ[:, 29:31] = True                                            # 가운데 벽
    occ[24:36, 29:31] = False                                       # 벽 사이 1.2m 틈 (칸 0.1m)
    occ[[0, -1], :] = occ[:, [0, -1]] = True
    m = GridMap(occ, 0.1, 0.0, 0.0)
    path = plan(m, (1.0, 1.0), (5.0, 1.0), robot_r=0.4, soft_r=0.8)   # 직선은 벽에 막힘 → 틈으로 돌아가야 함
    assert path is not None and len(path) >= 3, path
    assert all(m.clearance(*p) >= 0.4 - 1e-6 for p in densify(path, 0.05)), "경로가 장애물에 너무 가깝다"
    assert plan(m, (1.0, 1.0), (5.0, 1.0), robot_r=0.7) is None, "틈보다 큰 로봇은 못 지나가야 한다"
    x, y, yaw, dt = 1.0, 1.0, 0.0, 1 / 30
    pp = PurePursuit(path, lookahead=0.5, v_max=0.8)
    for n in range(3000):
        v, w, done = pp.step(x, y, yaw)
        if done:
            break
        x, y, yaw = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw + w * dt
        assert m.clearance(x, y) > 0.2, f"추종 중 장애물 근접 ({x:.2f}, {y:.2f})"
    assert done and math.hypot(x - 5, y - 1) < 0.3, (x, y, n)
    print(f"demo OK — 경유점 {len(path)}개, {n} 스텝({n * dt:.1f}초)에 도착")

    # 재계획: 지도에 없는 상자가 직선 경로 위에 있다. 센서(반경 3m 안 상자 칸)로 보면 피하고, 안 보면 부딪힌다.
    free = np.zeros((60, 60), bool)
    free[[0, -1], :] = free[:, [0, -1]] = True
    box = np.zeros_like(free)
    box[27:33, 27:33] = True                                        # (2.7~3.3, 2.7~3.3) 0.6m 상자
    static, truth = GridMap(free, 0.1, 0.0, 0.0), GridMap(free | box, 0.1, 0.0, 0.0)
    bxy = np.array([static.xy(i, j) for i, j in zip(*np.nonzero(box))])
    for sense in (False, True):
        nav = Navigator(static, (1.0, 1.0), (5.0, 5.0), robot_r=0.5, confirm=3, lookahead=0.5, v_max=0.8)
        x, y, yaw, low = 1.0, 1.0, math.pi / 4, 9.0
        for n in range(3000):
            near = bxy[np.hypot(bxy[:, 0] - x, bxy[:, 1] - y) < 3.0] if sense else None
            v, w, done = nav.step(x, y, yaw, near, ())
            if done:
                break
            x, y, yaw = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw + w * dt
            low = min(low, truth.clearance(x, y))
        if sense:
            assert done and low > 0.3 and nav.replans >= 1, (done, low, nav.replans)
        else:
            assert low == 0.0, "대응 없음이면 상자를 지나가야(부딪혀야) 한다"
    print(f"demo OK — 재계획 {nav.replans}회로 상자를 {low:.2f}m 떨어져 피해 도착, 대응 없음은 부딪힘")

    # 가로지르는 사람(반지름 0.3m 원판): 로봇 경로 (1,3)→(5,3)를 x=3.5에서 아래→위로 1.2m/s로 건넌다.
    # 센서 = 앞 ±30°, 3m 안 사람 점과 바닥 점. 대응하면 멈췄다 지나가고, 지나간 자리는 지워져 돌아가지 않는다.
    gx, gy = np.meshgrid(np.arange(0.05, 6, 0.1), np.arange(0.05, 6, 0.1))
    floor = np.c_[gx.ravel(), gy.ravel()]
    for sense in (False, True):
        nav = Navigator(static, (1.0, 3.0), (5.0, 3.0), robot_r=0.5, front=0.4, half_w=0.33, lookahead=0.5, v_max=0.8)
        x, y, yaw, low, t0 = 1.0, 3.0, 0.0, 9.0, None
        for n in range(3000):
            t0 = n if t0 is None and x > 1.6 else t0                   # 로봇이 1.6m 지나면 사람 출발
            py = 1.0 + 1.2 * (n - t0) * dt if t0 is not None else 1.0
            py = min(py, 5.5)
            ang = np.arctan2(np.r_[py - y], np.r_[3.5 - x]) - yaw
            person = np.array([[3.5 + 0.3 * math.cos(a), py + 0.3 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 12)])
            inview = lambda q: (np.hypot(q[:, 0] - x, q[:, 1] - y) < 3) & (np.abs(np.arctan2(q[:, 1] - y, q[:, 0] - x) - yaw) < 1.05)  # noqa: E731
            obs = person[inview(person)] if sense and n % 3 == 0 else None
            fl = floor[inview(floor) & (np.hypot(floor[:, 0] - 3.5, floor[:, 1] - py) > 0.4)] if obs is not None else ()
            v, w, done = nav.step(x, y, yaw, obs, fl)
            if done:
                break
            x, y, yaw = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw + w * dt
            c, s_ = math.cos(yaw), math.sin(yaw)
            du, dv = c * (3.5 - x) + s_ * (py - y), -s_ * (3.5 - x) + c * (py - y)   # 사람 중심 → 차체 직사각형(뒤 0.6, 앞 0.4)
            low = min(low, math.hypot(max(-0.6 - du, 0, du - 0.4), max(abs(dv) - 0.33, 0)) - 0.3)
        if sense:
            assert done and low > 0.1 and n * dt < 12, (done, low, n * dt)
        else:
            assert low < 0, "대응 없음이면 사람과 부딪혀야 한다"
    print(f"demo OK — 가로지르는 사람: 대응하면 {low:.2f}m 남기고 {n * dt:.1f}초에 도착, 대응 없음은 부딪힘")

    # BEV: 20×20m 지도, 벽 x = 10.0~10.1. AGV (8, 10)이 +x를 보면 벽은 앞 2m 가로줄,
    # AGV (9.5, 8)이 +y를 보면 벽은 오른쪽 0.5m 세로줄로 보여야 한다
    occ = np.zeros((200, 200), bool)
    occ[:, 100] = True
    m = GridMap(occ, 0.1, 0.0, 0.0)
    b = bev(m, 8.0, 10.0, 0.0, obs_xy=[(9.0, 10.6)], free_xy=[(8.5, 10.0)])
    rows = np.nonzero(b[:, 32, 0])[0]
    assert len(rows) <= 2 and abs((BEV_ROW0 - rows.max() - 1) * BEV_CELL - 2.0) < 0.16, rows   # 앞 2m에 벽 한 줄
    r, q = np.argwhere(b[..., 1])[0]
    assert abs((BEV_ROW0 - r - 0.5) * BEV_CELL - 1.0) < 0.1 and abs((BEV_N / 2 - q - 0.5) * BEV_CELL - 0.6) < 0.1, (r, q)
    assert b[BEV_ROW0 - 4, 32, 2] == 255                                             # 앞 0.5m 바닥 보임
    b2 = bev(m, 9.5, 8.0, math.pi / 2)
    cols = np.nonzero(b2[:, :, 0].all(axis=0))[0]
    assert len(cols) and all(abs((BEV_N / 2 - c - 0.5) * BEV_CELL + 0.5) < 0.16 for c in cols), cols  # 오른쪽 0.5m 세로줄
    print("demo OK — BEV: 앞 2m 벽, 앞 1m·왼쪽 장애물, 회전 시 벽 방향 맞음")

    # B′ 등속 외삽: 사람(점 묶음)이 AGV 앞 2m·왼쪽 3m에서 AGV 경로(+x 직선)를 향해 1.2m/s로 내려온다.
    # 1m/s로 그대로 가면 2초 안에 만나고, 멈추면 안 만나야 한다. 화각에서 사라져도 1초는 이어 가야 한다.
    tk = Tracker()
    legs = np.array([[0.0, 0.0], [0.25, 0.0], [0.1, 0.1]])
    for k in range(4):                                             # 0.2초마다 (2, 3) → 아래로
        tk.update(0.2 * k, legs + (2.0, 3.0 - 1.2 * 0.2 * k))
    v_est = tk.tracks[0]["v"]
    assert abs(v_est[1] + 1.2) < 0.15 and abs(v_est[0]) < 0.1, v_est
    path = [(0.1 * i, 0.0) for i in range(80)]                     # AGV는 (0,0)에서 +x로
    r = dict(zip(SPEEDS, tk.risks(path, 1.0)))
    assert r[1.0][0] == 1.0 and r[0.0][0] == 0.0, r
    tk.update(1.0, [])                                             # 안 보여도
    assert len(tk.tracks) == 1 and tk.tracks[0]["c"][1] < 3.0 - 1.2 * 0.9, "보이지 않아도 등속으로 이어 가야 한다"
    tk.update(2.2, [])
    assert not tk.tracks, "keep(1초)이 지나면 버린다"
    nv = Navigator(GridMap(np.zeros((80, 80), bool), 0.1, 0.0, 0.0), (2.0, 4.0), (7.0, 4.0), robot_r=0.5, swept=True)
    nv.last_obs = np.array([[2.0 - 0.9, 4.0 + 0.40]])                # 뒤 왼쪽 옆 0.07m에 선 사람
    assert not nv.move_free(2.0, 4.0, 0.0, 0.24, -0.38), "오른쪽으로 돌며 가면 뒤가 왼쪽으로 쓸려 부딪힌다"
    assert nv.move_free(2.0, 4.0, 0.0, 0.24, 0.0) and nv.move_free(2.0, 4.0, 0.0, 0.24, 0.38), "곧게·왼쪽으로는 괜찮다"
    nv.last_obs = np.array([[2.0 - 0.9, 4.0 + 0.30]])                # 이미 닿아 있으면 막지 않는다 (교착 방지)
    assert nv.move_free(2.0, 4.0, 0.0, 0.24, -0.38)
    assert choose_speed([(0.0, 3), (0.0, 3), (0.0, 3), (0.0, 3), (0.0, 3)]) == 1.0
    assert choose_speed([(1.0, 0.4), (1.0, 0.6), (0.05, 3), (0.0, 3), (0.0, 3)]) == 0.3
    assert choose_speed([(1.0, 0.4), (1.0, 0.6), (1.0, 0.8), (1.0, 1.0), (1.0, 1.6)]) == -0.3
    print(f"demo OK — 등속 외삽: 속도 {np.round(v_est, 2)} m/s 추정, 그대로 가면 충돌·멈추면 안전, 화각 밖 1초 유지")


if __name__ == "__main__":
    demo()
