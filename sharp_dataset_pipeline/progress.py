import time


class OrderedProgress:
    def __init__(self, range_start: int, range_end: int, frontier_offset: int = 0):
        self.range_start = int(range_start)
        self.range_end = int(range_end)
        self.frontier = int(frontier_offset)
        self.holes = []
        self.offset_to_id = {}
        self.done_confirmed = set()
        self.claimed = set()
        self.blocked_until = {}

    def _in_range(self, offset: int) -> bool:
        try:
            o = int(offset)
        except Exception:
            return False
        return (o >= int(self.range_start)) and (o <= int(self.range_end))

    def remember(self, offset: int, photo_id: str) -> None:
        if not self._in_range(offset):
            return
        pid = str(photo_id or '').strip()
        if not pid:
            return
        self.offset_to_id[int(offset)] = pid

    def _add_hole_point(self, offset: int) -> None:
        if not self._in_range(offset):
            return
        o = int(offset)
        if o < int(self.frontier):
            return
        for l, r in self.holes:
            if l <= o <= r:
                return
        new_l = o
        new_r = o
        out = []
        inserted = False
        for l, r in self.holes:
            if r + 1 < new_l:
                out.append((l, r))
                continue
            if new_r + 1 < l:
                if not inserted:
                    out.append((new_l, new_r))
                    inserted = True
                out.append((l, r))
                continue
            new_l = min(new_l, l)
            new_r = max(new_r, r)
        if not inserted:
            out.append((new_l, new_r))
        self.holes = sorted(out, key=lambda x: int(x[0]))

    def _remove_hole_point(self, offset: int) -> None:
        if not self.holes:
            return
        try:
            o = int(offset)
        except Exception:
            return
        out = []
        for l, r in self.holes:
            if o < l or o > r:
                out.append((l, r))
                continue
            if l == r == o:
                continue
            if o == l:
                out.append((l + 1, r))
                continue
            if o == r:
                out.append((l, r - 1))
                continue
            out.append((l, o - 1))
            out.append((o + 1, r))
        self.holes = out

    def mark_seen_unfinished(self, offset: int) -> None:
        self._add_hole_point(offset)

    def mark_done(self, offset: int) -> None:
        if not self._in_range(offset):
            return
        o = int(offset)
        self.done_confirmed.add(o)
        self.claimed.discard(o)
        self.blocked_until.pop(o, None)
        self._remove_hole_point(o)
        while (self.frontier in self.done_confirmed) or (self.frontier in self.claimed):
            self.frontier += 1

    def mark_filled(self, offset: int) -> None:
        self.mark_done(offset)

    def mark_claimed(self, offset: int, hold_secs: float) -> None:
        try:
            hold = float(hold_secs)
        except Exception:
            hold = 60.0
        try:
            o = int(offset)
            if not self._in_range(o):
                return
            self.claimed.add(o)
            self._remove_hole_point(o)
            while (self.frontier in self.done_confirmed) or (self.frontier in self.claimed):
                self.frontier += 1
        except Exception:
            pass
        self.mark_blocked(offset, time.time() + max(5.0, hold))

    def mark_claimed_until(self, offset: int, until_ts: float) -> None:
        try:
            o = int(offset)
        except Exception:
            return
        if not self._in_range(o):
            return
        self.claimed.add(o)
        self._remove_hole_point(o)
        try:
            self.mark_blocked(o, float(until_ts))
        except Exception:
            self.mark_blocked(o, time.time() + 30.0)
        while (self.frontier in self.done_confirmed) or (self.frontier in self.claimed):
            self.frontier += 1

    def mark_error_retry(self, offset: int, hold_secs: float) -> None:
        try:
            o = int(offset)
        except Exception:
            return
        if not self._in_range(o):
            return
        self._add_hole_point(o)
        self.mark_blocked(o, time.time() + max(1.0, float(hold_secs)))

    def refresh_expired(self, now_ts: float) -> None:
        try:
            now = float(now_ts)
        except Exception:
            now = time.time()
        expired = []
        for o in list(self.claimed):
            try:
                bu = float(self.blocked_until.get(int(o), 0.0) or 0.0)
            except Exception:
                bu = 0.0
            if bu <= now:
                expired.append(int(o))
        for o in expired:
            self.claimed.discard(int(o))
            try:
                self.blocked_until.pop(int(o), None)
            except Exception:
                pass
            if int(o) not in self.done_confirmed:
                self._add_hole_point(int(o))

    def mark_blocked(self, offset: int, until_ts: float) -> None:
        if not self._in_range(offset):
            return
        o = int(offset)
        try:
            until_v = float(until_ts)
        except Exception:
            until_v = time.time() + 30.0
        prev = self.blocked_until.get(o)
        if (prev is None) or (float(until_v) > float(prev)):
            self.blocked_until[o] = float(until_v)

    def has_pending(self) -> bool:
        return bool(self.holes)

    def next_hole_offset(self, now_ts: float):
        try:
            now = float(now_ts)
        except Exception:
            now = time.time()
        self.refresh_expired(now)
        for l, r in self.holes:
            o = int(l)
            while o <= int(r):
                if o < int(self.frontier):
                    o += 1
                    continue
                if o not in self.offset_to_id:
                    o += 1
                    continue
                bu = float(self.blocked_until.get(o, 0.0) or 0.0)
                if bu > now:
                    o += 1
                    continue
                return int(o)
        return None

    def to_dict(self) -> dict:
        try:
            return {
                "range_start": int(self.range_start),
                "range_end": int(self.range_end),
                "frontier": int(self.frontier),
                "holes": [(int(l), int(r)) for (l, r) in (self.holes or [])],
            }
        except Exception:
            return {
                "range_start": int(self.range_start),
                "range_end": int(self.range_end),
                "frontier": int(self.frontier),
                "holes": [],
            }

    def apply_dict(self, obj: dict) -> None:
        if not isinstance(obj, dict):
            return
        try:
            f = obj.get("frontier")
            if f is not None:
                ff = int(f)
                if int(self.range_start) <= int(ff) <= int(self.range_end) + 1:
                    self.frontier = int(ff)
        except Exception:
            pass
        try:
            holes = obj.get("holes")
            out = []
            if isinstance(holes, list):
                for it in holes:
                    if not (isinstance(it, (list, tuple)) and len(it) == 2):
                        continue
                    l, r = it
                    try:
                        ll = int(l)
                        rr = int(r)
                    except Exception:
                        continue
                    if rr < ll:
                        continue
                    ll = max(int(ll), int(self.range_start))
                    rr = min(int(rr), int(self.range_end))
                    if ll <= rr:
                        out.append((int(ll), int(rr)))
            self.holes = sorted(out, key=lambda x: int(x[0]))
        except Exception:
            pass
