# Gom chữ từ bản viết trên vở ô ly (26/10): đường chân chữ và cỡ chữ lấy theo dòng kẻ của vở
import json, numpy as np, cv2, os, sys
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE + '/../font/src2/'; SRCB = HERE + '/../font/src2b/'
OLD = HERE + '/../font/work2'
OUT = HERE + '/work'; os.makedirs(OUT, exist_ok=True)
FNT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
XH1 = 35.0      # chiều cao chữ thường chuẩn (px) sau khi quy cỡ, giữ như bản trước để phần dựng font không đổi

def load(path):
    im = cv2.imread(path)
    return im.max(2)            # kênh sáng nhất: dòng kẻ xanh, lề đỏ gần như biến mất, mực vẫn đậm

# ---------- dòng kẻ ----------
def big_square(g):
    H, W = g.shape
    ink = cv2.dilate((g < 200).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    Ps = []
    for x0 in range(int(W * 0.2), int(W * 0.8) - 200, 150):
        d = 255.0 - g[:, x0:x0 + 200]; m = ~ink[:, x0:x0 + 200]
        prof = (d * m).sum(1) / np.maximum(1, m.sum(1)); prof = prof - np.convolve(prof, np.ones(41) / 41, 'same')
        prof = prof[int(H * 0.08):int(H * 0.9)]
        ac = np.correlate(prof, prof, 'full')[len(prof) - 1:]
        Ps.append(80 + int(np.argmax(ac[80:170])))
    return float(np.median(Ps))

def line_profile(g, inkmask, cx, half=150):
    H, W = g.shape
    x0, x1 = max(0, int(cx - half)), min(W, int(cx + half))
    d = 255.0 - g[:, x0:x1]; m = ~inkmask[:, x0:x1]
    prof = (d * m).sum(1) / np.maximum(1, m.sum(1))
    return np.convolve(prof, np.ones(3) / 3, 'same')

def peak_near(prof, y, rad):
    lo, hi = max(1, int(y - rad)), min(len(prof) - 1, int(y + rad))
    if hi <= lo: return None
    k = lo + int(np.argmax(prof[lo:hi]))
    # đỉnh thật: tinh chỉnh dưới px bằng parabol
    if 0 < k < len(prof) - 1:
        a, b, c = prof[k - 1], prof[k], prof[k + 1]; den = a - 2 * b + c
        if den < 0: return k + 0.5 * (a - c) / den
    return float(k)

from clean import cleaned
class Page:
    def __init__(self, name, path, box=None):
        self.name = name; self.raw = load(path); self.g, _ = cleaned(path)
        self.P = big_square(self.raw); self.s = self.P / 69.0
        self.inkmask = cv2.dilate((self.g < 205).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        self.box = box
    def baseline_at(self, cx, est):
        """dòng kẻ đậm gần ước lượng nhất tại hoành độ cx, và khoảng cách ô lớn tại chỗ đó"""
        prof = line_profile(self.raw, self.inkmask, cx)
        y = peak_near(prof, est, 0.3 * self.P)
        up = peak_near(prof, y - self.P, 0.15 * self.P)
        dn = peak_near(prof, y + self.P, 0.15 * self.P)
        cand = [y - up] if up is not None else []
        if dn is not None: cand.append(dn - y)
        cand = [c for c in cand if 0.85 * self.P < c < 1.15 * self.P]
        return y, (float(np.mean(cand)) if cand else self.P)

# ---------- tách mảnh mực ----------
def components(pg, box=None, thr=200, edge=25, erode=45):
    g = pg.g; s = pg.s; H, W = g.shape
    b = (cv2.GaussianBlur(g, (0, 0), 6) > 170).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 4)
    big = 1 + int(np.argmax(st[1:, 4]))
    cnts, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper = np.zeros_like(b); cv2.drawContours(paper, cnts, -1, 1, -1)
    paper = cv2.erode(paper, np.ones((erode, erode), np.uint8))
    ink = ((g < thr) & (paper > 0)).astype(np.uint8)
    box = box or pg.box
    if box is not None:
        m = np.zeros_like(ink); x0, y0, x1, y1 = box; m[y0:y1, x0:x1] = 1; ink &= m
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    C = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        if a < 10 * s * s or x <= edge or y <= edge or x + w >= W - edge or y + h >= H - edge: continue
        if w > 9 * pg.P: continue                       # gạch ngang dài
        if x + w < 0.13 * W: continue                    # bên trái lề đỏ
        vals = g[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i]
        if np.percentile(vals, 20) > 165: continue         # phần dòng kẻ còn sót (nhạt hơn mực)
        C.append(dict(i=i, x=x, y=y, w=w, h=h, a=a, cx=float(cen[i, 0]), cy=float(cen[i, 1])))
    for c in C:            # tâm dọc của phần thân: hàng có nhiều mực nhất (bỏ qua vòng trên, vòng dưới mảnh)
        m = (lab[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] == c['i']).sum(1).astype(float)
        k = max(1, int(0.12 * pg.P)); m = np.convolve(m, np.ones(2 * k + 1), 'same')
        c['by'] = c['y'] + float(np.argmax(m))
    return lab, C

def cluster_rows(C, P, amin, nrows, dy_link=0.55):
    """gom các mảnh lớn thành dòng: hai mảnh cùng dòng nếu gần nhau theo ngang và lệch dọc ít (dòng có thể hơi nghiêng)"""
    big = [c for c in C if c['a'] >= amin]
    par = list(range(len(big)))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            a, b = big[i], big[j]
            if abs(a['by'] - b['by']) < dy_link * P and abs(a['cx'] - b['cx']) < 3.2 * P: par[f(i)] = f(j)
    groups = {}
    for i in range(len(big)): groups.setdefault(f(i), []).append(big[i])
    rows = sorted(groups.values(), key=lambda g: np.mean([c['by'] for c in g]))
    while len(rows) > nrows:
        areas = [sum(c['a'] for c in r) for r in rows]; med = np.median(areas)
        tiny = [i for i, a in enumerate(areas) if a < 0.25 * med]
        if tiny:
            i = min(tiny, key=lambda i: areas[i]); print('   bo cum nho', [(int(c['cx']), int(c['cy']), c['a']) for c in rows[i]]); rows.pop(i); continue
        my = [np.mean([c['by'] for c in r]) for r in rows]
        k = min(range(len(rows) - 1), key=lambda i: my[i + 1] - my[i])
        rows[k] = rows[k] + rows.pop(k + 1)
    if len(rows) != nrows: print('  !! so dong', len(rows), 'can', nrows, [len(r) for r in rows])
    fits = []
    for r in rows:
        xs = np.array([c['cx'] for c in r]); ys = np.array([c['by'] for c in r])
        fits.append(np.polyfit(xs, ys, 1) if len(r) >= 3 and np.ptp(xs) > P else np.array([0.0, ys.mean()]))
    def rf(c):
        d = [abs(c['by'] - np.polyval(ft, c['cx'])) for ft in fits]
        return int(np.argmin(d))
    return rf, rows

def rows_of(C, nrows, amin):
    ys = sorted(set(round(c['cy']) for c in C if c['a'] >= amin))
    gaps = sorted([(ys[i + 1] - ys[i], i) for i in range(len(ys) - 1)], reverse=True)[:nrows - 1]
    cuts = sorted((ys[i] + ys[i + 1]) / 2 for _, i in gaps)
    return (lambda c: sum(c['cy'] > t for t in cuts)), cuts

def group_row(cs, expect, s, attach_small=True):
    cs = sorted(cs, key=lambda c: c['x'])
    top = sorted([c['a'] for c in cs], reverse=True)[:max(1, len(expect))]
    amed = float(np.median(top)) if top else 0
    seeds = [c for c in cs if c['a'] >= 0.3 * amed] if attach_small else list(cs)
    small = [c for c in cs if c not in seeds]
    groups = [[q] for q in seeds]
    for c in small:            # chấm, râu nhỏ: gắn vào chữ gần nhất nếu nằm phía trên
        best, bd = None, 1e9
        for gr in groups:
            for q in gr[:1]:
                dx = max(q['x'] - c['cx'], 0, c['cx'] - (q['x'] + q['w'])); dy = max(q['y'] - c['cy'], 0, c['cy'] - (q['y'] + q['h']))
                d = (dx * dx + dy * dy) ** 0.5
                if d < bd: bd, best = d, gr
        if best and bd < 55 * s:
            sd = best[0]
            if c['cy'] < sd['y'] + sd['h']: best.append(c)        # mảnh nằm trong hoặc trên chữ; mảnh vụn phía dưới chữ thì bỏ
    def ext(gr): return min(c['x'] for c in gr), max(c['x'] + c['w'] for c in gr)
    merged = True
    while merged:
        merged = False
        groups.sort(key=lambda gr: ext(gr)[0])
        for i in range(len(groups) - 1):
            a0, a1 = ext(groups[i]); b0, b1 = ext(groups[i + 1])
            if min(a1, b1) - max(a0, b0) > 0.5 * min(a1 - a0, b1 - b0):
                groups[i] += groups.pop(i + 1); merged = True; break
    while len(groups) > len(expect):
        groups.sort(key=lambda gr: ext(gr)[0])
        k = min(range(len(groups) - 1), key=lambda i: ext(groups[i + 1])[0] - ext(groups[i])[1])
        groups[k] += groups.pop(k + 1)
    groups.sort(key=lambda gr: ext(gr)[0])
    return groups

def alpha_of(g, lab, parts, pad=8, hi=235):
    xs = [p['x'] for p in parts]; ys = [p['y'] for p in parts]
    x0 = max(0, min(xs) - pad); y0 = max(0, min(ys) - pad)
    x1 = max(p['x'] + p['w'] for p in parts) + pad; y1 = max(p['y'] + p['h'] for p in parts) + pad
    m = np.isin(lab[y0:y1, x0:x1], [p['i'] for p in parts]).astype(np.uint8)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    crop = g[y0:y1, x0:x1].astype(float)
    dark = np.percentile(crop[m > 0], 4)
    a = np.clip((hi - crop) / max(30, hi - dark - 15), 0, 1) * m      # nền vở có dòng kẻ ~220–245: cắt ở 235
    return a.astype(np.float32), [int(x0), int(y0)], float(dark)

def ink_box(a, org, t=0.5):
    ys, xs = np.nonzero(a > t)
    return [int(xs.min() + org[0]), int(ys.min() + org[1]), int(xs.max() + org[0]), int(ys.max() + org[1])]

DB = []
def add(page, kind, key, var, a, org, k, base, extra=None):
    fn = f"{page}_{abs(hash(key)) % 10**8}_{var}_{kind}.npy"
    np.save(f"{OUT}/{fn}", a)
    e = dict(page=page, kind=kind, key=key, var=var, file=fn, origin=org, k=k, base=base, ink=ink_box(a, org))
    if extra: e.update(extra)
    DB.append(e); return e

def draw_vis(pg, entries, path):
    vis = Image.fromarray(cv2.cvtColor(pg.g, cv2.COLOR_GRAY2RGB)); dr = ImageDraw.Draw(vis)
    for e in entries:
        x0, y0, x1, y1 = e['ink']; dr.rectangle([x0, y0, x1, y1], outline=(220, 40, 40), width=2)
        dr.text((x0, y1 + 3), e['key'].replace('mark:', '') + str(e['var']) + ('*' if e.get('pos') == 'fina' else ''), fill=(20, 90, 220), font=FNT)
        if e.get('base') is not None: dr.line([x0 - 10, e['base'], x1 + 10, e['base']], fill=(0, 170, 0), width=2)
        if e.get('xline') is not None: dr.line([x0 - 10, e['xline'], x1 + 10, e['xline']], fill=(240, 150, 0), width=1)
    vis.save(path)

# cách ước lượng đường chân chữ trước khi bắt vào dòng kẻ
DESC_TOP = set('gpqyj') | {'p@fina', 'y@fina'}        # có đuôi dưới: lấy đỉnh thân + nửa ô lớn
DESC_ALL = set('gpqyj') | {'gh', 'gi', 'ngh', 'qu', 'ng', 'ph', 'p@fina', 'y@fina', 'ng@fina'}
RIGHT_STRIP = {'gh', 'gi', 'ngh', 'qu'}                 # chữ cuối (h, i, u) ngồi trên dòng kẻ
BAND = {'ng', 'ng@fina', 'ph'}                          # dải cao nửa ô lớn chứa nhiều mực nhất là dải chữ thường
def estimate_base(key, a, org, P):
    m = a > 0.5; H_, W_ = m.shape; ib = ink_box(a, org)
    if key == 'f': return ib[1] + 1.25 * P
    if key in DESC_TOP: return ib[1] + 0.4 * P
    if key in RIGHT_STRIP:
        ys = np.nonzero(m[:, int(W_ * 0.78):].any(1))[0]; return org[1] + ys.max()
    if key in BAND:
        rows = m.sum(1).astype(float); hb = int(0.4 * P)
        cs = np.concatenate([[0], np.cumsum(rows)])
        best = max(range(hb, H_ + 1), key=lambda y: cs[y] - cs[y - hb])
        return org[1] + best
    return ib[3]

def place_row(pg, items):
    """đặt đường chân chữ cho cả một dòng: bắt vào dòng kẻ đậm, rồi làm trơn theo x (dòng kẻ hơi cong)"""
    pts = []
    for t in items:
        ib = ink_box(t['a'], t['org']); cx = (ib[0] + ib[2]) / 2
        est = estimate_base(t['key'], t['a'], t['org'], pg.P)
        y, Pl = pg.baseline_at(cx, est)
        t.update(cx=cx, est=est, base=y, Pl=Pl); pts.append((cx, y))
    if len(pts) >= 3:
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        c = np.polyfit(xs, ys, 1); r = ys - np.polyval(c, xs)
        for t, rr in zip(items, r):
            if abs(rr) > 0.08 * pg.P:        # lệch hẳn so với cả dòng: dùng đường của dòng
                print(f'   ~ {pg.name} {t["key"]}{t["var"]}: chan chu lech {rr:.1f}px, dung duong cua dong')
                t['base'] = float(np.polyval(c, t['cx']))
    Prow = float(np.median([t['Pl'] for t in items]))
    for t in items:
        t['P'] = Prow
        if t['key'] not in DESC_ALL:      # chữ không có đuôi dưới: đáy nét chạm dòng kẻ như nhau (lún xuống 0,05 chiều cao chữ thường)
            t['base'] = float(ink_box(t['a'], t['org'])[3]) - 0.05 * 0.4 * Prow
        t['xline'] = t['base'] - 0.4 * Prow

def letters_page(pg, expect_rows, pos=None, box=None, drop=(), attach=True, keepX=False, exclude=None):
    lab, C = components(pg, box)
    if exclude: C = [c for c in C if not exclude(c)]
    rf, _ = cluster_rows(C, pg.P, 150 * pg.s * pg.s, len(expect_rows), 0.4)
    out = []
    for r, exp in enumerate(expect_rows):
        cs = [c for c in C if rf(c) == r]
        gl = group_row(cs, exp, pg.s, attach)
        if len(gl) != len(exp): print('  !!', pg.name, 'dong', r, 'tim thay', len(gl), 'can', len(exp), exp[0])
        cnt = {}; items = []
        for j, gr in enumerate(gl[:len(exp)]):
            ch = exp[j]
            if ch == 'X': continue                          # chữ gạch chéo: bỏ
            cnt[ch] = cnt.get(ch, 0) + 1
            a, org, dark = alpha_of(pg.g, lab, gr)
            p = pos[r] if isinstance(pos, list) else pos
            items.append(dict(key=ch, var=cnt[ch], row=r, a=a, org=org, dark=dark, pos=p, key2=ch + ('@' + p if p else '')))
        if items and items[0]['key'] not in '.,:;!?“”‘’()-—…+×0123456789':
            for t in items: t['key_est'] = t['key2']
            tmp = [dict(t, key=t['key2']) for t in items]
            place_row(pg, tmp)
            for t, u in zip(items, tmp): t.update(base=u['base'], P=u['P'], xline=u['xline'])
        out += items
    return out, lab

def commit(pg, items, kind='glyph', drop=()):
    ents = []
    for t in items:
        if (t['key'], t['var']) in drop: continue
        extra = dict(row=t['row'], dark=t['dark'], P=t['P'])
        if t.get('pos'): extra['pos'] = t['pos']
        e = add(pg.name, kind, t['key'], t['var'], t['a'], t['org'], None, t['base'], extra)
        e['xline'] = t.get('xline'); ents.append(e)
    return ents

def snap_row_base(pg, items, refkeys):
    """đường chân chữ chung cho một dòng số / dấu câu: bắt đáy các dấu ngồi trên dòng kẻ vào dòng kẻ đậm"""
    ref = [t for t in items if t['key'] in refkeys] or items
    ys = []
    for t in ref:
        ib = ink_box(t['a'], t['org']); y, Pl = pg.baseline_at((ib[0] + ib[2]) / 2, ib[3]); ys.append((y, Pl, (ib[0] + ib[2]) / 2))
    xs = np.array([q[2] for q in ys]); yy = np.array([q[0] for q in ys])
    c = np.polyfit(xs, yy, 1) if len(ys) >= 3 else np.array([0.0, float(np.median(yy))])
    Pl = float(np.median([q[1] for q in ys]))
    for t in items:
        ib = ink_box(t['a'], t['org']); t['base'] = float(np.polyval(c, (ib[0] + ib[2]) / 2)); t['P'] = Pl; t['xline'] = t['base'] - 0.4 * Pl

def punct_base(pg, items, dot_h):
    """dấu câu: đặt lại chân từng dấu. Chấm, hai chấm, ba chấm, chấm than, hỏi: đáy chấm nằm trên chân chữ.
       Phẩy, chấm phẩy: đầu dấu phẩy (cao bằng một chấm) nằm trên chân chữ. Gạch, cộng, nhân: giữa dải chữ thường."""
    for t in items:
        ch = t['key']; ib = ink_box(t['a'], t['org']); P = t['P']
        if ch in '.:…!?': t['base'] = float(ib[3])
        elif ch == ',': t['base'] = float(ib[1] + dot_h)
        elif ch == ';':
            m = t['a'] > 0.5; rows = np.nonzero(m.any(1))[0]
            gaps = np.nonzero(np.diff(rows) > 1)[0]
            ytop = rows[gaps[-1] + 1] if gaps.size else rows[0]
            t['base'] = float(t['org'][1] + ytop + dot_h)
        elif ch in '-—+×': t['base'] = float((ib[1] + ib[3]) / 2 + 0.18 * P)
        t['xline'] = t['base'] - 0.4 * P

def mixed_rows(pg, box, EXP, refs, nrows_big=None):
    """dòng số và dòng dấu câu: chia dòng theo các mảnh lớn, rồi gán mảnh nhỏ (chấm, phẩy) theo đường của dòng"""
    lab, C = components(pg, box)
    rf, rows = cluster_rows(C, pg.P, 60 * pg.s * pg.s, len(EXP), 0.4)
    out = []
    for r, exp in enumerate(EXP):
        gl = group_row([c for c in C if rf(c) == r], exp, pg.s, attach_small=False)
        if len(gl) != len(exp): print('  !!', pg.name, 'dong hon hop', r, len(gl), len(exp), exp)
        cnt = {}; items = []
        for j, gr in enumerate(gl[:len(exp)]):
            ch = exp[j]
            if ch == 'X': continue
            cnt[ch] = cnt.get(ch, 0) + 1
            a, org, dark = alpha_of(pg.g, lab, gr)
            items.append(dict(key=ch, var=cnt[ch], row=r, a=a, org=org, dark=dark, pos=None))
        snap_row_base(pg, items, refs[r]); out += items
    return out

def marks_page(pg):
    """trang dấu: dấu viết trên chữ a; mỗi dấu (phần trên, phần dưới) lưu cùng tâm và đỉnh chữ a nó được viết lên"""
    s = pg.s
    lab, C = components(pg)
    NR = [5, 5, 4, 5, 5, 5, 5]
    MK = [['acute', 'grave', 'hook', 'tilde', 'dotbelow']] * 2 + [['circumflex', 'breve', 'circumflex', 'breve']] + \
         [['circumflex_acute', 'circumflex_grave', 'circumflex_hook', 'circumflex_tilde', 'circumflex_dotbelow']] * 2 + \
         [['breve_acute', 'breve_grave', 'breve_hook', 'breve_tilde', 'breve_dotbelow']] * 2
    bodies = sorted([c for c in C if c['h'] > 0.3 * pg.P], key=lambda c: -c['a'])[:sum(NR)]
    rf, rows = cluster_rows(bodies, pg.P, 0, len(NR), 0.4)
    A = []
    for r, n_ in enumerate(NR):
        rb = sorted(rows[r], key=lambda c: c['x'])
        if len(rb) != n_: print('  !! dau dong', r, len(rb), [(int(c['cx']), int(c['cy']), c['a']) for c in rb])
        for j, b in enumerate(rb[:n_]): A.append(dict(r=r, j=j, body=b, marks=[], pieces=[b]))
    used = set(id(a['body']) for a in A)
    for c in C:
        if id(c) in used: continue
        cxr = (c['x'], c['x'] + c['w']); cyc = c['cy']
        host = None
        for a in A:
            b = a['body']
            ov = min(c['y'] + c['h'], b['y'] + b['h']) - max(c['y'], b['y'])
            hx = min(cxr[1], b['x'] + b['w'] + 6 * s) - max(cxr[0], b['x'] - 6 * s)
            if ov > 0.6 * c['h'] and hx > 0: host = a; break
        if host: host['pieces'].append(c); continue
        dot = c['a'] < 150 * s * s and c['w'] < 18 * s and c['h'] < 20 * s
        best, bs = None, 1e9
        for a in A:
            b = a['body']; bx0, bx1 = b['x'] - 22 * s, b['x'] + b['w'] + 22 * s
            if min(cxr[1], bx1) - max(cxr[0], bx0) <= 0: continue
            if dot and cyc > b['y'] + b['h'] - 4 * s: gap = cyc - (b['y'] + b['h'])
            elif not dot or cyc < b['y']:
                gap = b['y'] - (c['y'] + c['h'] * 0.5)
                if gap < -8 * s: continue
            else: continue
            if gap < bs: bs, best = gap, a
        if best is not None and bs < 150 * s: best['marks'].append(c)
    cnt = {}; ents = []
    for a in A:
        key = MK[a['r']][a['j']]; cnt[key] = cnt.get(key, 0) + 1; v = cnt[key]
        b = a['body']; bcx = b['x'] + b['w'] / 2.0
        base, Pl = pg.baseline_at(bcx, b['y'] + b['h'])
        mid = b['y'] + b['h'] / 2.0
        up = [c for c in a['marks'] if c['cy'] < mid]; dn = [c for c in a['marks'] if c['cy'] >= mid]
        for part, comps in (('above', up), ('below', dn)):
            if not comps: print('  !! thieu dau', key, v, part) if (part == 'above' and key != 'dotbelow') or (part == 'below' and 'dotbelow' in key) else None; continue
            al, org, dark = alpha_of(pg.g, lab, comps)
            ents.append(add(pg.name, 'mark', key + ':' + part, v, al, org, None, base, dict(part=part, P=Pl)))
        al, org, dark = alpha_of(pg.g, lab, a['pieces'])
        ents.append(add(pg.name, 'abody', 'a', len([e for e in DB if e['kind'] == 'abody']) + 1, al, org, None, base, dict(mark=key, var_m=v, P=Pl)))
    return ents

if __name__ == '__main__':
    pages = {}
    def P_(name, f, box=None):
        pages[name] = Page(name, f, box); print(name, 'o lon', pages[name].P); return pages[name]
    def run(pg, rows, pos=None, box=None, tag=None, exclude=None, drop=()):
        it, _ = letters_page(pg, rows, pos=pos, box=box, exclude=exclude); e = commit(pg, it, drop=drop); draw_vis(pg, e, OUT + '/vis-%s.png' % (tag or pg.name)); return e
    # trang 1, 2, 3: chữ thường
    p1 = P_('n1', SRC + 'p-000.png'); run(p1, [list(c * 5) for c in 'abcdđeghi'])
    p2 = P_('n2', SRC + 'p-001.png'); run(p2, [list(c * 5) for c in 'klmno'] + [list(c * 6) for c in 'pqrs'])
    p3 = P_('n3', SRC + 'p-002.png'); H3, W3 = p3.g.shape
    e3 = run(p3, [list('tttttt'), list('uuuuu'), list('ưưưưư'), list('vvvvvv'), list('xxxxx'), list('yyyyy')], box=(0, 0, W3, int(0.655 * H3)))
    # số và dấu câu (dấu hai chấm ở đây quá mờ: bỏ, dùng bản ở trang bổ sung)
    EXP = [list('012345'), list('6789'), ['.', '.', '.', ',', ',', ',', 'X', ';', 'X', '!', '?', '“', '”'], ['‘', '’', '(', ')', '-', '—', '…', '+', '×']]
    REFS = [set('012345'), set('6789'), set('.!?'), set('…')]
    it = mixed_rows(p3, (0, int(0.655 * H3), W3, H3), EXP, REFS)
    dot_h = float(np.median([ink_box(t['a'], t['org'])[3] - ink_box(t['a'], t['org'])[1] for t in it if t['key'] == '.']))
    punct_base(p3, [t for t in it if t['row'] >= 2], dot_h)
    e3b = commit(p3, it); draw_vis(p3, e3 + e3b, OUT + '/vis-n3.png')
    # trang dấu
    p4 = P_('n4', SRC + 'p-003.png'); em = marks_page(p4); draw_vis(p4, em, OUT + '/vis-n4.png')
    # trang âm ghép
    p5 = P_('n5', SRC + 'p-004.png'); H5, W5 = p5.g.shape
    run(p5, [['ch'] * 5, ['gh'] * 5, ['gi'] * 5, ['X', 'X', 'X', 'kh', 'kh', 'kh'], ['ng'] * 5, ['ngh'] * 5, ['nh'] * 5, ['ph'] * 5, ['th'] * 5], box=(0, 60, W5, H5), drop={('gh', 1)})     # gh thứ nhất mất nét vòng dưới chỗ lề đỏ
    # trang 6: tr, qu ở trên gạch ngang; dưới là dạng cuối từ
    p6 = P_('n6', SRC + 'p-005.png'); H6, W6 = p6.g.shape
    ea = run(p6, [['tr'] * 5, ['qu'] * 5], box=(0, 0, W6, 615), tag='n6a')
    eb = run(p6, [['c'] * 5, ['ch'] * 4, ['m'] * 5, ['n'] * 6, ['p'] * 5, ['t'] * 5], pos='fina', box=(0, 800, W6, H6), tag='n6b')
    draw_vis(p6, ea + eb, OUT + '/vis-n6.png')
    # trang 7: tiếp dạng cuối từ; hàng ng đầu bỏ, chỉ lấy hàng dưới
    p7 = P_('n7', SRC + 'p-006.png')
    run(p7, [['X'] * 5, ['ng'] * 5, ['nh'] * 5, ['i'] * 5, ['y'] * 6, ['u'] * 6, ['o'] * 6], pos='fina')
    # trang bổ sung
    pb = P_('nb', SRCB + 'b-000.png'); Hb, Wb = pb.g.shape
    lab, Cb = components(pb)
    colon = [c for c in Cb if 0.64 * Hb < c['cy'] < 0.70 * Hb and c['a'] < 400]
    eb1 = run(pb, [['ơ'] * 5, ['X'] * 6, ['f'] * 5, ['j'] * 5, ['w'] * 5, ['z'] * 5, ['ơ', 'ơ', 'X', 'ơ', 'ơ', 'ơ', 'ơ'], ['X', 'ư', 'ư', 'ư', 'ư', 'ư']],
              pos=[None] * 6 + ['fina', 'fina'], box=None, exclude=lambda c: 0.64 * Hb < c['cy'] < 0.70 * Hb and c['a'] < 400)
    # dấu hai chấm: 3 cặp chấm
    gl = group_row(colon, [':'] * 3, pb.s, attach_small=False)
    its = []
    for j, gr in enumerate(gl[:3]):
        a, org, dark = alpha_of(pb.g, lab, gr); its.append(dict(key=':', var=j + 1, row=99, a=a, org=org, dark=dark, pos=None))
    snap_row_base(pb, its, {':'}); punct_base(pb, its, 0); ec = commit(pb, its); draw_vis(pb, eb1 + ec, OUT + '/vis-nb.png')
    # chữ hoa (27/10): chữ gạch bằng mực cam là bỏ
    SRCH = HERE + '/../font/src2c/'
    CAPS_DESC = set('GJY')
    def caps_page(name, path, rows, box_frac=(0, 1), hi=145):
        im = cv2.imread(path); B_, G_, R_ = [im[:, :, i].astype(int) for i in range(3)]
        orange = ((R_ - B_ > 100) & (G_ - B_ > 45) & (R_ > 170)).astype(np.uint8)
        orange = cv2.dilate(orange, np.ones((25, 25), np.uint8))
        # ảnh chụp (không qua máy quét): làm phẳng độ sáng nền trước, để nền đồng đều như các trang quét
        bg = cv2.morphologyEx(im, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
        bg = cv2.GaussianBlur(bg, (0, 0), 15).astype(np.float32)
        im = np.clip(im.astype(np.float32) / np.maximum(bg, 1) * 250, 0, 255).astype(np.uint8)
        im2 = im.copy(); im2[orange > 0] = 255          # xóa hẳn chỗ gạch cam (cả mực bên dưới) trước khi tách chữ
        p2 = OUT + '/' + name + '-khong-cam.png'; cv2.imwrite(p2, im2)
        pg = P_(name, p2)
        n, olab, ost, _ = cv2.connectedComponentsWithStats(orange, 8)
        oboxes = [ost[i] for i in range(1, n) if ost[i][4] > 2000]
        H_, W_ = pg.g.shape
        lab, C = components(pg, (0, int(box_frac[0] * H_), W_, int(box_frac[1] * H_)))
        def crossed(c):
            if orange[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']][lab[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] == c['i']].any(): return True
            m = 0.3 * pg.P
            for x, y, w, h, a in oboxes:
                if x <= c['cx'] <= x + w and y <= c['cy'] <= y + h: return True
                ix = max(0, min(c['x'] + c['w'], x + w + m) - max(c['x'], x - m)); iy = max(0, min(c['y'] + c['h'], y + h + m) - max(c['y'], y - m))
                if ix * iy > 0.6 * c['w'] * c['h']: return True        # mảnh còn sót của chữ bị gạch, nằm sát chỗ gạch cam
            return False
        C = [c for c in C if not crossed(c)]
        # chữ hoa không có chấm hay dấu rời: mảnh nhỏ không chạm khung của chữ lớn nào là mảnh vụn của chữ bị gạch
        amed = float(np.median(sorted([c['a'] for c in C], reverse=True)[:sum(len(r) for r in rows)]))
        bigs = [c for c in C if c['a'] >= 0.25 * amed]
        def inside(c):
            return any(c['x'] >= b['x'] - 4 and c['x'] + c['w'] <= b['x'] + b['w'] + 4 and c['y'] >= b['y'] - 4 and c['y'] + c['h'] <= b['y'] + b['h'] + 4 for b in bigs if b is not c)
        C = [c for c in C if c['a'] >= 0.25 * amed or inside(c)]
        rf, _ = cluster_rows(C, pg.P, 150 * pg.s * pg.s, len(rows), 0.45)
        items = []
        for r, exp in enumerate(rows):
            gl = group_row([c for c in C if rf(c) == r], exp, pg.s, True)
            if len(gl) != len(exp): print('  !!', name, 'dong', r, 'tim thay', len(gl), 'can', len(exp), exp)
            for j, gr in enumerate(gl[:len(exp)]):
                a, org, dark = alpha_of(pg.g, lab, gr, hi=hi)     # ảnh chụp đã làm phẳng: mực rất đậm, vệt dòng kẻ còn sót nhạt hơn hẳn
                items.append(dict(key=exp[j], var=1, row=r, a=a, org=org, dark=dark, pos=None))
        return pg, items
    def drop_grid(a, P):
        """bỏ vệt dòng kẻ thẳng, mảnh còn dính vào chữ (ảnh chụp, dòng kẻ đậm hơn trang quét); nét chữ dày hơn nên giữ lại"""
        m = (a > 0.3).astype(np.uint8); L = max(15, int(0.4 * P))
        er = cv2.erode(m, np.ones((3, 3), np.uint8))
        out = a.copy()
        for k in ((L, 1), (1, L)):
            ker = np.ones(k, np.uint8)
            line = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
            stroke = cv2.dilate(cv2.morphologyEx(er, cv2.MORPH_OPEN, ker), np.ones((5, 5), np.uint8))
            rm = (line > 0) & (stroke == 0)
            out[cv2.dilate(rm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 0
        return out
    def caps_base(pg, items, capH):
        for t in items:
            ib = ink_box(t['a'], t['org']); cx = (ib[0] + ib[2]) / 2
            # đáy chữ: bỏ qua vệt dòng kẻ mảnh còn dính vào khung chữ (làm T, H bị đặt cao lên)
            er = cv2.erode((t['a'] > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))
            rows = np.nonzero(er.any(1))[0]; bot = float(t['org'][1] + rows.max() + 1)
            if t['key'] in CAPS_DESC:
                y, Pl = pg.baseline_at(cx, ib[1] + capH * pg.P)
            elif t['key'] == 'T':                            # đuôi lượn góc dưới trái chỉ để trang trí: thân chữ T ngồi trên dòng kẻ (Nguyên dặn, 27/10)
                y, Pl = pg.baseline_at(cx, bot)
            else:
                y, Pl = pg.baseline_at(cx, bot)
                y = bot - 0.05 * 0.4 * Pl          # như chữ thường: đáy nét chạm dòng như nhau
            t['base'] = y; t['P'] = Pl; t['xline'] = y - 0.4 * Pl
    ph1, it1 = caps_page('nh1', SRCH + 'h-000.png', [['A', 'B'], ['C'], ['D'], ['Đ', 'E', 'F'], ['G'], ['H', 'I'], list('KLMNO'), list('PQRST')], (0.14, 0.92))
    ph2, it2 = caps_page('nh2', SRCH + 'h-001.png', [list('UVWX'), ['Y', 'Z', 'J']], (0, 0.4), hi=200)   # trang này mực nhạt hơn: cắt ở 145 thì nét U V W X Y Z J bị đứt (26/10)
    hs = []
    for pg, its in ((ph1, it1), (ph2, it2)):
        for t in its:
            if t['key'] not in CAPS_DESC:
                ib = ink_box(t['a'], t['org']); hs.append((ib[3] - ib[1]) / pg.P)
    capH = float(np.median(hs)); print('chu hoa cao', round(capH, 2), 'o lon')
    for pg, its in ((ph1, it1), (ph2, it2)):
        caps_base(pg, its, capH); e = commit(pg, its); draw_vis(pg, e, OUT + '/vis-%s.png' % pg.name)
    # Ơ, Ư hoa viết lại (26/10, ảnh chụp một trang vở)
    ph3, it3 = caps_page('nh3', HERE + '/../font/src2e/ou-000.png', [['Ơ'], ['Ư']], (0.15, 0.62))
    caps_base(ph3, it3, capH); e = commit(ph3, it3); draw_vis(ph3, e, OUT + '/vis-nh3.png')
    # chữ in hoa (27/10): dùng cho chữ viết tắt toàn chữ hoa như "LGBTQIA" trong bài "một đứa trẻ"; chữ G nhỏ phía trên chữ G là bỏ
    gp = cv2.imread(HERE + '/../font/src2d/in-hoa.jpg', 0)
    nP, labP, stP, cenP = cv2.connectedComponentsWithStats((gp < 150).astype(np.uint8), 8)
    CP = [dict(i=i, x=int(stP[i][0]), y=int(stP[i][1]), w=int(stP[i][2]), h=int(stP[i][3]), a=int(stP[i][4]), cx=float(cenP[i][0]), cy=float(cenP[i][1]))
          for i in range(1, nP) if stP[i][4] > 3 and stP[i][1] > 35]
    ROWSP = [list('ABCDEFGHIJ'), list('KLMNOPQRSTUVW'), list('XYZ')]
    cutsP = [80, 150]
    printed = []
    for r, exp in enumerate(ROWSP):
        cs = [c for c in CP if sum(c['cy'] > t for t in cutsP) == r]
        gl = group_row(cs, exp, 0.3, True)
        if len(gl) != len(exp): print('  !! chu in dong', r, len(gl), len(exp))
        for j, gr in enumerate(gl[:len(exp)]):
            a, org, dark = alpha_of(gp, labP, gr, pad=4, hi=215)
            printed.append(dict(key=exp[j], row=r, a=a, org=org, ib=ink_box(a, org)))
    capP = float(np.median([t['ib'][3] - t['ib'][1] for t in printed if t['key'] not in 'QJY']))
    kP = 2.4 * XH1 / capP                 # chữ in hoa cao 2,4 lần chữ thường
    for r in range(len(ROWSP)):
        row = [t for t in printed if t['row'] == r]
        bots = [t['ib'][3] for t in row if t['key'] not in 'QJY']
        base = float(np.median(bots))
        for t in row:
            e = add('pin', 'glyph', t['key'], 1, t['a'], t['org'], kP, base, dict(row=r, pos='print', dir='print', P=1.0))
    print('chu in hoa:', len(printed), 'cao', capP, 'px, ti le', round(kP, 2))
    # cỡ chữ: chiều cao chữ thường đo theo ô lớn, quy về XH1 px
    def frac(keys):
        v = []
        for e in DB:
            if e['kind'] == 'glyph' and e.get('dir') != 'old' and e['key'] in keys and not e.get('pos'):
                v.append((e['base'] - e['ink'][1]) / e['P'])
        return float(np.median(v))
    xh = frac(set('acemnorsuvxw')); asc = frac(set('bdhkl'))
    KP = XH1 / xh
    for e in DB:
        if e.get('dir') not in ('old', 'print'): e['k'] = KP / e['P']
    print('chu thuong cao', round(xh, 3), 'o lon; vong tren', round(asc, 3), 'o lon =', round(asc / xh, 2), 'lan chu thuong; ti le', round(KP, 1), 'px/o lon')
    json.dump(dict(db=DB, xh1=XH1, asc_ratio=asc / xh, KP=KP), open(OUT + '/db.json', 'w'), ensure_ascii=False, indent=0)
    from collections import Counter
    print(sorted(Counter((e['key'] + ('@' + e['pos'] if e.get('pos') else '')) for e in DB if e['kind'] == 'glyph').items()))
