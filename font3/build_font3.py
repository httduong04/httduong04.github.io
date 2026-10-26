# Dựng font từ bản viết trên vở ô ly (26/10): giữ đúng tỉ lệ như trên vở, ghép dấu tiếng Việt, bản thay thế cv01–cv11, dạng cuối từ cv20–cv31
import sys, json, os, unicodedata, numpy as np, cv2, potrace
from skimage.morphology import skeletonize
from skimage.filters import apply_hysteresis_threshold
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.transformPen import TransformPen
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
HERE = os.path.dirname(os.path.abspath(__file__)); WD = HERE + '/work'; OLDWD = HERE + '/../font/work2'
OUT = sys.argv[1] if len(sys.argv) > 1 else WD + '/HPTTay.ttf'
P = dict(overlap=9.0, gap_xh=1.1, side=0.04)
D = json.load(open(WD + '/db.json')); DB = D['db']; XH = D['xh1']
UPM = 1000; S = 0.45 * UPM / XH          # chiều cao chữ thường: thân chữ đo được khoảng 0,43 em, bằng cỡ nhìn của bản trước
UP = 4                                   # độ phân giải khi dò nét: 4 điểm cho mỗi px của trang 1
NODESC = set('abcdđefhiklmnorstuvwxzơư') | set('ABCDĐEFHIKLMNOPQRSTUVWXZƠƯ') | set('0123456789') | {'ch', 'kh', 'nh', 'th', 'tr'}
DIGRAPHS = ['ngh', 'ch', 'gh', 'gi', 'kh', 'ng', 'nh', 'ph', 'qu', 'th', 'tr']   # âm ghép viết liền

def sw_of(b):
    """độ dày nét (px của ảnh b): gấp đôi khoảng cách trung bình từ xương nét ra mép"""
    if b.sum() < 20: return None
    dt = cv2.distanceTransform(b.astype(np.uint8), cv2.DIST_L2, 3); sk = skeletonize(b)
    v = dt[sk]; return float(2 * np.median(v)) if v.size else None

ASC = D['asc_ratio']   # vòng trên cao bao nhiêu lần chữ thường, đo trên vở (không nén nữa)
def keff(e):
    if e.get('dir') == 'old':      # chữ hoa bản cũ: cao bằng vòng trên của chữ thường mới
        return e['k'] * ASC / (0.9 * e['old_asc1'] / XH)
    return e['k']
def prepared(e):
    """ảnh nhị phân của chữ ở độ phân giải chung (px chuẩn × UP) và gốc tọa độ tương ứng"""
    a = np.load((OLDWD if e.get('dir') == 'old' else WD) + '/' + e['file']); k = keff(e) * UP
    big = cv2.resize(a, (max(1, int(round(a.shape[1] * k))), max(1, int(round(a.shape[0] * k)))), interpolation=cv2.INTER_CUBIC if k >= 1 else cv2.INTER_AREA)
    if e.get('dir') == 'print':      # ảnh chữ in hoa độ phân giải thấp (chữ cao ~25 px): làm mượt mép trước khi dò nét
        big = cv2.GaussianBlur(big, (0, 0), 0.55 * k)
    big = cv2.GaussianBlur(big, (0, 0), UP * 0.8)
    pad = 8 * UP; big = np.pad(big, pad)
    # ngưỡng theo độ đậm tại chỗ: nét mảnh, nhạt (vòng dưới của g viết nhanh) mờ đi khi thu nhỏ vẫn được giữ,
    # còn nét đậm thì vẫn cắt ở nửa độ đậm như cũ nên không dày thêm
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7 * UP, 7 * UP))
    lm = cv2.dilate(big.astype(np.float32), ker)
    keep = big > np.maximum(0.14, 0.5 * np.minimum(lm, 1.0))
    return apply_hysteresis_threshold(np.where(keep, big, 0), 0.14, 0.5), pad

# độ dày nét chuẩn: chữ thường trang 1 (bỏ dòng đầu viết nhạt)
sws = []
for e in DB:
    if e['page'] in ('n1', 'n2') and e['kind'] == 'glyph':
        b, _ = prepared(e); s = sw_of(b)
        if s: sws.append(s)
SW0 = float(np.median(sws))
THICK = 1.35                              # Nguyên muốn nét dày hơn bản vở một chút (27/10)
SW = SW0 * THICK; DOTX = (SW - SW0) / 2   # chấm i, j nở thêm cùng mức
print('do day net tren vo', round(SW0 / UP, 2), 'px chuan; dung', round(SW / UP, 2))

def normalize(b):
    """làm nét dày bằng nét chuẩn: nở ra hoặc gọt bớt theo khoảng cách tới mép"""
    s = sw_of(b)
    if not s: return b
    d = max(-1.5 * UP, min(2.5 * UP, (SW - s) / 2))
    if abs(d) < 0.25 * UP: return b
    if d > 0: return cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5) <= d
    return cv2.distanceTransform(b.astype(np.uint8), cv2.DIST_L2, 5) > -d

def frame(e, b, pad):
    """hàm đổi tọa độ ảnh (đã phóng) sang đơn vị font, gốc đặt ở (mép trái thân chữ, đường chân chữ)"""
    ox, oy = e['origin']; k = e['k']
    def px1(x, y): return ((x - pad) / UP + ox * k, (y - pad) / UP + oy * k)   # px trang 1 tuyệt đối (đã nhân k)
    base1 = e['base'] * k
    return px1, base1

def trace(b):
    return potrace.Bitmap(~b).trace(turdsize=6 * UP * UP // 4, alphamax=1.1, opticurve=True, opttolerance=0.5)

def body_cols(b, pad, e, base_img_y, lo=1.15, hi=0.12):
    """cột có mực trong dải chữ thường (để tính bề rộng): bỏ phần vòng trên, vòng dưới"""
    xh = XH * UP; top = int(base_img_y - lo * xh); bot = int(base_img_y + hi * xh)
    band = b[max(0, top):max(1, bot)]
    cols = np.nonzero(band.any(0))[0]
    if cols.size == 0: cols = np.nonzero(b.any(0))[0]
    return int(cols.min()), int(cols.max())

def anchor_of(b, base_img_y, horn=False, cap=False):
    """tâm ngang (trọng tâm mực ở giữa thân chữ) và đỉnh thân chữ: chỗ đặt dấu.
       Đo cùng một cách cho chữ a ở trang dấu và cho mọi chữ khác, nên lệch do nét lên đầu chữ tự triệt tiêu."""
    if cap:     # chữ hoa có nét lượn dài (như A): đặt dấu theo phần đỉnh chữ, không theo trọng tâm cả chữ
        rows = np.nonzero(b[:int(base_img_y)].any(1))[0]
        ytop = int(rows.min()); band = b[ytop:ytop + max(2, int(0.3 * (base_img_y - ytop)))]
        ys, xs = np.nonzero(band)
        return float(xs.mean()), float(ytop)
    xh = XH * UP * (2.6 if cap else 1.0)
    lo, hi = (0.7, 0.15) if horn else (0.95, 0.25)
    top = int(base_img_y - lo * xh); bot = int(base_img_y - hi * xh)
    band = b[max(0, top):max(1, bot)]
    if band.sum() == 0: band = b; top = 0
    ys, xs = np.nonzero(band)
    cx = float(xs.mean())
    w = xs.max() - xs.min()
    # đỉnh: điểm cao nhất của mực quanh tâm (±30% bề rộng), trong vùng không quá cao (bỏ vòng trên, râu)
    t2 = int(base_img_y - (1.25 if not cap else 1.1) * xh)
    win = b[max(0, t2):int(base_img_y), max(0, int(cx - 0.3 * w)):int(cx + 0.3 * w) + 1]
    rows = np.nonzero(win.any(1))[0]
    topy = float(rows.min() + max(0, t2)) if rows.size else float(top)
    if horn: topy = max(topy, base_img_y - 1.0 * xh)
    return cx, topy

glyphs = {}   # tên -> dict(contours=[...], adv, comps)
def to_units(pt, x0, base_img_y):
    return ((pt.x - x0) / UP * S, (base_img_y - pt.y) / UP * S)

def make_simple(name, b, x0, base_img_y, adv, lsb_shift):
    pl = trace(b)
    pen = TTGlyphPen(None); qp = Cu2QuPen(pen, max_err=1.0, reverse_direction=True)
    tp = TransformPen(qp, (1, 0, 0, 1, lsb_shift, 0))
    for c in pl:
        tp.moveTo(to_units(c.start_point, x0, base_img_y))
        for s in c.segments:
            if s.is_corner: tp.lineTo(to_units(s.c, x0, base_img_y)); tp.lineTo(to_units(s.end_point, x0, base_img_y))
            else: tp.curveTo(to_units(s.c1, x0, base_img_y), to_units(s.c2, x0, base_img_y), to_units(s.end_point, x0, base_img_y))
        tp.closePath()
    glyphs[name] = dict(glyph=pen.glyph(), adv=int(round(adv)))

def gid(ch, i=1, j=1):
    pos = ''
    if '@' in ch: ch, pos = ch.split('@'); pos = '.' + pos       # dạng đầu từ (init) hoặc cuối từ (fina)
    return ('u%04X' % ord(ch) if len(ch) == 1 else 'lig_' + ch) + '%s.%d%d' % (pos, i, j)

info = {}     # chữ -> số đo (x thân, đỉnh, bề rộng) theo đơn vị font, cho từng bản
COUNT = {}
def add_glyph(e, ch=None, drop_dot=False, scale=1.0, kind='letter'):
    ch = ch or e['key']
    b0, pos = (ch.split('@') + [None])[:2]        # chữ gốc và vị trí trong từ (nếu là dạng đầu từ / cuối từ)
    b, pad = prepared(e) if scale == 1.0 else prepared_scaled(e, scale)
    ox, oy = e['origin']; k = keff(e) * (scale if scale != 1.0 else 1)
    base_img_y = (e['base'] - oy) * k * UP + pad
    dots = None
    if b0 in ('i', 'j', 'ı', 'gi'):   # chấm của i, j: tách riêng, không làm dày, nhấc lên một chút cho khỏi dính vào thân khi chữ nhỏ
        n, lab, st, cen = cv2.connectedComponentsWithStats(b.astype(np.uint8), 8)
        dots = np.zeros_like(b)
        for q in range(1, n):
            if st[q, 1] + st[q, 3] < base_img_y - 0.85 * XH * UP and st[q, 4] < (XH * UP) ** 2 * 0.15:
                dots |= (lab == q); b[lab == q] = False
        lift = int(0.2 * XH * UP); dots = np.roll(dots, -lift, axis=0)
    b = normalize(b)
    if dots is not None and not drop_dot:
        if dots.any() and DOTX > 0: dots = cv2.distanceTransform((~dots).astype(np.uint8), cv2.DIST_L2, 5) <= DOTX
        b = b | dots
    if kind == 'punct':
        cols = np.nonzero(b.any(0))[0]; l, r = int(cols.min()), int(cols.max()); side = 0.05 * UPM
    elif kind == 'digit':
        cols = np.nonzero(b.any(0))[0]; l, r = int(cols.min()), int(cols.max()); side = 0.05 * UPM
    elif kind == 'print':     # chữ in hoa: đứng rời, không nối
        cols = np.nonzero(b.any(0))[0]; l, r = int(cols.min()), int(cols.max()); side = 0.07 * UPM
    else:
        if len(b0) > 1 or b0 in 'bdđhkltfgjpqyGJY':      # vòng trên, vòng dưới thò sang chữ bên cạnh: chỉ tính bề rộng phần thân
            l, r = body_cols(b, pad, e, base_img_y, hi=0.12 if b0 in NODESC else -0.08)
        else:
            cols = np.nonzero(b.any(0))[0]; l, r = int(cols.min()), int(cols.max())
            if b0.isupper():      # chữ hoa: mép phải tính theo phần dưới chữ, nét lượn trên đầu (H, T, I...) được chìa sang chữ sau
                rows = np.nonzero(b.any(1))[0]; ytop = rows.min()
                low = b[int(base_img_y - 0.6 * (base_img_y - ytop)):int(base_img_y + 0.15 * XH * UP)]
                cl = np.nonzero(low.any(0))[0]
                if cl.size: r = int(cl.max())
        # nét nối: đoạn mảnh ở hai đầu chữ (nét lên đầu chữ, nét ra cuối chữ). Chỉ cho chữ bên cạnh chồng lên đoạn đó,
        # để chữ không đè lên phần thân (như vòng của s, chữ o không có nét lên)
        band = b[max(0, int(base_img_y - 1.3 * XH * UP)):int(base_img_y + 0.2 * XH * UP)]
        cnt = band.sum(0); thin = 1.7 * SW
        def conn(cs):
            n = 0
            for c in cs:
                if cnt[c] > thin: break
                n += 1
            return n / UP
        cl = conn(range(l, r + 1)); cr = conn(range(r, l - 1, -1))
        ovl, ovr = max(4.0, min(P['overlap'], 0.85 * cl)), max(4.0, min(P['overlap'], 0.85 * cr))   # ít nhất 4 px cho chữ chạm nhau
        if b0 in ('ơ', 'ư', 'Ơ', 'Ư'): ovr = 2.0             # râu của ơ, ư: chữ sau chỉ chạm nhẹ
        if b0 in ('ơ', 'Ơ') and pos != 'fina':               # râu ơ chìa sang phải: tính mép phải theo thân o, để râu chìa lên trên chữ sau
            bb = b[max(0, int(base_img_y - 0.7 * XH * UP)):int(base_img_y + 0.1 * XH * UP)]
            cb = np.nonzero(bb.any(0))[0]
            if cb.size: r = int(cb.max()); ovr = 3.0
        sideL, sideR = P['side'] * UPM - ovl * S, P['side'] * UPM - ovr * S
        allc = np.nonzero(b.any(0))[0]
        if pos == 'init':        # đầu từ: không có chữ đứng trước, giữ trọn nét lên đầu chữ
            l = int(allc.min()); sideL = 0.02 * UPM
        if pos == 'fina':        # cuối từ: đuôi kéo dài, không để chữ sau (từ sau) đè lên
            r = int(allc.max()); sideR = 0.03 * UPM
    if kind != 'letter': sideL = sideR = side
    w = (r - l) / UP * S; adv = w + sideL + sideR; side = sideL
    COUNT[ch] = COUNT.get(ch, 0) + 1; v = COUNT[ch]
    name = gid(ch, v, 1)
    make_simple(name, b, l, base_img_y, adv, side)
    cx, topy = anchor_of(b, base_img_y, horn=b0 in ('ơ', 'ư', 'Ơ', 'Ư'), cap=b0.isupper())
    info.setdefault(ch, []).append(dict(name=name, cx=(cx - l) / UP * S + side, top=(base_img_y - topy) / UP * S, adv=adv))
    return name

def prepared_scaled(e, s):
    e2 = dict(e); e2['k'] = e['k'] * s; return prepared(e2)

# ---------- chữ cái, số, dấu câu ----------
order = sorted([e for e in DB if e['kind'] == 'glyph'], key=lambda e: (e['key'], e['page'], e['var']))
A_TARGET = float(np.median([(e['base'] - e['ink'][1]) / (0.4 * e['P']) for e in order if e['key'] == 'o' and not e.get('pos') and e.get('dir') != 'old']))
print('chieu cao o (phan 2 o li):', round(A_TARGET, 3))
for e in order:
    ch = e['key']
    kind = 'digit' if ch.isdigit() else ('punct' if not ch.isalpha() else 'letter')
    if e.get('pos') == 'print': kind = 'print'
    sc = 1.0
    if ch == 'a' and not e.get('pos'):     # chữ a viết hơi nhỏ hơn o, u: phóng cho thân a cao bằng o (Nguyên dặn, 27/10)
        h = (e['base'] - e['ink'][1]) / (0.4 * e['P'])
        sc = float(np.clip(A_TARGET / h, 1.0, 1.3))
    add_glyph(e, ch=ch + ('@' + e['pos'] if e.get('pos') else ''), kind=kind, scale=sc)
    if ch == 'i': e2 = dict(e); add_glyph(e2, ch='ı' + ('@' + e['pos'] if e.get('pos') else ''), drop_dot=True)

# ---------- dấu (mỗi phần trên/dưới là một glyph riêng, gốc tọa độ ở tâm và đỉnh chữ a nó được viết lên) ----------
abody = {e['mark']: [] for e in DB if e['kind'] == 'abody'}
for e in DB:
    if e['kind'] == 'abody': abody[e['mark']].append(e)
marks = {}
for e in DB:
    if e['kind'] != 'mark': continue
    key, part = e['key'].split(':')
    host = abody[key][e['var'] - 1]
    hb, hpad = prepared(host); hbase = (host['base'] - host['origin'][1]) * host['k'] * UP + hpad
    hcx, htop = anchor_of(hb, hbase)
    # tâm và đỉnh chữ a theo px trang 1 tuyệt đối
    acx = (hcx - hpad) / UP + host['origin'][0] * host['k']; atop = (htop - hpad) / UP + host['origin'][1] * host['k']
    abase = host['base'] * host['k']
    b, pad = prepared(e); b = normalize(b)
    ox, oy = e['origin'][0] * e['k'], e['origin'][1] * e['k']
    refy = atop if part == 'above' else abase
    pl = trace(b)
    def U(pt): return (((pt.x - pad) / UP + ox - acx) * S, (refy - ((pt.y - pad) / UP + oy)) * S)
    name = 'mk.%s.%s.%d' % (key, part, e['var'])
    pen = TTGlyphPen(None); qp = Cu2QuPen(pen, max_err=1.0, reverse_direction=True)
    for c in pl:
        qp.moveTo(U(c.start_point))
        for s in c.segments:
            if s.is_corner: qp.lineTo(U(s.c)); qp.lineTo(U(s.end_point))
            else: qp.curveTo(U(s.c1), U(s.c2), U(s.end_point))
        qp.closePath()
    glyphs[name] = dict(glyph=pen.glyph(), adv=0)
    marks.setdefault(key, {}).setdefault(e['var'], {})[part] = name
# đỉnh chữ a ở trang dấu, để dấu trên chữ khác nằm cùng khoảng cách
# ---------- chữ có dấu: ghép chữ gốc + dấu (tham chiếu, không chép nét) ----------
TONES = {'́': 'acute', '̀': 'grave', '̉': 'hook', '̃': 'tilde', '̣': 'dotbelow'}
def plan(ch):
    d = unicodedata.normalize('NFD', ch)
    base = d[0]; rest = d[1:]
    if base == 'i' and rest: base = 'ı'
    hat = None; tone = None
    if '̂' in rest: hat = 'circumflex'
    if '̆' in rest: hat = 'breve'
    if '̛' in rest: base = {'o': 'ơ', 'u': 'ư', 'O': 'Ơ', 'U': 'Ư'}[base]
    for c in rest:
        if c in TONES: tone = TONES[c]
    if hat and tone: key = hat + '_' + tone
    else: key = hat or tone
    return base, key
VN = 'aăâeêioôơuưyAĂÂEÊIOÔƠUƯY'
chars = set()
for v in VN:
    for t in ['', '́', '̀', '̉', '̃', '̣']:
        c = unicodedata.normalize('NFC', unicodedata.normalize('NFD', v) + t)
        if len(c) == 1 and c not in info: chars.add(c)
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphComponent
def composite(parts):
    g = Glyph(); g.numberOfContours = -1; g.components = []
    for nm, dx, dy in parts:
        c = GlyphComponent(); c.glyphName = nm; c.x = int(round(dx)); c.y = int(round(dy)); c.flags = 0x4; g.components.append(c)
    return g
NV = {}
for ch in sorted(chars):
    base, key = plan(ch)
    if key is None or base not in info: print('bo qua', ch); continue
    bases = info[base]; mk = marks[key]
    NV[ch] = (len(bases), len(mk))
    for i, bi in enumerate(bases, 1):
        for j in sorted(mk):
            parts = [(bi['name'], 0, 0)]
            capup = 0.0
            if base.isupper(): capup = 0.08 * UPM
            for part, nm in mk[j].items():
                if part == 'above': parts.append((nm, bi['cx'], bi['top'] + capup))
                else:
                    dx = bi['cx'] - (0.12 * bi['adv'] if base in 'yY' else 0)
                    parts.append((nm, dx, 0))
            glyphs[gid(ch, i, j)] = dict(glyph=composite(parts), adv=int(round(bi['adv'])))
    for pos in ('init', 'fina'):     # y đầu từ / cuối từ mang dấu: ý, ỳ, ỷ, ỹ, ỵ
        if base + '@' + pos not in info: continue
        pb = info[base + '@' + pos]; NV[ch + '@' + pos] = (len(pb), len(mk))
        for i, bi in enumerate(pb, 1):
            for j in sorted(mk):
                parts = [(bi['name'], 0, 0)]
                for part, nm in mk[j].items():
                    parts.append((nm, bi['cx'], bi['top']) if part == 'above' else (nm, bi['cx'] - (0.12 * bi['adv'] if base in 'yY' else 0), 0))
                glyphs[gid(ch + '@' + pos, i, j)] = dict(glyph=composite(parts), adv=int(round(bi['adv'])))
for ch, lst in info.items(): NV.setdefault(ch, (len(lst), 1))

# ---------- dựng font ----------
names = ['.notdef', 'space'] + sorted(n for n in glyphs)
pen = TTGlyphPen(None); empty = pen.glyph()
G = {'.notdef': empty, 'space': empty}; M = {'.notdef': (500, 0), 'space': (int(round(P['gap_xh'] * XH * S - 2 * (P['side'] * UPM - P['overlap'] * S))), 0)}
for n in names[2:]:
    G[n] = glyphs[n]['glyph']; M[n] = (glyphs[n]['adv'], 0)
fb = FontBuilder(UPM, isTTF=True)
fb.setupGlyphOrder(names)
cmap = {32: 'space', 0xA0: 'space'}
for ch, (nb, nm) in NV.items():
    if len(ch) == 1 and ch != 'ı': cmap[ord(ch)] = gid(ch, 1, 1)
cmap[0x131] = gid('ı', 1, 1)
for src, dst in (('"', '“'), ('–', '—'), ('’', '”'), ("'", '”')):
    if dst in NV: cmap[ord(src)] = gid(dst, 1, 1)
fb.setupCharacterMap(cmap)
fb.setupGlyf(G)
for n in names:       # lề trái theo nét thật
    g = fb.font['glyf'][n]
    if g.numberOfContours != 0: g.recalcBounds(fb.font['glyf'])      # cả chữ ghép (chữ + dấu) để lề trái và độ cao font đúng
fb.setupHorizontalMetrics({n: (M[n][0], getattr(fb.font['glyf'][n], 'xMin', 0) or 0) for n in names})
# độ cao, độ sâu khai báo của font phải bao trọn nét thật (vòng trên cao, đuôi dưới dài): Safari dùng hai số này
# để tính vùng vẽ chữ, khai báo thấp hơn nét thật thì chữ bị cắt ngọn (27/10)
YMAX = max(getattr(fb.font['glyf'][n], 'yMax', 0) or 0 for n in names); YMIN = min(getattr(fb.font['glyf'][n], 'yMin', 0) or 0 for n in names)
ASCENT, DESCENT = max(2314, int(YMAX) + 7), int(YMIN - 30)   # 2314: số đang dùng trên trang (26/10), giữ để dòng thơ không xê dịch; chữ nào cao hơn thì tăng theo
print('cao nhat', YMAX, 'thap nhat', YMIN)
fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
fb.setupNameTable({'familyName': 'HPT Tay', 'styleName': 'Regular', 'uniqueFontIdentifier': 'HPT Tay 2026-10-26', 'fullName': 'HPT Tay', 'psName': 'HPTTay-Regular', 'version': 'Version 2.0'})
fb.setupOS2(sTypoAscender=ASCENT, sTypoDescender=DESCENT, sTypoLineGap=0, usWinAscent=ASCENT, usWinDescent=-DESCENT, sxHeight=int(0.43 * UPM), sCapHeight=int(ASC * 0.43 * UPM))
fb.setupPost()
# cv01..cv11: (bản chữ i, bản dấu j) -> số tính năng (i-1) + 6(j-1); dạng cuối từ: cv20 + (i-1) + 6(j-1)
NB = 6
fea = []
ligs = [f"sub {' '.join(gid(c, 1, 1) for c in d)} by {gid(d, 1, 1)};" for d in DIGRAPHS if d in NV]
fea.append("feature liga { " + ' '.join(ligs) + " } liga;")   # đặt trước để chữ ghép thành âm ghép rồi mới đổi bản
for f in range(1, 2 * NB):
    i, j = f % NB + 1, f // NB + 1
    subs = [f"sub {gid(ch, 1, 1)} by {gid(ch, i, j)};" for ch, (nb, nm) in sorted(NV.items()) if '@' not in ch and i <= nb and j <= nm and gid(ch, i, j) in G]
    if subs: fea.append(f"feature cv{f:02d} {{ {' '.join(subs)} }} cv{f:02d};")
# ss02: chữ in hoa (cho chữ viết tắt toàn chữ hoa)
subs = [f"sub {gid(k.split('@')[0], 1, 1)} by {gid(k, 1, 1)};" for k in sorted(NV) if k.endswith('@print') and gid(k.split('@')[0], 1, 1) in G]
if subs: fea.append("feature ss02 { " + ' '.join(subs) + " } ss02;")
POSV = {}
for f in range(0, 2 * NB):
    i, j = f % NB + 1, f // NB + 1
    subs = [f"sub {gid(k.split('@')[0], 1, 1)} by {gid(k, i, j)};" for k, (nb, nm) in sorted(NV.items())
            if k.endswith('@fina') and i <= nb and j <= nm and gid(k, i, j) in G and gid(k.split('@')[0], 1, 1) in G]
    if subs: fea.append(f"feature cv{20 + f:02d} {{ {' '.join(subs)} }} cv{20 + f:02d};")
for k, (nb, nm) in NV.items():
    if k.endswith('@fina') and not k.startswith('ı'):
        b0 = k.split('@')[0]
        if gid(b0, 1, 1) in G: POSV.setdefault(b0, {})['f'] = [nb, nm]
json.dump(POSV, open(WD + '/positional.json', 'w'), ensure_ascii=False)
# đồng bộ từ phiên khác (27/10): âm ghép ch bản 1, 2 không dùng, thay bằng bản 3, 4 — lookup cuối cùng, chạy sau các cv như font đang chạy trên trang
fea.append("feature liga { sub lig_ch.11 by lig_ch.31; sub lig_ch.21 by lig_ch.41; } liga;")
addOpenTypeFeaturesFromString(fb.font, '\n'.join(fea))
fb.save(OUT)
json.dump({ch: list(v) for ch, v in NV.items() if ch != 'ı' and '@' not in ch}, open(WD + '/variants.json', 'w'), ensure_ascii=False)
print('font', OUT, len(names), 'glyph,', len(cmap), 'ky tu')
