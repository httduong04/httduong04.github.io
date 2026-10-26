# Xem thử font mới: âm ghép, bản thay thế, dạng cuối từ — cùng cách chọn như trong phòng thơ
import sys, json, random, re
from PIL import Image, ImageDraw, ImageFont
fp, out = sys.argv[1], sys.argv[2]
HV = json.load(open('work/variants.json')); HP = json.load(open('work/positional.json'))
DG = ['ngh', 'ch', 'gh', 'gi', 'kh', 'ng', 'nh', 'ph', 'qu', 'th', 'tr']
LET = re.compile(r'[A-Za-zÀ-ɏḀ-ỿ]')
def units(t):
    i = 0
    while i < len(t):
        u = t[i]
        for d in DG:
            if t.startswith(d, i): u = d; break
        before = i > 0 and LET.match(t[i - 1]); i += len(u); after = i < len(t) and LET.match(t[i])
        yield u, bool(before and not after)
def line(d, f, t, x, y, r, col):
    for u, fin in units(t):
        pv = HP.get(u, {}).get('f') if fin else None
        v = pv or HV.get(u); ft = []
        if v:
            bi, mj = r.randrange(v[0]), r.randrange(v[1])
            if pv: ft = ['cv%02d' % (20 + bi + 6 * mj)]
            elif bi + 6 * mj: ft = ['cv%02d' % (bi + 6 * mj)]
        d.text((x, y), u, font=f, fill=col, features=ft, anchor='ls'); x += f.getlength(u, features=ft)
L = sys.argv[4:] if len(sys.argv) > 4 else ['những người phụ nữ nóng tính', 'khi đi dạo trên con đường cỏ', 'ngày hôm nay bình thường', 'tôi nghĩ về giấc mơ quanh thành phố', 'nghẹt thở, chim hót, nhà ghế']
size = int(sys.argv[3]) if len(sys.argv) > 3 else 80
f = ImageFont.truetype(fp, size); lh = int(size * 1.9)
im = Image.new('RGB', (int(size * 22), int(size * 1.2) + lh * len(L)), (18, 16, 16)); d = ImageDraw.Draw(im)
for k, t in enumerate(L): line(d, f, t, int(size * 0.5), int(size * 1.4) + k * lh, random.Random(k + 3), (247, 239, 226))
im.save(out)
