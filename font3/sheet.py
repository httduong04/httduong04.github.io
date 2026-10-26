# Bảng xem mọi bản viết: mỗi dòng một chữ, đủ các bản (thường và cuối từ)
import json, sys
from PIL import Image, ImageDraw, ImageFont
fp = sys.argv[1]; out = sys.argv[2]; keys = sys.argv[3].split(',')
HV = json.load(open('work/variants.json')); HP = json.load(open('work/positional.json'))
size = 64; f = ImageFont.truetype(fp, size); lab = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
rows = []
for k in keys:
    fin = k.endswith('*'); ch = k.rstrip('*')
    v = (HP.get(ch, {}).get('f') if fin else HV.get(ch)) or [1, 1]
    rows.append((k, ch, fin, v[0]))
im = Image.new('RGB', (1500, 40 + 120 * len(rows)), (18, 16, 16)); d = ImageDraw.Draw(im)
for r, (k, ch, fin, n) in enumerate(rows):
    y = 100 + r * 120
    d.text((10, y - 30), k, font=lab, fill=(200, 120, 120))
    for i in range(n):
        ft = ['cv%02d' % (20 + i)] if fin else (['cv%02d' % i] if i else [])
        x = 90 + i * 230
        txt = ('a' + ch) if fin else ch
        d.line([(x - 10, y), (x + 200, y)], fill=(60, 90, 60))
        if fin:
            d.text((x, y), 'a', font=f, fill=(150, 150, 150), anchor='ls'); x += f.getlength('a')
        d.text((x, y), ch, font=f, fill=(247, 239, 226), features=ft, anchor='ls')
im.save(out)
