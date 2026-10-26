# Thay nền chất liệu của một số bài trong trang đã gộp: ảnh thu về cao 1200, tính màu trung bình, chỗ tối/sáng, màu chữ gợi ý
import json, io, base64, colorsys, sys, numpy as np
from PIL import Image
def lumOf(c):
    ch = lambda v: (v / 255) / 12.92 if v / 255 <= 0.03928 else (((v / 255) + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(c[0]) + 0.7152 * ch(c[1]) + 0.0722 * ch(c[2])
def entry(path, crop=None):
    im = Image.open(path).convert('RGB')
    if crop: im = im.crop(crop)
    im = im.resize((round(im.width * 1200 / im.height), 1200), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=82, optimize=True, progressive=True)
    a = np.asarray(im, float).reshape(-1, 3); avg = a.mean(0)
    s = im.resize((64, round(64 * im.height / im.width)), Image.BOX); b = np.asarray(s, float).reshape(-1, 3)
    o = np.argsort(b @ [0.299, 0.587, 0.114])
    lo = b[o[int(len(o) * 0.02)]].round().astype(int).tolist(); hi = b[o[min(len(o) - 1, int(len(o) * 0.98))]].round().astype(int).tolist()
    h, l_, sat = colorsys.rgb_to_hls(*(avg / 255))
    hx = lambda c: '#%02x%02x%02x' % tuple(int(round(v * 255)) for v in c)
    cands = [['trắng ngà', '#f7efe2'], ['đen mực', '#1c1814'], ['sẫm theo nền', hx(colorsys.hls_to_rgb(h, 0.161, sat))], ['nhạt theo nền', hx(colorsys.hls_to_rgb(h, 0.929, 0.333))]]
    def rd(ink):
        li = lumOf([int(ink[k:k + 2], 16) for k in (1, 3, 5)]); a_, b_ = lumOf(lo), lumOf(hi); bg = b_ if li > (a_ + b_) / 2 else a_
        return round((max(li, bg) + 0.05) / (min(li, bg) + 0.05), 1)
    ink = max(['#f7efe2', '#1c1814'], key=rd)
    e = dict(src='data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode(), avg=hx(avg / 255), lo=lo, hi=hi, cands=cands, ink=ink, veil=0)
    print('  ', im.size, len(buf.getvalue()) // 1024, 'KB, avg', e['avg'], 'lo', lo, 'hi', hi, 'chu', ink, {c[0]: rd(c[1]) for c in cands})
    return e
def apply(src_html, dst_html, jobs):
    L = open(src_html, encoding='utf-8').read().split('\n')
    i = [k for k, l in enumerate(L) if l.startswith('var TEX = ')][0]
    T = json.loads(L[i][len('var TEX = '):].rstrip().rstrip(';'))
    for title, path, crop in jobs:
        assert title in T, title
        json.dump(T[title], open('tex-cu-' + title[:20].replace(' ', '_') + '.json', 'w'), ensure_ascii=False)
        print(title); T[title] = entry(path, crop)
    L[i] = 'var TEX = ' + json.dumps(T, ensure_ascii=False) + ';'
    open(dst_html, 'w', encoding='utf-8').write('\n'.join(L))
