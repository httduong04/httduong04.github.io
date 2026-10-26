# Tách nét mực của hai bài haiku viết tay: làm phẳng nền giấy, xóa dòng kẻ vở, giữ nguyên bố cục Nguyên viết
import sys, numpy as np, cv2
sys.path.insert(0, '../font3')
from clean import cleaned
def flatten(V, k=61):
    bg = cv2.morphologyEx(V, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), 20).astype(np.float32)
    return np.clip(V.astype(np.float32) / np.maximum(bg, 1) * 250, 0, 255)
def alpha_from(F, keep):
    ink = F[keep]; dark = np.percentile(ink, 3) if ink.size else 40
    hi = 200.0
    a = np.clip((hi - F) / max(30, hi - dark - 10), 0, 1)
    a[~keep] = 0
    return a
def components_mask(F, thr, minA, box, far=100):
    """giữ nét chữ; mảnh nhỏ chỉ giữ khi đậm như mực (chấm chữ i, chấm dưới) và nằm gần chữ; vệt dòng kẻ còn sót thì nhạt hơn nên bỏ"""
    b = (F < thr).astype(np.uint8); x0, y0, x1, y1 = box; m = np.zeros_like(b); m[y0:y1, x0:x1] = 1; b &= m
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    big = np.isin(lab, [i for i in range(1, n) if st[i, 4] >= 300]).astype(np.uint8)
    dist = cv2.distanceTransform(1 - big, cv2.DIST_L2, 5)
    keep = np.zeros(b.shape, bool)
    for i in range(1, n):
        sel = lab == i
        if dist[sel].min() > far: continue                                   # chấm bẩn nằm xa chữ
        if st[i, 4] >= minA or (st[i, 4] >= 6 and F[sel].min() < 80): keep |= sel
    return keep, st
def save(a, path, pad=24):
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.08).astype(np.uint8), 8)
    for i in range(1, n):            # vệt mờ, nhỏ (dòng kẻ còn sót): bỏ
        if st[i, 4] < 60 and a[lab == i].max() < 0.8: a[lab == i] = 0
    ys, xs = np.nonzero(a > 0.08)
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
    a = a[y0:y1, x0:x1]
    rgba = np.zeros(a.shape + (4,), np.uint8); rgba[..., :3] = 255; rgba[..., 3] = (a * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    print(path, a.shape[::-1])
    return a
# bài 1: mướp chín (ảnh chụp, giấy trắng)
im = cv2.imread('p-000.png'); V = im.max(2)
F = flatten(V)
H, W = V.shape
keep, st = components_mask(F, 150, 12, (60, 60, W - 60, H - 60))
keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
a1 = save(alpha_from(F, keep), 'muop-chin-muc.png')
# bài 2: lông tay (vở ô ly, xoay lại cho chữ nằm ngang)
cv2.imwrite('p2-xoay.png', cv2.rotate(cv2.imread('p-001.png'), cv2.ROTATE_90_COUNTERCLOCKWISE))
G, _ = cleaned('p2-xoay.png')
F2 = flatten(G)
H2, W2 = G.shape
keep2, st2 = components_mask(F2, 130, 25, (int(0.18 * W2), int(0.04 * H2), W2 - 20, int(0.45 * H2)))   # trên: chừa đủ chỗ cho vòng trên của l, h dòng đầu
keep2 = cv2.dilate(keep2.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
a2 = save(alpha_from(F2, keep2), 'long-tay-muc.png')
for name, a in (('muop-chin', a1), ('long-tay', a2)):
    vis = (255 - a * 255).astype(np.uint8); cv2.imwrite(name + '-xem.png', vis)
