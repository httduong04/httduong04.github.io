# Xóa dòng kẻ ô ly (ngang, dọc) và lề đỏ, giữ nét mực.
# Chỗ dòng kẻ: lấy giá trị hai bên dòng kẻ (trên/dưới với dòng ngang, trái/phải với dòng dọc), lấy bên sáng hơn.
# Nét mực cắt ngang dòng kẻ thì hai bên đều tối nên nét được nối liền; dòng kẻ trơn thì hai bên sáng nên biến mất.
import numpy as np, cv2

def _level(V, mask, axis, n=81):
    """độ sáng thường gặp của dòng kẻ quanh mỗi điểm, đo dọc theo dòng kẻ (mực đậm tính như 160 để không kéo lệch)"""
    v = np.clip(V.astype(np.float32), 160, 255) * mask
    k = (n, 1) if axis == 0 else (1, n)        # cỡ khung (rộng, cao): dòng ngang đo theo x, dòng dọc đo theo y
    num = cv2.blur(v, k); den = cv2.blur(mask.astype(np.float32), k)
    return num / np.maximum(den, 1e-3)

def _fill(V, mask, allmask, axis, maxd=14):
    lo = np.full(V.shape, -1, np.int32); hi = np.full(V.shape, -1, np.int32)
    for k in range(1, maxd + 1):
        for sgn, arr in ((1, lo), (-1, hi)):
            Vs = np.roll(V, sgn * k, axis); Ms = np.roll(allmask, sgn * k, axis)
            take = (arr < 0) & ~Ms
            arr[take] = Vs[take]
    lo[lo < 0] = 255; hi[hi < 0] = 255
    out = V.astype(np.int32).copy()
    lev = _level(V, mask, axis)
    sel = mask & (V > lev - 18)          # điểm tối hơn hẳn dòng kẻ là mực đè lên dòng kẻ: giữ nguyên
    out[sel] = np.maximum(lo, hi)[sel]
    return out.astype(np.uint8)

def cleaned(path, L=101):
    im = cv2.imread(path); B, G, R = [im[:, :, i].astype(int) for i in range(3)]
    V = im.max(2).astype(np.uint8)
    cand = (V < 218).astype(np.uint8)
    cd = cv2.dilate(cand, np.ones((3, 3), np.uint8))
    lh = cv2.morphologyEx(cd, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    lv = cv2.morphologyEx(cd, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    red = ((R - B > 22) & (R - G > 12)).astype(np.uint8)
    mh = cv2.dilate(lh, np.ones((3, 3), np.uint8)) > 0
    mv = cv2.dilate(lv | red, np.ones((3, 3), np.uint8)) > 0
    allm = mh | mv
    out = _fill(V, mh, allm, 0)
    out = _fill(out, mv, allm, 1)
    return out, allm
