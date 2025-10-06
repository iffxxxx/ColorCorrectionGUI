# compare_two_pick_both.py
import os, cv2, numpy as np

PATCH_SIZE = 3
DRAW_R = 5

# ===== sRGB → Lab / ΔE00 =====
D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
M_srgb_to_xyz_d65 = np.array([[0.4124564, 0.3575761, 0.1804375],
                              [0.2126729, 0.7151522, 0.0721750],
                              [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)

def _srgb_to_linear(v01):
    return np.where(v01 <= 0.04045, v01/12.92, ((v01+0.055)/1.055)**2.4)

def encodedRGB_to_Lab_sRGB(rgb_255):
    v01 = np.clip(rgb_255/255.0, 0.0, 1.0).astype(np.float64)
    lin = _srgb_to_linear(v01)
    XYZ = lin @ M_srgb_to_xyz_d65.T
    Xn, Yn, Zn = D65
    x = XYZ[...,0]/Xn; y = XYZ[...,1]/Yn; z = XYZ[...,2]/Zn
    eps = 216/24389; k = 24389/27
    fx = np.where(x > eps, np.cbrt(x), (k*x+16)/116)
    fy = np.where(y > eps, np.cbrt(y), (k*y+16)/116)
    fz = np.where(z > eps, np.cbrt(z), (k*z+16)/116)
    L = 116*fy - 16; a = 500*(fx - fy); b = 200*(fy - fz)
    return np.stack([L, a, b], axis=-1)

def deltaE2000(Lab1, Lab2):
    L1,a1,b1 = Lab1[...,0], Lab1[...,1], Lab1[...,2]
    L2,a2,b2 = Lab2[...,0], Lab2[...,1], Lab2[...,2]
    L_bar = (L1+L2)/2
    C1 = np.sqrt(a1*a1+b1*b1); C2 = np.sqrt(a2*a2+b2*b2)
    C_bar = (C1+C2)/2; C_bar7 = C_bar**7
    G = 0.5*(1 - np.sqrt(C_bar7/(C_bar7 + 25**7)))
    a1p=(1+G)*a1; a2p=(1+G)*a2
    C1p=np.sqrt(a1p*a1p+b1*b1); C2p=np.sqrt(a2p*a2p+b2*b2)
    h1p=(np.degrees(np.arctan2(b1,a1p))%360); h2p=(np.degrees(np.arctan2(b2,a2p))%360)
    dLp = L2-L1; dCp = C2p-C1p
    dhp = h2p-h1p; dhp=np.where(dhp>180,dhp-360,dhp); dhp=np.where(dhp<-180,dhp+360,dhp)
    dHp = 2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp)/2)
    L_bar_p=(L1+L2)/2; C_bar_p=(C1p+C2p)/2
    h_bar_p=(h1p+h2p)/2; h_bar_p=np.where(np.abs(h1p-h2p)>180, h_bar_p+180, h_bar_p)%360
    T=(1 - 0.17*np.cos(np.radians(h_bar_p-30))
         + 0.24*np.cos(np.radians(2*h_bar_p))
         + 0.32*np.cos(np.radians(3*h_bar_p+6))
         - 0.20*np.cos(np.radians(4*h_bar_p-63)))
    d_ro = 30*np.exp(-((h_bar_p-275)/25)**2)
    RC = 2*np.sqrt((C_bar_p**7)/((C_bar_p**7)+25**7))
    RT = -RC*np.sin(2*np.radians(d_ro))
    SL = 1 + (0.015*(L_bar_p-50)**2)/np.sqrt(20+(L_bar_p-50)**2)
    SC = 1 + 0.045*C_bar_p
    SH = 1 + 0.015*C_bar_p*T
    return np.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2 + RT*(dCp/SC)*(dHp/SH))

# ===== ROI 샘플링 & 보드 =====
def get_patch_colors(img_rgb_float, centers, patch_size=3):
    hs = patch_size // 2
    h, w = img_rgb_float.shape[:2]
    out = []
    for (cx, cy) in centers:
        x0 = max(int(round(cx)) - hs, 0)
        x1 = min(int(round(cx)) + hs + 1, w)
        y0 = max(int(round(cy)) - hs, 0)
        y1 = min(int(round(cy)) + hs + 1, h)
        roi = img_rgb_float[y0:y1, x0:x1, :]
        out.append(roi.reshape(-1, 3).mean(axis=0))
    return np.asarray(out, dtype=np.float32)

def make_swatch_board_2imgs(rgb1_24, rgb2_24, dE00, cell=80, gap=6, out_path="biz_swatches.png"):
    rows, cols = 4, 6
    H = rows*cell + (rows+1)*gap
    W = cols*cell + (cols+1)*gap
    th = 30
    canvas = np.full((H+th, W, 3), 245, np.uint8)
    title = f"1.png vs 2.png  |  mean ΔE00={float(np.mean(dE00)):.2f}"
    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,40), 2, cv2.LINE_AA)

    for i in range(24):
        r, c = divmod(i, 6)
        y0 = th + gap + r*(cell+gap)
        x0 = gap + c*(cell+gap)
        y1, x1 = y0+cell, x0+cell
        c1 = np.clip(rgb1_24[i], 0, 255).astype(np.uint8)
        c2 = np.clip(rgb2_24[i], 0, 255).astype(np.uint8)
        canvas[y0:y1, x0:x1] = 230
        pts1 = np.array([[x0,y0],[x1,y0],[x0,y1]], np.int32)
        pts2 = np.array([[x1,y1],[x1,y0],[x0,y1]], np.int32)
        cv2.fillConvexPoly(canvas, pts1, (int(c1[2]), int(c1[1]), int(c1[0])))
        cv2.fillConvexPoly(canvas, pts2, (int(c2[2]), int(c2[1]), int(c2[0])))
        cv2.rectangle(canvas, (x0,y0), (x1,y1), (180,180,180), 1, cv2.LINE_AA)
        label = f"{i:02d} dE{dE00[i]:.2f}"
        (tw, th_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tx, ty = x0 + 5, y0 + cell - 7
        cv2.rectangle(canvas, (tx-2, ty-th_txt-3), (tx+tw+2, ty+3), (40,40,40), -1)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)
    print(f"[SAVE] {out_path}")

# ===== 마우스 클릭으로 center 찍기 =====
class ClickPicker:
    def __init__(self, img_bgr, win):
        self.base = img_bgr
        self.view = img_bgr.copy()
        self.win = win
        self.points = []
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 24:
            self.points.append((x, y)); self._redraw()
    def _redraw(self):
        self.view = self.base.copy()
        for i,(x,y) in enumerate(self.points):
            cv2.circle(self.view, (x, y), DRAW_R, (0,215,255), -1, cv2.LINE_AA)
            cv2.putText(self.view, f"{i:02d}", (x+7,y-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(self.view, f"{i:02d}", (x+7,y-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),1,cv2.LINE_AA)
    def pick(self):
        self._redraw()
        print(f"[{self.win}] 24개 클릭: 좌→우, 위→아래. (u=undo, c=clear, Enter=완료, q=종료)")
        while True:
            cv2.imshow(self.win, self.view)
            key = cv2.waitKey(20) & 0xFF
            if key==ord('u') and self.points: self.points.pop(); self._redraw()
            elif key==ord('c'): self.points.clear(); self._redraw()
            elif key in (13,10):
                if len(self.points)==24:
                    cv2.destroyWindow(self.win); return self.points
                else: print(f"{len(self.points)}/24개. 24개 찍어야 완료.")
            elif key==ord('q'): cv2.destroyWindow(self.win); raise SystemExit

# ===== 메인 =====
def main():
    p1, p2 = "12_ori.png","20_ori.png"
    if not os.path.exists(p1) or not os.path.exists(p2):
        raise FileNotFoundError("12.png, 20.png 필요")
    bgr1, bgr2 = cv2.imread(p1), cv2.imread(p2)
    picker1 = ClickPicker(bgr1, "Pick centers on 1.png"); centers1 = picker1.pick()
    picker2 = ClickPicker(bgr2, "Pick centers on 2.png"); centers2 = picker2.pick()
    rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB).astype(np.float32)
    cols1 = get_patch_colors(rgb1, centers1, patch_size=PATCH_SIZE)
    cols2 = get_patch_colors(rgb2, centers2, patch_size=PATCH_SIZE)
    lab1, lab2 = encodedRGB_to_Lab_sRGB(cols1), encodedRGB_to_Lab_sRGB(cols2)
    dE00 = deltaE2000(lab1, lab2)
    print("=== ΔE00 per-patch ===")
    for i,v in enumerate(dE00):
        r,c = divmod(i,6); print(f"idx={i:02d} (r{r},c{c}) : {v:.3f}")
    print(f"mean={np.mean(dE00):.3f}, median={np.median(dE00):.3f}, "
          f"p95={np.quantile(dE00,0.95):.3f}, max={np.max(dE00):.3f}")
    make_swatch_board_2imgs(cols1, cols2, dE00, out_path="biz_swatches.png")

if __name__=="__main__": main()
