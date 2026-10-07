"""Thư viện dùng chung cho Topic A: LiDAR-camera projection QA.

Ý tưởng đo (không cần model):
  * "Quần thể điểm của object": những điểm LiDAR nằm TRONG box 3D ground-truth (xác định ở LiDAR/camera
    frame với calibration ĐÚNG) và chiếu được vào ảnh. Quần thể này cố định, không phụ thuộc drift.
  * hit rate  = % điểm của quần thể đó rơi vào 2D box của đúng object sau khi chiếu bằng calibration BỊ LỆCH.
  * precision = % điểm rơi vào 2D box (sau khi lệch) mà thực sự thuộc box 3D của object đó.
  * inside_fov = % điểm LiDAR hữu hạn rơi trong khung ảnh.
  * alignment score = các điểm LiDAR nằm ở biên độ sâu (depth discontinuity) phải trùng với cạnh ảnh (Canny):
    điểm số = trung bình exp(-khoảng_cách_tới_cạnh_gần_nhất / sigma) tại các điểm biên đó.

Mọi hàm ở đây thuần numpy/opencv, không có phần ngẫu nhiên => cùng input luôn cho cùng số.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import cam_to_image, perturb_extrinsic, velo_to_cam

BAND_EDGES_M = (0.0, 15.0, 30.0, 50.0, np.inf)
BAND_NAMES = ("0-15m", "15-30m", "30-50m", ">50m")
N_BANDS = len(BAND_NAMES)
ROT_KINDS = ("roll", "pitch", "yaw")           # đơn vị: độ
TRANS_KINDS = ("tx", "ty", "tz")               # đơn vị: cm
MIN_POP_POINTS = 10                            # object phải có >= 10 điểm LiDAR nhìn thấy trong ảnh
MAX_OBJECT_DEPTH_M = 80.0
CANNY_LO, CANNY_HI = 50, 150
EDGE_ABS_THR_M, EDGE_REL_THR = 1.0, 0.10       # điểm là "biên độ sâu" nếu sâu hơn láng giềng gần nhất > max(1 m, 10% depth)
MIN_EDGE_POINTS = 30
SIGMA_DEG = 0.3                                # độ rộng "hố hút" quanh cạnh ảnh, tính theo góc nhìn
NULL_SHIFT_DEG = 3.0                           # dịch ảnh giả để ước lượng mức "ngẫu nhiên" của chính frame này


def band_of(depth_m: float) -> int:
    for i in range(N_BANDS):
        if BAND_EDGES_M[i] <= depth_m < BAND_EDGES_M[i + 1]:
            return i
    return N_BANDS - 1


def unit_of(kind: str) -> str:
    return "deg" if kind in ROT_KINDS else "cm"


def drifted_calib(calib: KittiCalib, kind: str, value: float) -> KittiCalib:
    """Calibration bị lệch `value` theo `kind` (roll/pitch/yaw: độ; tx/ty/tz: cm). Lệch áp trong LiDAR frame."""
    if kind in ROT_KINDS:
        return perturb_extrinsic(calib, **{f"{kind}_deg": float(value)})
    if kind in TRANS_KINDS:
        t = [0.0, 0.0, 0.0]
        t["xyz".index(kind[1])] = float(value) / 100.0
        return perturb_extrinsic(calib, t_xyz_m=tuple(t))
    raise ValueError(f"kind không hợp lệ: {kind}")


def project_full(points_xyz: np.ndarray, calib: KittiCalib, image_shape):
    """Như `project_velo_to_image` nhưng giữ nguyên thứ tự N điểm: trả (uv (N,2) NaN nếu không hợp lệ, z_cam (N,), mask (N,))."""
    cam = velo_to_cam(points_xyz[:, :3], calib)
    uv_full, mask = _project_cam_full(cam, calib.P2, image_shape)
    return uv_full, cam[:, 2], mask


def _project_cam_full(cam: np.ndarray, P2: np.ndarray, image_shape):
    uv, _, mask = cam_to_image(cam, P2, image_shape)
    uv_full = np.full((len(mask), 2), np.nan)
    uv_full[mask] = uv
    return uv_full, mask


def points_in_box3d(points_cam: np.ndarray, obj: KittiObject, xz_margin: float = 0.1,
                    y_top_margin: float = 0.05, y_bottom_clear: float = 0.15) -> np.ndarray:
    """Mask (N,) điểm (camera frame) nằm trong box 3D của label.

    `location` là tâm ĐÁY, y hướng xuống nên thân box là local_y trong [-h, 0]. Bỏ 15 cm sát đáy để không
    lẫn điểm mặt đường vào object.
    """
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (points_cam - obj.location) @ R              # = R^T (p - loc) cho từng điểm hàng
    with np.errstate(invalid="ignore"):
        return ((np.abs(local[:, 0]) <= l / 2 + xz_margin) & (np.abs(local[:, 2]) <= w / 2 + xz_margin)
                & (local[:, 1] <= -y_bottom_clear) & (local[:, 1] >= -h - y_top_margin))


def is_eligible(obj: KittiObject) -> bool:
    x1, y1, x2, y2 = obj.bbox
    return (obj.type not in ("DontCare", "Misc") and obj.truncated < 0.5 and obj.occluded <= 2
            and x2 > x1 and y2 > y1 and 1.0 < obj.location[2] < MAX_OBJECT_DEPTH_M)


@dataclass
class ObjCtx:
    obj: KittiObject
    band: int
    idx_box: np.ndarray    # chỉ số điểm (trong mảng điểm hữu hạn) nằm trong box 3D
    idx_pop: np.ndarray    # tập con nhìn thấy trong ảnh khi calibration đúng


@dataclass
class FrameCtx:
    frame_id: str
    points: np.ndarray     # (N, 4) chỉ điểm hữu hạn
    n_raw_points: int
    calib: KittiCalib
    image_shape: tuple
    objects: list
    weight_map: np.ndarray  # exp(-dist_to_image_edge / sigma), float32 (H, W)
    edge_map: np.ndarray    # Canny uint8 (H, W)
    chance: float           # trung bình weight_map: điểm số kỳ vọng nếu LiDAR biên đặt ngẫu nhiên
    win: int                # cửa sổ tìm láng giềng cho biên độ sâu (pixel, lẻ)
    null_px: int            # độ dịch (pixel) dùng để ước lượng mức điểm số "ngẫu nhiên"


def edge_weight_map(image: np.ndarray, focal_px: float):
    """Thuật toán A (Canny): (edge_map, weight_map) với weight = exp(-khoảng_cách_tới_cạnh_Canny / sigma)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)
    dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    sigma = max(2.0, focal_px * np.deg2rad(SIGMA_DEG))     # góc nhìn -> pixel: KITTI f~721 px, nuScenes f~1266 px
    return edges, np.exp(-dist / sigma).astype(np.float32)


def build_ctx(fr: dict) -> FrameCtx:
    """Chuẩn bị mọi thứ không phụ thuộc drift cho 1 frame (gọi 1 lần, đánh giá nhiều calibration)."""
    pts_all = fr["points"]
    finite = np.isfinite(pts_all).all(axis=1)
    pts = pts_all[finite]
    calib, shape = fr["calib"], fr["image"].shape

    cam = velo_to_cam(pts[:, :3], calib)
    _, mask_true = _project_cam_full(cam, calib.P2, shape)

    objects = []
    for obj in fr["labels"]:
        if not is_eligible(obj):
            continue
        idx_box = np.flatnonzero(points_in_box3d(cam, obj))
        idx_pop = idx_box[mask_true[idx_box]]
        if len(idx_pop) >= MIN_POP_POINTS:
            objects.append(ObjCtx(obj, band_of(float(obj.location[2])), idx_box, idx_pop))

    focal = float(calib.P2[0, 0])
    edges, weight = edge_weight_map(fr["image"], focal)
    # Cửa sổ láng giềng ~1.2 lần khoảng cách trung bình giữa 2 điểm LiDAR trên ảnh (KITTI 64 beam ~5 px, nuScenes
    # 32 beam ~21 px), luôn lẻ. Cố định theo calibration đúng, không đổi khi drift => so sánh công bằng giữa các mức.
    spacing = np.sqrt(shape[0] * shape[1] / max(int(mask_true.sum()), 1))
    win = int(np.clip(round(1.2 * spacing) | 1, 3, 41))
    return FrameCtx(fr["frame_id"], pts, int(len(pts_all)), calib, shape, objects, weight, edges,
                    float(weight.mean()), win, int(round(focal * np.tan(np.deg2rad(NULL_SHIFT_DEG)))))


def lidar_edge_points(uv: np.ndarray, depth: np.ndarray, image_shape, win: int):
    """Đánh dấu điểm LiDAR là "biên độ sâu": sâu hơn hẳn láng giềng gần nhất trong cửa sổ win x win trên ảnh.

    Trả (is_edge (M,), u_int (M,), v_int (M,)). Điểm bị đánh dấu là phía XA của một bước nhảy depth, tức nằm sát
    mép vật thể phía trước, nên kỳ vọng trùng với cạnh ảnh khi calibration đúng.
    """
    height, width = image_shape[:2]
    ui = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, width - 1)
    vi = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, height - 1)
    order = np.argsort(-depth)                            # xa trước, gần sau => ghi đè cuối cùng là điểm gần nhất
    sparse = np.full((height, width), 1e6, dtype=np.float32)
    sparse[vi[order], ui[order]] = depth[order].astype(np.float32)
    nearest = cv2.erode(sparse, np.ones((win, win), np.uint8))   # min filter
    gap = depth - nearest[vi, ui]
    return gap > np.maximum(EDGE_ABS_THR_M, EDGE_REL_THR * depth), ui, vi


def alignment_score(ctx: FrameCtx, uv: np.ndarray, depth: np.ndarray):
    """(score, n_edge_points, contrast). NaN nếu quá ít biên độ sâu để đo.

    score    = trung bình weight_map tại các điểm LiDAR biên độ sâu (thô, phụ thuộc độ nhiều cạnh của cảnh).
    contrast = score - score_null, với score_null là điểm số của CHÍNH các điểm đó khi dịch ảnh +-NULL_SHIFT_DEG
               theo 4 hướng. Trừ như vậy loại bỏ phần "cảnh nhiều cạnh nên điểm số cao" giữa các frame.
    """
    if len(depth) == 0:
        return float("nan"), 0, float("nan")
    is_edge, ui, vi = lidar_edge_points(uv, depth, ctx.image_shape, ctx.win)
    n_edge = int(is_edge.sum())
    if n_edge < MIN_EDGE_POINTS:
        return float("nan"), n_edge, float("nan")
    height, width = ctx.image_shape[:2]
    ue, ve = ui[is_edge], vi[is_edge]
    score = float(ctx.weight_map[ve, ue].mean())
    d = ctx.null_px
    null = np.mean([ctx.weight_map[np.clip(ve + dv, 0, height - 1), np.clip(ue + du, 0, width - 1)].mean()
                    for du, dv in ((d, 0), (-d, 0), (0, d), (0, -d))])
    return score, n_edge, float(score - null)


def object_counts(ctx: FrameCtx, uv_full: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mảng (n_obj, 4): [n_pop, n_hit, n_in_box, n_in_box_true] cho từng object ở calibration hiện tại."""
    out = np.zeros((len(ctx.objects), 4), dtype=np.int64)
    ids = np.flatnonzero(mask)
    u_all, v_all = uv_full[ids, 0], uv_full[ids, 1]
    for k, oc in enumerate(ctx.objects):
        x1, y1, x2, y2 = oc.obj.bbox
        pu, pv = uv_full[oc.idx_pop, 0], uv_full[oc.idx_pop, 1]
        with np.errstate(invalid="ignore"):
            hit = int(((pu >= x1) & (pu <= x2) & (pv >= y1) & (pv <= y2)).sum())
        in_box = ids[(u_all >= x1) & (u_all <= x2) & (v_all >= y1) & (v_all <= y2)]
        true_n = int(np.isin(in_box, oc.idx_box, assume_unique=True).sum())
        out[k] = (len(oc.idx_pop), hit, len(in_box), true_n)
    return out


def evaluate(ctx: FrameCtx, calib: KittiCalib, with_align: bool = True) -> dict:
    """Đo 1 frame với 1 calibration. Trả dict số đếm (cộng được giữa các frame) + alignment score."""
    cam = velo_to_cam(ctx.points[:, :3], calib)
    uv_full, mask = _project_cam_full(cam, calib.P2, ctx.image_shape)
    counts = object_counts(ctx, uv_full, mask)
    per_band = np.zeros((N_BANDS, 4), dtype=np.int64)
    for oc, row in zip(ctx.objects, counts):
        per_band[oc.band] += row
    res = {"n_points": int(len(ctx.points)), "n_inside": int(mask.sum()), "n_objects": len(ctx.objects),
           "per_band": per_band, "per_object": counts}
    if with_align:
        res["align_score"], res["n_edge"], res["align_contrast"] = alignment_score(ctx, uv_full[mask], cam[mask, 2])
    else:
        res["align_score"], res["n_edge"], res["align_contrast"] = float("nan"), 0, float("nan")
    return res


def hit_rate(pop: float, hit: float) -> float:
    return float(hit / pop) if pop > 0 else float("nan")


# ---------------------------------------------------------------------------------------------
# Hình ảnh phục vụ demo / failure case
# ---------------------------------------------------------------------------------------------
def put_title(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """Thêm dải tiêu đề đen phía trên ảnh."""
    h = 30
    bar = np.zeros((h, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def crop_box(image: np.ndarray, bbox, margin: int = 60, min_size: int = 160):
    """Cắt vùng quanh bbox (kèm lề), đảm bảo kích thước tối thiểu. Trả (crop, x0, y0)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) / 2 + margin, min_size / 2)
    half_h = max((y2 - y1) / 2 + margin, min_size / 2)
    x0, xe = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    y0, ye = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    return image[y0:ye, x0:xe], x0, y0
