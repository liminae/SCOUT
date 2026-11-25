import numpy as np


def landmarks(points, width=1, height=1):
    p = np.asarray(points, dtype=float)
    if p.shape!=(4, 2) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('Expected four normalized (x,y) landmarks')
    if not np.isfinite([width, height]).all() or min(width, height)<=0:
        raise ValueError('Invalid image dimensions')
    a, b, c, d = p*[width, height]
    if a[0]>=b[0] or c[0]>=d[0]:
        raise ValueError('Mark the left landmark before the right landmark')
    scale = ((b[0]-a[0])+(d[0]-c[0]))/2
    return {
        'upper_offset': float((b[1]-a[1])/scale),
        'lower_offset': float((d[1]-c[1])/scale),
        'center_offset': float((c[0]+d[0]-a[0]-b[0])/(2*scale))
    }


def profile(mask):
    m = np.asarray(mask)
    if m.ndim!=2 or min(m.shape)<16:
        raise ValueError('Expected a two-dimensional silhouette mask')
    ys, xs = np.nonzero(m)
    if not len(ys):
        raise ValueError('Empty silhouette')
    top, bottom = int(ys.min()), int(ys.max())
    h = bottom-top+1
    if h<32:
        raise ValueError('Silhouette is too small')
    rows = []
    for y in range(top+int(.15*h), top+int(.85*h)):
        x = np.flatnonzero(m[y])
        if len(x)>1:
            rows.append((y, int(x[0]), int(x[-1])))
    if len(rows)<.6*h:
        raise ValueError('Silhouette is incomplete')
    rows = np.asarray(rows, dtype=float)
    y, left, right = rows.T
    center = (left+right)/2
    width = right-left+1
    scale = np.median(width)
    if scale<5:
        raise ValueError('Silhouette is too narrow')
    t = (y-y.min())/(y.max()-y.min())
    fit = np.polyval(np.polyfit(t, center, 1), t)
    error = (center-fit)/scale
    upper = center[t<=.15].mean()
    lower = center[t>=.85].mean()
    metrics = {
        'center_shift': float((lower-upper)/scale),
        'centerline_rms': float(np.sqrt(np.mean(error**2))),
        'centerline_max': float(np.max(np.abs(error))),
        'width_variation': float(np.std(width)/scale)
    }
    return metrics, rows
