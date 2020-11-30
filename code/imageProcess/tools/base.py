import numpy as np


#bgr2rgb
def Bgr2Rgb(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img
# Gray scale
def Bgr2Gray(img):
	b = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	r = img[:, :, 2].copy()

	# Gray scale
	out = 0.2126 * r + 0.7152 * g + 0.0722 * b
	out = out.astype(np.uint8)

	return out    
#colorgray
def Color2Gray(img, bgr=False):
    b = img[:, :, 0 if bgr else 2].copy().astype(np.float)
    g = img[:, :, 1].copy().astype(np.float)
    r = img[:, :, 2 if bgr else 0].copy().astype(np.float)
    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)
    return out

# binalization二值化
def Binarization(img, th=128):
	img[img < th] = 0
	img[img >= th] = 255
	return img


# Otsu Binarization
# 大津二值化算法（Otsu's Method）
def Otsu_binarization(img, retx=False):
    max_sigma = 0
    max_t = 0
    out = img.copy()
    H, W = img.shape

    # determine threshold
    for _t in range(1, 256):
        v0 = out[np.where(out < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = out[np.where(out >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # Binarization
    # print("threshold >>", max_t)
    th = max_t
    out[out < th] = 0
    out[out >= th] = 255
    if retx:
        return out, max_t
    return out

#bgr2hsv
# BGR -> HSV
def BGR2HSV(_img):
	img = _img.copy() / 255.

	hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
		
	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()
	
	return hsv


def HSV2BGR(_img, hsv):
	img = _img.copy() / 255.

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()

	out = np.zeros_like(img)

	H = hsv[..., 0]
	S = hsv[..., 1]
	V = hsv[..., 2]

	C = S
	H_ = H / 60.
	X = C * (1 - np.abs( H_ % 2 - 1))
	Z = np.zeros_like(H)

	vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

	for i in range(6):
		ind = np.where((i <= H_) & (H_ < (i+1)))
		out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
		out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
		out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

	out[np.where(max_v == min_v)] = 0
	out = np.clip(out, 0, 1)
	out = (out * 255).astype(np.uint8)

	return out
#bgr2yuv
#yuv2bgr
#erode
#dialet
#close
#open
#masking
#get_mask
# Erosion

# Erosion
def Erode(img, Erode_time=1):
    H, W = img.shape
    out = img.copy()

    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)

    # each erode
    for i in range(Erode_time):
        tmp = np.pad(out, (1, 1), 'edge')
        # erode
        for y in range(1, H+1):
            for x in range(1, W+1):
                if np.sum(MF * tmp[y - 1 : y + 2 , x - 1 : x + 2]) < 1 * 4:
                    out[y-1 , x-1] = 0

    return out


# Dilation
def Dilate(img, Dil_time=1):
    H, W = img.shape

    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)

    # each dilate time
    out = img.copy()
    for i in range(Dil_time):
        tmp = np.pad(out, (1, 1), 'edge')
        for y in range(1, H+1):
            for x in range(1, W+1):
                if np.sum(MF * tmp[y - 1 : y + 2, x - 1 : x + 2]) >= 1:
                    out[y-1 , x-1] = 1

    return out


# Opening morphology
def Morphology_Opening(img, time=1):
    out = Erode(img, Erode_time=time)
    out = Dilate(out, Dil_time=time)
    return out

# Closing morphology
def Morphology_Closing(img, time=1):
    out = Dilate(img, Dil_time=time)
    out = Erode(out, Erode_time=time)
    return out
