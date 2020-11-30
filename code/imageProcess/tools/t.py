import cv2


def imgRead(fpath, float=False):
    img = cv2.imread(fpath)
    if float:
        return img.astype(np.float32)
    return img


def imgSave(fpath, img):
    print("Image Save\n\r", fpath)
    cv2.imwrite(fpath, img)
    return fpath

def imgShow(img, title="result"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
