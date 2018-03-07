from common import *

# draw -----------------------------------
def image_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)


def draw_mask(image, mask, color=(255,255,255), α=1,  β=0.25, λ=0., threshold=32 ):
    # image * α + mask * β + λ

    if threshold is None:
        mask = mask/255
    else:
        mask = clean_mask(mask,threshold,1)

    mask  = np.dstack((color[0]*mask,color[1]*mask,color[2]*mask)).astype(np.uint8)
    image[...] = cv2.addWeighted(image, α, mask, β, λ)




