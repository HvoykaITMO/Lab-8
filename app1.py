import cv2


img_bgr = cv2.imread('images/variant-3.jpeg')
assert img_bgr is not None, "file could not be read, check with os.path.exists()"

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

cv2.imshow('BGR', img_bgr)
cv2.imshow('HSV', img_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
