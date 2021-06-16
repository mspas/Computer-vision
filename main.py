import cv2
import numpy as np
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
import imutils

thres = 0.5
nms_thres = 0.5
width = 23

maxDimA = 300.0
maxDimB = 85.0


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def getPixelsPerMetric(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)

    cv2.imshow("edges", edged)
    cv2.waitKey(0)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    for c in cnts:
        if cv2.contourArea(c) < 500:
            continue

        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 0), -1)

        (tl, tr, br, bl) = box

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width

        return pixelsPerMetric


def main():
    img = cv2.imread('images/e10a.jpg')

    pixelsPerMetric = getPixelsPerMetric(img);

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confdence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classNames[classId - 1] == "bottle":
                x, y, w, h = box[0], box[1], box[2], box[3]

                dimA = h / pixelsPerMetric
                dimB = w / pixelsPerMetric

                colorRed = (0, 0, 255)
                colorGreen = (0, 255, 0)

                color = colorGreen
                colorTextA = colorGreen
                colorTextB = colorGreen

                if dimA > maxDimA:
                    color = colorRed
                    colorTextA = colorRed
                if dimB > maxDimB:
                    color = colorRed
                    colorTextB = colorRed

                cv2.rectangle(img, (x, y), (x + w, h + y), color=color, thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(img, "{:.1f}mm".format(dimB),
                            (int(x + w/2), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, colorTextB, 2)
                cv2.putText(img, "{:.1f}mm".format(dimA),
                            (int(x + w + 10), int(y + h/2)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, colorTextA, 2)

            cv2.imshow("out", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()