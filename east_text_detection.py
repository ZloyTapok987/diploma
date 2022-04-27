import os, sys
import numpy as np
import cv2
import time
from imutils.object_detection import non_max_suppression


def east_detect(image):
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    (newW, newH) = (256, 256)

    image = cv2.resize(image, (newW, newH))

    orig = image.copy()

    if len(image.shape) == 2:
        gray = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Morph open to remove noise and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        image = 255 - opening

    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height: Should be multiple of 32

    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions

    (H, W) = image.shape[:2]

    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()

    net.setInput(blob)

    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            # Set minimum confidence as required
            if scoresData[x] < 0.99:
                continue
            # compute the offset factor as our resulting feature maps will
            #  x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # loop over the bounding boxes

    if len(boxes) != 1:
        return False, None, None, None

    sx = 0
    sy = 0
    ex = newW
    ey = newH
    logo_part = None
    text_part = None
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image

        x0 = sx
        y0 = sy
        x1 = endX
        y1 = ey
        square = (x1-x0)*(y1-y0)
        logo_part = image[sy:ey, x1:ex]


        if (ex-startX)*(ey-sy) < square:
            square = (ex-startX)*(ey-sy)
            logo_part = image[sy:ey, sx:startX]
            x0 = startX
            y0 = sy
            x1 = ex
            y1 = ey

        if (ex-sx)*(endY-sy) < square:
            square = (ex-sx)*(endY-sy)
            logo_part = image[endY:ey, sx:ex]
            x0 = sx
            y0 = sy
            x1 = ex
            y1 = endY

        if (ex-sx)*(ey-startY) < square:
            logo_part = image[sx:startY, sx:ex]
            square = (ex-sx)*(ey-startY)
            x0 = sx
            y0 = startY
            x1 = ex
            y1 = ey

        if square > (newW * newH) / 2:
            return False, None, None, None

        text_part = image[y0:y1, x0:x1]
       # cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        #cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    print(time.time() - start)
    return True, image, logo_part, text_part


svg_dir = "/Users/dkochergin/Downloads/LLD-logo-files"

logo_out_dir = "logos_out"
text_out_dir = "text_out"

count = 0
for file in os.listdir(svg_dir):
    filename = os.path.join(svg_dir, file)
    to = os.path.join(logo_out_dir, file)

    count += 1

    if count % 10000 == 0:
        print(count)

    if os.path.isfile(to):
        continue

    image = cv2.imread(filename)

    f, out_image, logo, text = east_detect(image)
    if out_image is None:
        continue

    cv2.imwrite(os.path.join(logo_out_dir, file), logo)
    cv2.imwrite(os.path.join(text_out_dir, file), text)

    #cv2.imshow("kek", out_image)
    #cv2.waitKey(0)
    #cv2.imshow("logo", logo)
    #cv2.waitKey(0)
    #cv2.imshow("text", text)
    #cv2.waitKey(0)