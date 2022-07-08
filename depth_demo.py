import time

import cv2
import depthai as dai
import numpy as np

slice = 10
assert slice > 1
mean = True
fps = 30
confidence = 255
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
extended_disparity = True
subpixel = False

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth")

left.setResolution(monoResolution)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setFps(fps)
right.setResolution(monoResolution)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setFps(fps)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.initialConfig.setConfidenceThreshold(confidence)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(extended_disparity)
stereo.setSubpixel(subpixel)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.temporalFilter.enable = False
config.postProcessing.spatialFilter.enable = False
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)

# Linking
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.depth.link(depthOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Configure windows
    depthWindowName = "depth"
    cv2.namedWindow(depthWindowName)

    timer = time.time()
    while True:
        queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        packet = queue.get()
        frameDepth = packet.getFrame()
        print("FPS: {:.1f}".format(1 / (time.time() - timer)))
        timer = time.time()

        frameDepthOut = (
            (frameDepth - frameDepth.min())
            / (frameDepth.max() - frameDepth.min())
            * 255
        ).astype(np.uint8)
        frameDepthOut = cv2.applyColorMap(frameDepthOut, cv2.COLORMAP_JET)
        frameH = frameDepthOut.shape[0]
        frameW = frameDepthOut.shape[1]

        for i in range(1, slice):
            cv2.line(
                frameDepthOut,
                (int(frameW * i / slice), 0),
                (int(frameW * i / slice), frameH - 1),
                (0, 0, 255),
                3,
            )
            cv2.line(
                frameDepthOut,
                (0, int(frameH * i / slice)),
                (frameW - 1, int(frameH * i / slice)),
                (0, 0, 255),
                3,
            )

        for i in range(1, slice + 1):
            for j in range(1, slice + 1):
                roi = frameDepth[
                    int(frameH * (i - 1) / slice) : int(frameH * i / slice),
                    int(frameW * (j - 1) / slice) : int(frameW * j / slice),
                ]
                inRange = (200 <= roi) & (roi <= 30000)
                roi = roi[inRange]
                if len(roi) == 0:
                    continue
                dis = round(roi.mean() / 1000 if mean else roi.min() / 1000, 2)
                cv2.putText(
                    frameDepthOut,
                    str(dis),
                    (int(frameW * (j - 1) / slice) + 3, int(frameH * i / slice) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    1,
                )

        cv2.imshow(depthWindowName, frameDepthOut)

        if cv2.waitKey(1) == ord("q"):
            break
