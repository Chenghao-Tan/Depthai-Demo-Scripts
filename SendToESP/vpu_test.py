import time

import cv2
import depthai as dai
import numpy as np

from pipeline import create_pipeline


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


with dai.Device(create_pipeline()) as device:
    nnQueue = device.getOutputQueue(name="nn", maxSize=2, blocking=False)  # type: ignore
    imgQueue = device.getOutputQueue(name="img", maxSize=2, blocking=False)  # type: ignore
    depthQueue = device.getOutputQueue(name="depth", maxSize=2, blocking=False)  # type: ignore

    fps = FPSHandler()
    while True:
        msgs = nnQueue.get()
        img = imgQueue.get().getCvFrame()
        depth = depthQueue.get().getFrame()
        fps.next_iter()

        layer1 = msgs.getLayerFp16("out")
        grids = np.asarray(layer1).reshape(-1, 4)

        depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(
            np.uint8
        )
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

        H = img.shape[0]
        W = img.shape[1]

        weight = 0.5  # TODO
        frame = cv2.addWeighted(img, 1 - weight, depth, weight, 0)
        # frame = np.concatenate((img, depth), axis=0)

        slice = 3  # TODO
        grids = grids.reshape(slice, slice, 4)
        for i in range(1, slice):
            cv2.line(
                frame,
                (int(W * i / slice), 0),
                (int(W * i / slice), H - 1),
                (0, 0, 255),
                3,
            )
            cv2.line(
                frame,
                (0, int(H * i / slice)),
                (W - 1, int(H * i / slice)),
                (0, 0, 255),
                3,
            )
        for i in range(1, slice + 1):
            for j in range(1, slice + 1):
                cv2.putText(
                    frame,
                    "{:d}({:.2f},{:.2f},{:.2f})".format(
                        grids[i][j][0], grids[i][j][1], grids[i][j][2], grids[i][j][3]
                    ),
                    (2, frame.shape[0] - 4),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.4,
                    color=(255, 255, 255),
                )

        cv2.putText(
            frame,
            "Fps: {:.2f}".format(fps.fps()),
            (2, frame.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.4,
            color=(255, 255, 255),
        )
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break
