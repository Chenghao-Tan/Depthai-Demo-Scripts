from time import monotonic

import cv2
import depthai as dai
import numpy as np

imgs_path = "./dataset/imgs"
masks_path = "./dataset/masks"
blob = dai.OpenVINO.Blob("E:/Desktop/GSoC/boat/models/DDRNet/op.blob")
for name, tensorInfo in blob.networkInputs.items():
    print(name, tensorInfo.dims)
INPUT_SHAPE = blob.networkInputs["0"].dims[:2]


# Start defining a pipeline
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale((1, 3), (1, 3))
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam.setPreviewSize(*INPUT_SHAPE)
cam.setInterleaved(False)

# XLinkIn
xinFrame = pipeline.create(dai.node.XLinkIn)
xinFrame.setStreamName("inFrame")

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlob(blob)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)
# cam.preview.link(detection_nn.input)
xinFrame.out.link(detection_nn.input)

# NN output linked to XLinkOut
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)


def cal_iou(input, target):
    inter = np.dot(input.flatten(), target.flatten())
    total = np.sum(input) + np.sum(target) - inter
    return inter / total


# Pipeline is defined, now we can connect to the device
with dai.Device() as device:
    device.startPipeline(pipeline)
    q_in = device.getInputQueue(name="inFrame")  # type: ignore
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)  # type: ignore
    for i in range(1, 1326):
        pic = cv2.imread(imgs_path + "/({}).jpg".format(i))
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic, tuple(INPUT_SHAPE))
        pic = np.array(pic)
        msk = cv2.imread(masks_path + "/({}).png".format(i))
        msk = cv2.resize(msk, tuple(INPUT_SHAPE))
        msk = np.array(msk)
        msk = np.where(msk > 0, 0, 1)  # type: ignore

        img = dai.ImgFrame()
        img.setData(pic.transpose(2, 0, 1).flatten())
        img.setTimestamp(monotonic())  # type: ignore
        img.setWidth(INPUT_SHAPE[0])
        img.setHeight(INPUT_SHAPE[1])
        q_in.send(img)

        msgs = q_nn.get()
        layer1 = msgs.getFirstLayerFp16()
        frame = np.asarray(layer1).reshape(INPUT_SHAPE[1], INPUT_SHAPE[0])
        iou = cal_iou(frame, msk[:, :, 0])

        frame = (frame > 0.5).astype(np.uint8) * 255
        msk = msk.astype(np.uint8) * 255
        frame = np.concatenate((pic, np.stack((frame,) * 3, axis=2), msk), axis=1)

        cv2.putText(
            frame,
            "IoU: {:.2f}".format(iou),
            (2, frame.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.4,
            color=(255, 255, 255),
        )
        cv2.imshow("Frame", frame)

        if cv2.waitKey(0) == ord("q"):
            break
