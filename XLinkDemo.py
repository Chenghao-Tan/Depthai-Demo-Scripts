from time import monotonic

import cv2
import depthai as dai
import numpy as np

calculate_only = False  # TODO NO imshow() IF True
imgs_path = "./dataset/imgs"  # TODO IMAGE PATH -> (x).jpg
masks_path = "./dataset/masks"  # TODO MASK PATH -> (x).png
blob = dai.OpenVINO.Blob("./models/DDRNet/640_360.blob")  # TODO MODEL PATH
threshold = 0.5  # TODO CONFIDENCE THRESHOLD (0.-1.)
assert threshold >= 0 and threshold <= 1

for name, tensorInfo in blob.networkInputs.items():
    print(name, tensorInfo.dims)
INPUT_SHAPE = blob.networkInputs["0"].dims[:2]


# Start defining a pipeline
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale((1, 3), (1, 3))  # TODO RGB->640x360
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


def iou_rec_pre(input, target, e=1e-6):
    inter = np.dot(input.flatten(), target.flatten()) + e
    i_sum = np.sum(input) + e
    t_sum = np.sum(target) + e
    total = i_sum + t_sum - inter - e
    return inter / total, inter / t_sum, inter / i_sum


# Pipeline is defined, now we can connect to the device
with dai.Device() as device:
    device.startPipeline(pipeline)
    q_in = device.getInputQueue(name="inFrame")  # type: ignore
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)  # type: ignore

    iou_avg = 0
    rec_avg = 0
    pre_avg = 0
    count = 0
    for i in range(1, 1326):  # TODO x in (x).jpg/png
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
        iou, rec, pre = iou_rec_pre(frame, msk[:, :, 0])
        iou_avg += iou
        rec_avg += rec
        pre_avg += pre
        count += 1

        if not calculate_only:
            frame = (frame > threshold).astype(np.uint8) * 255
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

    print("average IoU: {:.2f}".format(iou_avg / count))
    print("average recall: {:.2f}".format(rec_avg / count))
    print("average precision: {:.2f}".format(pre_avg / count))
