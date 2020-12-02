import argparse
import cv2
import numpy as np
import os
from openvino.inference_engine import IECore

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", required = True, type = str, help = "The location of model .xml file")
    parser.add_argument("-i", "--input", required = True, type = str, help = "The path to input image")
    parser.add_argument("-d", "--device", default = "CPU", type = str, help = "To specify target device for inference")
    parser.add_argument("-pt", "--prob_threshold", default = 0.6, type = float, help = "Probability threshold for filters detection")

    args = parser.parse_args()

    return args

def infer_on_image(args):

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    ie = IECore()
    net = ie.read_network(model = model_xml, weights = model_bin)
    exec_net = ie.load_network(network = net, device_name = args.device, num_requests = 1)

    input_blob = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_blob].shape

    output_blob = next(iter(net.outputs))
    
    input_img = cv2.imread(args.input)
    height, width, channel = input_img.shape
    
    # Preprocessing
    p_frame = cv2.resize(input_img, (w, h))
    p_frame = p_frame.transpose((2, 0, 1))
    p_frame = p_frame.reshape((n, c, h, w))

    # Perform inference
    exec_net.infer({input_blob: p_frame})

    # Extract the output
    result = exec_net.requests[0].outputs[output_blob]

    count = 0
    for obj in result[0][0]:
        conf = obj[2]
        if conf >= args.prob_threshold:
            count += 1
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)

            cv2.rectangle(input_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.imshow("test_img", input_img)
    
    cv2.waitKey(0)

    print("Detected {} cars".format(count))

def main():
    args = get_args()
    infer_on_image(args)

if __name__ == "__main__":
    main()
