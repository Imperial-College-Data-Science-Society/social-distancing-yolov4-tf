import cv2
import numpy as np
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from itertools import combinations 
import math
import argparse

mouse_pts = []
def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 10, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 10, (0, 0, 255), 10)
        
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x,y), (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)

if __name__== "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
        help="path to (optional) input video file")
    ap.add_argument("-o1", "--output_human", type=str, default="",
        help="path to (optional) output video file")
    ap.add_argument("-o2", "--output_bev", type=str, default="",
        help="path to (optional) output video file")
    ap.add_argument("-m", "--model", type=str, default=1,
        help="path to output video file")
    ap.add_argument("-d", "--display", type=str, default=1,
        help="whether or not output frame should be displayed")
    args = ap.parse_args()
    print(args)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    saved_model_loaded = tf.saved_model.load(args.model, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    print('yolov4 model loaded')

    vid = cv2.VideoCapture(args.input)
    if args.output_human:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(args.output_human, codec, fps, (width, height))

    frame_id = 0 
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image = Image.fromarray(frame)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")
        if frame_id == 0:
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("image", get_mouse_points)
            image = cv2.imread("result1.png")
            while True:
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 6:
                    cv2.destroyWindow("image")
                    break
            rect = np.array(mouse_pts[:4], dtype = "float32")
            (tl, tr, br, bl) = rect
            print(tl, tr, br, bl)
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")
            M = cv2.getPerspectiveTransform(rect, dst)

            pts = np.float32(np.array([mouse_pts[4:7]]))
            warped_pts = cv2.perspectiveTransform(pts, M)
            warped_pt = warped_pts[0]
            distance_thresh = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
            # distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
            print(distance_thresh)
            cv2.destroyWindow("image") 
            frame_id += 1

            if args.output_bev:
                # by default VideoCapture returns float instead of int
                birdseyeview_fps = int(vid.get(cv2.CAP_PROP_FPS))
                birdseyeview_codec = cv2.VideoWriter_fourcc(*'MJPG')
                birdseyeview = cv2.VideoWriter(args.output_bev, birdseyeview_codec, birdseyeview_fps, (maxWidth, maxHeight))
        else:
            
            warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
            
            input_size = 416
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            prev_time = time.time()

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.25
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            image, bottom_coords = utils.draw_bbox(frame, pred_bbox)
            curr_time = time.time()
            exec_time = curr_time - prev_time
            result = np.asarray(image)
            info = "time: %.2f ms" %(1000*exec_time)
            print(info)

            bottom_coords_pts = np.float32(np.array([bottom_coords]))
            warped_bottom_pts = cv2.perspectiveTransform(bottom_coords_pts, M)
            comb = combinations(warped_bottom_pts[0], 2) 
            violate = set()
            for i in comb:
                dist = math.hypot(abs(i[0][0] - i[1][0]),abs(i[0][1] - i[1][1]))
                if dist < distance_thresh:
                    violate.add(tuple(i[0].astype(int)))
                    violate.add(tuple(i[1].astype(int)))
                    color = (0, 0, 255)
                    cv2.circle(warped, tuple(i[0].astype(int)), 20, color, 2)
                    cv2.circle(warped, tuple(i[1].astype(int)), 20, color, 2)
                    cv2.line(warped, tuple(i[0].astype(int)), tuple(i[1].astype(int)), (70, 70, 70), 2)
                    
                    text_coord = (int(min(i[0][0],i[1][0])+(abs(i[0][0] - i[1][0]) // 2)), int(min(i[0][1],i[1][1])+(abs(i[0][1] - i[1][1])// 2)))
                    cv2.putText(warped, str(dist), text_coord, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, lineType=cv2.LINE_AA)

            for i in warped_bottom_pts[0]:
                color = ( 0, 255, 0)
                if tuple(i.astype(int)) not in list(violate):
                    cv2.circle(warped, tuple(i.astype(int)), 20, color, 2)

            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if args.display == 'True':
                cv2.namedWindow("image", cv2.WINDOW_NORMAL)    
                cv2.imshow("image", warped)

                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            if args.output_human:
                out.write(result)
            if args.output_bev:
                birdseyeview.write(warped)
            frame_id += 1
