import argparse
import mmcv
import cv2 as cv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

def decode_detections(detections, conf_t=0.5):
    results = []
    for detection in detections:
        confidence = detection[4]

        if confidence > conf_t:
            left, top, right, bottom = detection[:4]
            results.append(((int(left), int(top), int(right), int(bottom)), confidence))

    return results

def draw_detections(frame, detections):
    """Draws detections and labels"""
    for i, rect in enumerate(detections):
        left, top, right, bottom = rect[0]
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        label = str('face') + '(' + str(round(rect[1], 2)) + ')'
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame

def main():
    parser = argparse.ArgumentParser(description='Face datection live demo script')
    parser.add_argument('--cam_id', type=int, default=0, help='Input cam')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--fd_thresh', type=float, default=0.5, help='Threshold for FD')

    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # construct the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, args.checkpoint)

    cap = cv.VideoCapture(args.cam_id)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    while cv.waitKey(1) != 27:
        has_frame, frame = cap.read()
        if not has_frame:
            return
        result = inference_detector(model, frame, cfg)
        boxes = decode_detections(result[0], args.fd_thresh)
        frame = draw_detections(frame, boxes)
        cv.imshow('Detection Demo', frame)

if __name__ == '__main__':
    main()
