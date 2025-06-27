import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Tuple, Optional, List
from numpy import log
from tqdm import tqdm


class PeopleDetector:
    """Video processor, that detects and annotates people on an input video. Use m, s or n as model sizes for less accuracy and faster work"""

    def __init__(self, model_size: str = 'x', multi_scale: bool = True):
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.multi_scale = multi_scale

    def validate_paths(self, input_path: str, output_path: str) -> Tuple[bool, str]:
        """Validate input and output paths."""
        if not Path(input_path).exists():
            return False, f"Input file {input_path} not found"
        if not os.access(input_path, os.R_OK):
            return False, f"No read permissions for {input_path}"

        output_dir = os.path.dirname(output_path) or '.'
        os.makedirs(output_dir, exist_ok=True)

        if not os.access(output_dir, os.W_OK):
            return False, f"No write permissions for {output_dir}"
        return True, "Paths validated"

    def perspective_aware_nms(self, boxes: np.ndarray, confs: np.ndarray, img_height: int) -> List[int]:
        """Adaptive NMS that considers object position in perspective."""
        if len(boxes) == 0:
            return []

        # Calculate adaptive thresholds
        box_centers = (boxes[:, 1] + boxes[:, 3]) / 2
        position_factors = box_centers / img_height
        iou_thresholds = 0.4 + position_factors * 0.3

        # Standard NMS implementation
        keep = []
        order = confs.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / ((boxes[i, 2]-boxes[i, 0])*(boxes[i, 3]-boxes[i, 1]) +
                           (boxes[order[1:], 2]-boxes[order[1:], 0])*(boxes[order[1:], 3]-boxes[order[1:], 1]) - inter)

            mask = iou <= iou_thresholds[order[1:]]
            order = order[1:][mask]

        return keep

    def detect_people(self, frame: np.ndarray) -> np.ndarray:
        """Detects people on a frame"""
        if self.multi_scale:
            results = self.model(frame, classes=[0], imgsz=[
                                 640, 1280], verbose=False)
        else:
            results = self.model(frame, classes=[0], verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                if int(cls) == 0:  # person class
                    # Confidence boost for small objects
                    area = (box[2]-box[0])*(box[3]-box[1])
                    if area < 1000:
                        conf = min(1.0, conf * (1 + 0.5*log(1000/area)))
                    detections.append([*box, conf])

        return np.array(detections) if detections else np.empty((0, 5))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detects people and annotates the frame"""
        detections = self.detect_people(frame)

        if len(detections) > 0:
            keep = self.perspective_aware_nms(
                detections[:, :4],
                detections[:, 4],
                frame.shape[0]
            )
            detections = detections[keep]

        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2, conf = det
            # using clear green (BGR)
            color = (0, 255, 0)
            alpha = 0.7  # transparency

            # adding box and text
            overlay = annotated.copy()
            cv2.rectangle(overlay, (int(x1), int(y1)),
                          (int(x2), int(y2)), color, 2)
            cv2.putText(overlay, f"Person: {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # applying with transparency
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        return annotated

    def process_video(self, input_path: str, output_path: str, show_preview: bool = False) -> Optional[str]:
        """Processes the whole video"""
        is_valid, msg = self.validate_paths(input_path, output_path)
        if not is_valid:
            print(f"Error: {msg}", file=sys.stderr)
            return None

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}", file=sys.stderr)
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(
                f"Error: Could not create output file {output_path}", file=sys.stderr)
            cap.release()
            return None

        try:
            with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    annotated = self.process_frame(frame)
                    out.write(annotated)
                    pbar.update(1)

                    if show_preview:
                        cv2.imshow('People Detection', annotated)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        except Exception as e:
            print(f"\nError during processing: {e}", file=sys.stderr)
            return None
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        return output_path


if __name__ == "__main__":
    detector = PeopleDetector(model_size='x', multi_scale=True)
    result = detector.process_video(
        input_path='crowd.mp4',
        output_path='output.mp4',
        show_preview=False
    )
