import torch
import torchvision
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics import SAM
from segment_anything import sam_model_registry, SamPredictor
import io
import logging


class AIProcessor:
    def __init__(self, yolo_path: str, sam_path: str, device: torch.device = torch.device("cpu")):
        self.yolo_path = yolo_path
        self.sam_path = sam_path
        self.device = device
        self.yolo_model = YOLO(self.yolo_path)  # yolo 로드
        self.sam_model = SAM(self.sam_path)  # None
        self.predictor = None
        self.indices = None
        self.alpha_channel = None
        self.size = 0
        self.height = 0
        self.width = 0

    def remove_points_in_bboxes(self, point_list, label_list, bbox_list):
        def is_point_in_bbox(p, b):
            x, y = p
            x_min, y_min, x_max, y_max = b
            return x_min <= x <= x_max and y_min <= y <= y_max

        filtered_p = []
        filtered_l = []
        for p, l in zip(point_list, label_list):
            if not any(is_point_in_bbox(p, b) for b in bbox_list):
                filtered_p.append(p)
                filtered_l.append(l)

        return filtered_p, filtered_l

    def remove_alpha(self, image):
        self.height, self.width = image.shape[:2]
        self.size = self.height * self.width
        if image.shape[2] == 4:  # RGBA or RGB
            self.alpha_channel = image[:, :, 3]
            image_rgb = image[:, :, :3]
        else:
            image_rgb = image

        return image_rgb

    def load_sam_model(self, model_type="vit_h"):  # sam 로드
        self.sam_model = sam_model_registry[model_type](checkpoint=self.sam_path)
        self.sam_model.to(self.device)
        self.predictor = SamPredictor(self.sam_model)

    def object_detection(self, img):
        results = self.yolo_model.predict(source=img, imgsz=640, device=self.device,
                                          iou=0.3, conf=0.3)
        bbox = results[0].boxes.xyxy.tolist()
        self.indices = [index for index, value in enumerate(results[0].boxes.cls) if value == 1.0]
        logging.info(f'객체 탐지 - {self.indices} 박스에 동그라미 존재')
        return bbox

    def segment_from_boxes(self, image, bbox, save_path=None):
        '''
        input_boxes = torch.tensor(bbox, device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        '''
        results = self.sam_model.predict(source=image, bboxes=bbox)
        mask_boxes = results[0].masks.data

        masks_np = np.zeros(mask_boxes.shape[-2:], dtype=np.uint8)
        for i, mask in enumerate(mask_boxes):
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # True는 255, False는 0으로 변환
            mask_np = mask_np.squeeze()
            if i in self.indices:
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Adjust size as needed
                kernel = np.ones((15, 15), np.uint8)
                mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_GRADIENT, kernel)
            masks_np = cv2.bitwise_or(masks_np, mask_np)
            # cv2.imwrite(f'mask_box{i}.jpg', masks_np)

        # cv2.imwrite(save_path, masks_np)
        logging.info(f'1차 마스킹 - 바운딩 박스 {len(mask_boxes)}개 세그먼트 완료.')
        return masks_np

    def segment_from_points(self, image, user_points, user_labels, bbox,  save_path=None):
        '''input_point = np.array(user_points)
        input_label = np.array(user_labels)

        masks_points, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )'''
        # filtered_points, filtered_labels = self.remove_points_in_bboxes(user_points, user_labels, bbox)
        # logging.info(f'2차 마스킹 - 사용자 입력 필터링: from {len(user_points)}개 to {len(filtered_points)}개')

        # results = self.sam_model.predict(source=image, points=filtered_points, labels=filtered_labels)
        results = self.sam_model.predict(source=image, bboxes=[user_points])
        mask_points = results[0].masks.data

        masks_np = np.zeros(mask_points.shape[-2:], dtype=np.uint8)
        for mask in mask_points:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            mask_np = mask_np.squeeze()
            masks_np = cv2.bitwise_or(masks_np, mask_np)

        # cv2.imwrite(save_path, mask_points_uint8)
        logging.info(f'2차 마스킹 - 사용자 입력에 대해 {len(mask_points)}개 영역으로 세그먼트 완료.')
        return masks_np

    def inpainting(self, image, mask_total):
        inpainted_image = cv2.inpaint(image.copy(), mask_total, 15, cv2.INPAINT_TELEA)

        '''inpainted_image = image.copy()
        inpainted_image[mask_total == 255] = [255, 255, 255]'''
        final_image = cv2.convertScaleAbs(inpainted_image, alpha=1.6, beta=10)
        # cv2.imwrite('test_images/inpainted_init.png', inpainted_image)
        # cv2.imwrite('test_images/inpainted_final.png', final_image)
        logging.info('인페인팅 및 후보정 완료.')

        return final_image

    def inpaint_from_points(self, image, user_boxes, save_path=None):
        masks_np = np.zeros(image.shape[:2], dtype=np.uint8)
        for b in user_boxes:  # 박스 내부 다 채우기(마스킹)
            minx, miny, maxx, maxy = b
            masks_np[miny:maxy, minx:maxx] = 255  # 박스 영역을 255로 채움

        # cv2.imwrite(save_path, mask_points_uint8)
        inpainted_image = image.copy()
        inpainted_image[masks_np == 255] = [255, 255, 255]
        # final_image = cv2.convertScaleAbs(inpainted_image, alpha=1.5, beta=10)

        logging.info(f'2차 마스킹 & 인페인팅 - 사용자 입력 {len(user_boxes)}개 영역 인페인팅 완료.')
        return masks_np, inpainted_image

    def combine_alpha(self, image_rgb):
        if self.alpha_channel is not None:  # RGBA
            image_rgba = cv2.merge([image_rgb, self.alpha_channel])
            return image_rgba
        else:
            return image_rgb

    def process(self, img_bytes, user_inputs, user_labels, extension='jpg'):
        """ local test 용도 Vs. server test 용도 구분 """
        ### local 용도
        # img_path = img_bytes
        # image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        ### server 용도
        buffer = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        image_output = image.copy()
        logging.info(f"이미지 처리 시작 - 사이즈: {image.shape[:2]}")

        ### ready
        #self.load_sam_model()
        #self.predictor.set_image(image)
        masks_total = np.zeros(image.shape[:2], dtype=np.uint8)

        ### 1차: Object Detection & Segment by Box
        bbox = self.object_detection(image)
        if len(bbox) > 0:
            logging.info("***** 객체 탐지 세그멘테이션 시작 ******")
            masks_by_yolo = self.segment_from_boxes(image, bbox, save_path=None)  # 'test_images/seg_box.png'
            masks_total = cv2.bitwise_or(masks_total, masks_by_yolo)
            #logging.info( f"1차 마스킹 후 shape 점검: YOLOv11 감지된 영역 shape: {masks_by_box.shape}, 이미지 영역 shape: {image.shape}")  # (1893, 1577, 3)  (1893, 1577)
        else:
            logging.info("***** 객체 탐지 세그멘테이션 스킵 ******")
            masks_by_yolo = None
        ### 2차: points arguments by User & Segment by Points
        if len(user_inputs) > 0:
            logging.info("***** 사용자 입력 세그멘테이션 시작 ******")
            # masks_by_user = self.segment_from_points(image, user_inputs, user_labels, bbox, save_path=None)  # save_path='test_images/seg_points.png'
            # masks_total = cv2.bitwise_or(masks_total, masks_by_user)
            masks_by_user, image_output = self.inpaint_from_points(image, user_inputs, save_path=None)
            _, mask_bytes = cv2.imencode("." + extension, image_output)
        else:
            logging.info("***** 사용자 입력 세그멘테이션 스킵 ******")
            masks_by_user = None
            mask_bytes = None
        # cv2.imwrite('test_images/mask_total.png', masks_total)
        if isinstance(masks_total, np.ndarray):
            image_output = self.inpainting(image_output, masks_total)
            logging.info('***** 인페인팅 수행 완료 ******')
        else:
            logging.info('입력과 동일한 결과 이미지입니다.')

        _, input_bytes = cv2.imencode("." + extension, image)
        # _, mask_bytes = cv2.imencode("." + extension, masks_total.astype(np.uint8))
        _, result_bytes = cv2.imencode("." + extension, image_output)

        if masks_by_yolo is not None:
            _, mask_by_yolo_bytes = cv2.imencode("." + extension, masks_by_yolo.astype(np.uint8))
        else:
            mask_by_yolo_bytes = None
        if masks_by_user is not None:
            _, mask_by_user_bytes = cv2.imencode("." + extension, masks_by_user.astype(np.uint8))
        else:
            mask_by_user_bytes = None
        return (io.BytesIO(input_bytes), io.BytesIO(mask_bytes), io.BytesIO(result_bytes),
                io.BytesIO(mask_by_yolo_bytes), io.BytesIO(mask_by_user_bytes))
