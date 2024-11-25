import torch
import torchvision
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics import SAM
# from segment_anything import sam_model_registry, SamPredictor
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
        # self.sam_model = sam_model_registry[model_type](checkpoint=self.sam_path)
        # self.sam_model.to(self.device)
        # self.predictor = SamPredictor(self.sam_model)
        logging.info("SAM model cannot be loaded")

    def object_detection(self, img):
        results = self.yolo_model.predict(source=img, imgsz=640, device=self.device,
                                          iou=0.3, conf=0.3)
        bbox = results[0].boxes.xyxy.tolist()
        self.indices = [index for index, value in enumerate(results[0].boxes.cls) if value == 1.0]
        logging.info(f'객체 탐지 - {self.indices} 박스에 동그라미 존재')
        return bbox

    def segment_from_yolo(self, image, bbox, save_path=None):
        results = self.sam_model.predict(source=image, bboxes=bbox)
        mask_boxes = results[0].masks.data

        masks_np = np.zeros(mask_boxes.shape[-2:], dtype=np.uint8)
        for i, mask in enumerate(mask_boxes):
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # True는 255, False는 0으로 변환
            mask_np = mask_np.squeeze()
            '''if i in self.indices:  # 원형 마킹이라면
                mask_np = cv2.dilate(mask_np, np.ones((3, 3), np.uint8), iterations=1)
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Adjust size as needed
                # kernel = np.ones((15, 15), np.uint8)
                # mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_GRADIENT, kernel)'''
            masks_np = cv2.bitwise_or(masks_np, mask_np)

        logging.info(f'1차 마스킹 - 바운딩 박스 {len(mask_boxes)}개 세그먼트 완료.')
        return masks_np

    def segment_from_user(self, image, user_inputs, bbox,  save_path=None):
        # filtered_points, filtered_labels = self.remove_points_in_bboxes(user_points, user_labels, bbox)
        # logging.info(f'2차 마스킹 - 사용자 입력 필터링: from {len(user_points)}개 to {len(filtered_points)}개')

        # results = self.sam_model.predict(source=image, points=filtered_points, labels=filtered_labels)
        results = self.sam_model.predict(source=image, bboxes=user_inputs)
        mask_points = results[0].masks.data

        masks_np = np.zeros(mask_points.shape[-2:], dtype=np.uint8)
        for mask in mask_points:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            mask_np = mask_np.squeeze()
            masks_np = cv2.bitwise_or(masks_np, mask_np)

        logging.info(f'2차 마스킹 - 사용자 입력에 대해 {len(mask_points)}개 영역으로 세그먼트 완료.')
        return masks_np

    def inpaint_from_yolo(self, image, bbox=None):
        inpainted_image = image.copy()
        masks_np = np.zeros(image.shape[:2], dtype=np.uint8)
        text_np = np.zeros(image.shape[:2], dtype=np.uint8)  # 전체 roi_mask 저장
        for b in bbox:
            minx, miny, maxx, maxy = map(int, b)
            masks_np[miny:maxy, minx:maxx] = 255  # 박스 영역을 255로 채움

            roi = image[miny:maxy, minx:maxx]  # 박스 내 복사하고 텍스트만 추출하기
            roi_mask = cv2.inRange(roi, (0, 0, 0), (40, 40, 45))  # roi 내에서 인쇄된 글씨는 255값
            text_np[miny:maxy, minx:maxx] = roi_mask

            if maxy < image.shape[0]-2:  # 박스 근처 BG 컬러 샘플링
                sample_color = image[maxy + 1, (minx+maxx)//2].tolist()  # 박스 하단 모서리 중앙 아래 픽셀값 샘플링
            else:
                sample_color = [255, 255, 255]  # 경계에 있을 경우, 기본 흰색
            inpainted_image[miny:maxy, minx:maxx] = sample_color  # 인페인팅
            logging.info(f'1차 인페인팅 - yolo 박스 샘플링 컬러 rgb: {sample_color}')

        # 복원 수행
        text_np = cv2.dilate(text_np, np.ones((4, 4), np.uint8), iterations=1)
        text_np = cv2.erode(text_np, np.ones((2, 2), np.uint8), iterations=2)

        inpainted_image[text_np == 255] = [30, 30, 30]
        # inpainted_image = cv2.inpaint(inpainted_image, mask_total, 10, cv2.INPAINT_TELEA)

        return text_np, masks_np, inpainted_image

    def inpaint_from_user(self, image, user_boxes, save_path=None):
        inpainted_image = image.copy()
        masks_np = np.zeros(image.shape[:2], dtype=np.uint8)  # 박스 위치 마스킹
        for b in user_boxes:
            minx, miny, maxx, maxy = map(int, b)
            masks_np[miny:maxy, minx:maxx] = 255  # 박스 영역을 255로 채움
            if minx > 2:  # 박스 근처 BG 컬러 샘플링
                sample_color = image[(miny + maxy) // 2, minx-2].tolist()  # 우측 상단 모서리 바로 외부의 픽셀
            else:
                sample_color = [255, 255, 255]  # 경계에 있을 경우, 기본 흰색
            inpainted_image[miny:maxy, minx:maxx] = sample_color  # 인페인팅
            logging.info(f'2차 인페인팅 - user 박스 샘플링 컬러 RGB: {sample_color}')

        return masks_np, inpainted_image

    def combine_alpha(self, image_rgb):
        if self.alpha_channel is not None:  # RGBA
            image_rgba = cv2.merge([image_rgb, self.alpha_channel])
            return image_rgba
        else:
            return image_rgb

    def process(self, img_bytes, user_inputs, extension='jpg'):
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

        ### 1차: Segment by Object Detection
        bbox = self.object_detection(image)
        if len(bbox) > 0:
            logging.info("***** 1차: 객체 탐지 세그멘테이션 시작 ******")
            # masks_by_yolo = self.segment_from_yolo(image, bbox, save_path=None)  # 'test_images/seg_box.png'
            recovery, masks_by_yolo, image_output = self.inpaint_from_yolo(image_output, bbox)
            masks_total = cv2.bitwise_or(masks_total, masks_by_yolo)
            logging.info('***** 1차: 객차 탐지 인페인팅 수행 완료 ******')
            # cv2.imwrite('inpainted_yolo_복원.jpg', recovery.astype(np.uint8))  # 인식 및 복원하는 문항
            # cv2.imwrite('inpainted_yolo_결과.jpg', image_output)  # 1차 마스킹 및 인페인팅 결과
        else:
            logging.info("***** 1차: 객체 탐지 세그멘테이션 스킵 ******")
            masks_by_yolo = None

        ### 2차: Segment by User Prompt
        if len(user_inputs) > 0:
            logging.info("***** 2차: 사용자 입력 세그멘테이션 시작 ******")
            # masks_by_user = self.segment_from_user(image, user_inputs, bbox, save_path=None)  # save_path='test_images/seg_points.png'
            masks_by_user, image_output = self.inpaint_from_user(image_output, user_inputs)
            masks_total = cv2.bitwise_or(masks_total, masks_by_user)
            logging.info('***** 2차: 사용자 입력 인페인팅 수행 완료 ******')
            # cv2.imwrite('inpainted_yolo&user_결과.jpg', image_output)
        else:
            logging.info("***** 2차: 사용자 입력 세그멘테이션 스킵 ******")
            masks_by_user = None

        image_output = cv2.convertScaleAbs(image_output, alpha=1.5, beta=10)
        logging.info("***** 필기 제거 결과 후보정 완료 ******")

        # 5가지 bytes 생성
        _, input_bytes = cv2.imencode("." + extension, image)
        _, result_bytes = cv2.imencode("." + extension, image_output)
        if masks_by_yolo is not None:
            mask_by_yolo_img = image.copy()
            mask_by_yolo_img[masks_by_yolo == 255] = [0, 0, 0]
            _, mask_by_yolo_bytes = cv2.imencode("." + extension, mask_by_yolo_img)
        else:
            mask_by_yolo_bytes = None
        if masks_by_user is not None:
            mask_by_user_img = image.copy()
            mask_by_user_img[masks_by_user == 255] = [0, 0, 0]
            _, mask_by_user_bytes = cv2.imencode("." + extension, mask_by_user_img)
        else:
            mask_by_user_bytes = None
        mask_total_img = image.copy()
        mask_total_img[masks_total == 255] = [0, 0, 0]
        _, mask_bytes = cv2.imencode("." + extension, mask_total_img)

        # cv2.imwrite('input.jpg', image)
        # cv2.imwrite('mask_total.jpg', mask_total_img)
        # cv2.imwrite('output.jpg', image_output)

        return (io.BytesIO(input_bytes), io.BytesIO(mask_bytes), io.BytesIO(result_bytes),
                io.BytesIO(mask_by_yolo_bytes), io.BytesIO(mask_by_user_bytes))
