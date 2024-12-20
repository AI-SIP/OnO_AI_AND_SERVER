import logging
import cv2
import numpy as np
import io


def rgb_to_hsv(r, g, b):
    color = np.uint8([[[r, g, b]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color


def rgb_to_hsv_list(rgb_list):
    colors = np.uint8([[list(rgb) for rgb in rgb_list]])
    hsv_colors = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV)
    return hsv_colors.reshape(-1, 3)


class ColorRemover:
    def __init__(self, target_rgb_list, intensity):  # 팀 주정뱅이: 30, 150, 150
        self.size = 0
        self.height = 0
        self.width = 0

        self.target_rgb_list = target_rgb_list
        self.target_hsv_list = None
        self.pencil_tolerance = None
        self.color_tolerance = None
        self.intensity = intensity

        self.alpha_channel = None
        self.masks = None

    def combine_alpha(self, image_rgb):
        if self.alpha_channel is not None:  # RGBA
            image_rgba = cv2.merge([image_rgb, self.alpha_channel])
            return image_rgba
        else:
            return image_rgb

    def masking(self, image_rgb):  # important
        image_mask = image_rgb.copy()
        image_hsv = cv2.cvtColor(image_mask, cv2.COLOR_BGR2HSV)

        self.masks = np.zeros(image_hsv.shape[:2], dtype=np.uint8)
        self.target_hsv_list = rgb_to_hsv_list(self.target_rgb_list)
        if self.intensity == 0:  # 강도 약하게
            self.color_tolerance = (5, 180, 160)  # 튀는 색상펜 특화
            self.sharp_tolerance = (150, 30, 60)  # 샤프 특화
            lower_min_v = 50
        elif self.intensity == 1:  # 강도 중간
            self.color_tolerance = (15, 180, 160)  # 튀는 색상펜 특화
            self.sharp_tolerance = (150, 30, 60)  # 샤프 특화
            lower_min_v = 30
        else:  # 강도 강하게
            self.color_tolerance = (30, 180, 160)  # 튀는 색상펜 특화
            self.sharp_tolerance = (150, 40, 90)  # 샤프 특화
            lower_min_v = 10

        for target_hsv in self.target_hsv_list:
            circle_mask = None
            if target_hsv[1] <= 25:  # 색상의 채도가 낮다면 -> 펜슬 간주
                lower_bound = np.array([max(0, target_hsv[0] - self.sharp_tolerance[0]),
                                        max(0, target_hsv[1] - self.sharp_tolerance[1]),
                                        max(lower_min_v, target_hsv[2] - self.sharp_tolerance[2])])
                upper_bound = np.array([min(180, target_hsv[0] + self.sharp_tolerance[0]),
                                        min(30, target_hsv[1] + self.sharp_tolerance[1]),
                                        min(150, target_hsv[2] + self.sharp_tolerance[2])])
            else:  # 색상의 채도가 있다면 -> 색상펜 간주
                lower_bound = np.array([max(0, target_hsv[0] - self.color_tolerance[0]),
                                        max(20, target_hsv[1] - self.color_tolerance[1]),
                                        max(lower_min_v, target_hsv[2] - self.color_tolerance[2])])
                upper_bound = np.array([min(180, target_hsv[0] + self.color_tolerance[0]),
                                        min(255, target_hsv[1] + self.color_tolerance[1]),
                                        min(255, target_hsv[2] + self.color_tolerance[2])])
                lower_hue = target_hsv[0] - self.color_tolerance[0]
                upper_hue = target_hsv[0] + self.color_tolerance[0]
                if lower_hue < 0:  # 빨간색 순환처리
                    circle_lower_bound = lower_bound.copy()
                    circle_upper_bound = upper_bound.copy()
                    circle_lower_bound[0] = (target_hsv[0] - self.color_tolerance[0] + 180) % 180
                    circle_upper_bound[0] = 180
                    circle_mask = cv2.inRange(image_hsv, circle_lower_bound, circle_upper_bound)
                elif upper_hue > 180:
                    circle_lower_bound = lower_bound.copy()
                    circle_upper_bound = upper_bound.copy()
                    circle_lower_bound[0] = 0
                    circle_upper_bound[0] = (target_hsv[0] + self.color_tolerance[0] - 180) % 180
                    circle_mask = cv2.inRange(image_hsv, circle_lower_bound, circle_upper_bound)

            temp_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
            logging.info("target_hsv: %s, from: %s  to: %s", target_hsv, lower_bound, upper_bound)
            if circle_mask is not None:
                temp_mask = cv2.bitwise_or(temp_mask, circle_mask)
                logging.info("+ red target_hsv: %s, from: %s  to: %s", target_hsv, circle_lower_bound, circle_upper_bound)
            self.masks = cv2.bitwise_or(self.masks, temp_mask)

        # Opening 적용
        '''kernel = np.ones((3, 3), np.uint8)  # mask 내 노이즈 제거
        self.masks = cv2.morphologyEx(self.masks, cv2.MORPH_OPEN, kernel)'''

        # Dilation 적용
        '''kernel = np.ones((1, 1), np.uint8)
        if self.masks is not None:
            self.masks = cv2.dilate(self.masks, kernel, iterations=2)'''

    def inpainting(self, image_rgb):
        if self.masks is not None and isinstance(self.masks, np.ndarray):
            inpainted_image = image_rgb.copy()
            inpainted_image = cv2.inpaint(inpainted_image, self.masks, 10, cv2.INPAINT_TELEA)
            # inpainted_image = cv2.inpaint(inpainted_image, self.masks, 2, cv2.INPAINT_TELEA)
            # inpainted_image[self.masks != 0] = [255, 255, 255]
            return inpainted_image
        else:
            raise ValueError("Mask is not properly defined or is not a numpy array.")

    def remove_alpha(self, image):
        self.height, self.width = image.shape[:2]
        self.size = self.height * self.width
        if image.shape[2] == 4:  # RGBA or RGB
            self.alpha_channel = image[:, :, 3]
            image_rgb = image[:, :, :3]
        else:
            image_rgb = image

        return image_rgb

    def process(self, img_bytes, extension):
        # Ensure the bytes are in a buffer format that OpenCV can understand
        buffer = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)

        image_rgb = self.remove_alpha(img)
        # image_rgb = self.scaling(image_rgb)  # perspective transforming

        if len(self.target_rgb_list) != 0:
            self.masking(image_rgb)  # masking
            image_inpainted = self.inpainting(image_rgb)  # inpainting
            # image_input = self.background_filtering(image_rgb, extension)  # input filtering
            # image_output = self.background_filtering(image_inpainted, extension)  # output filtering
            image_output = self.filtering(image_inpainted)  # output filtering
        else:
            self.masks = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            image_output = self.filtering(image_rgb)

        image_input = self.combine_alpha(image_rgb)
        image_output = self.combine_alpha(image_output)

        _, input_bytes = cv2.imencode("." + extension, image_input)
        _, mask_bytes = cv2.imencode("." + extension, self.masks.astype(np.uint8))
        _, result_bytes = cv2.imencode("." + extension, image_output)

        return io.BytesIO(input_bytes), io.BytesIO(mask_bytes), io.BytesIO(result_bytes)


    def scaling(self, img):
        # buffer = np.frombuffer(img_bytes, dtype=np.uint8)
        # img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray scale로 변환
        gray_image = cv2.equalizeHist(gray_image)
        gray_image = cv2.GaussianBlur(gray_image, (15, 15), 0)
        gray_image = cv2.bilateralFilter(gray_image, 9, 75, 75)  # 엣지 보존 필터 적용
        gray_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 115, 4)  # 적응형 임계값 적용
        gray_image = cv2.medianBlur(gray_image, 11)  # 중앙값 필터 적용
        gray_image = cv2.copyMakeBorder(gray_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # 이미지 테두리 추가

        edges = cv2.Canny(gray_image, 100, 200)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        # logging.info(f"contours found: {contours}")
        contour_image = img.copy()
        # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 5)  # 상위 5개 컨투어를 녹색으로 그림

        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            logging.warning("No contour found")
            newImage = img  # 변환에 실패한 경우 원본 이미지 반환
        else:
            cv2.drawContours(contour_image, [screenCnt], -1, (0, 255, 0), 5)  # 사각 컨투어를 녹색으로 그림
            logging.info(f"Rectangle Contour Found")
            newImage = self.four_point_transform(img, screenCnt.reshape(4, 2))

        # _, edges_bytes = cv2.imencode("." + extension, edges)
        # _, contour_bytes = cv2.imencode("." + extension, cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR))
        # _, result_bytes = cv2.imencode("." + extension, cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))

        # return io.BytesIO(edges_bytes), io.BytesIO(contour_bytes), io.BytesIO(result_bytes)
        return newImage

    def four_point_transform(self, image, pts):
        def order_points(pts):
            r = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            r[0] = pts[np.argmin(s)]
            r[2] = pts[np.argmax(s)]
            r[1] = pts[np.argmin(diff)]
            r[3] = pts[np.argmax(diff)]
            return r

        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        docArea = maxWidth * maxHeight
        orgArea = image.shape[0] * image.shape[1]
        logging.info(f"\ndoc area: ({maxWidth},{maxHeight})={docArea},\norg area: {image.shape}={orgArea}")
        if docArea > orgArea*0.5:
            logging.info("transformed")
            dst = np.array([
                [0, 0],
                [maxWidth, 0],
                [maxWidth, maxHeight],
                [0, maxHeight]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        else:
            logging.info("originals")
            warped = image

        return warped

    def background_filtering(self, img_rgb, extension):
        image_filtered = img_rgb.copy()
        hsv = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        hist_s, _ = np.histogram(s, bins=256, range=(0, 256))
        dominant_s = np.argmax(hist_s)
        hist_v, _ = np.histogram(v, bins=256, range=(0, 256))
        dominant_v = np.argmax(hist_v)

        gray_mask = (s < dominant_s + 15) & (v > dominant_v - 60)
        v[gray_mask] = 255
        s[gray_mask] = 0
        image_hsv = cv2.merge([h, s, v])
        image_filtered = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        return image_filtered

    def filtering(self, img_rgb):
        """ 이미지의 명암대비 강화 및 밝기 증가"""

        alpha = 1.1  # 대비 계수
        beta = 1.0  # 밝기 조절이 필요 없는 경우 1.0
        enhanced_contrast_image = cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)  # 대비가 강화된 이미지 생성

        return enhanced_contrast_image


