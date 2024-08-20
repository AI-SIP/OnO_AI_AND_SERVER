import logging
import cv2
import numpy as np
import io


def rgb_to_hsv(r, g, b):
    color = np.uint8([[[r, g, b]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color


class ColorRemover:
    def __init__(self, target_rgb=(58, 58, 152), tolerance=20):
        self.size = 0
        self.height = 0
        self.width = 0
        self.target_rgb = target_rgb
        self.tolerance = tolerance
        self.target_hsv = rgb_to_hsv(*target_rgb)
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
        lower_bound = np.array([105, 10, 30])  # blue's hue is 105~135
        upper_bound = np.array([135, 255, 255])
        self.masks = cv2.inRange(image_hsv, lower_bound, upper_bound)

        if self.size > 1024 * 1536:
            kernel = np.ones((5, 5), np.uint8) # mask 내 노이즈 제거
            self.masks = cv2.morphologyEx(self.masks, cv2.MORPH_OPEN, kernel)

    def inpainting(self, image_rgb):
        if self.masks is not None and isinstance(self.masks, np.ndarray):
            inpainted_image = image_rgb.copy()
            inpainted_image = cv2.inpaint(inpainted_image, self.masks, 5, cv2.INPAINT_TELEA)
            # blurred_region = cv2.GaussianBlur(inpainted_image, (10, 10), 1.5)
            # inpainted_image[self.masks != 0] = blurred_region[self.masks != 0]
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
        image_rgb = self.scaling(image_rgb)  # perspective transforming

        self.masking(image_rgb)  # masking
        image_inpainted = self.inpainting(image_rgb)  # inpainting

        # image_input = self.background_filtering(image_rgb, extension)  # input filtering
        # image_output = self.background_filtering(image_inpainted, extension)  # output filtering
        image_output = self.filtering(image_inpainted)  # output filtering

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

        alpha = 1.2  # 대비 계수
        beta = 1.1  # 밝기 조절이 필요 없는 경우 0
        enhanced_contrast_image = cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)  # 대비가 강화된 이미지 생성

        return enhanced_contrast_image


