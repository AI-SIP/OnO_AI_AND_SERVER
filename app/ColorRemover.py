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

    def masking(self, image_rgb):  # important
        image_mask = image_rgb.copy()
        image_hsv = cv2.cvtColor(image_mask, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([105, 30, 30])  # blue's hue is 105~135
        upper_bound = np.array([140, 255, 255])
        self.masks = cv2.inRange(image_hsv, lower_bound, upper_bound)

        if self.size > 2048 * 1536:
            kernel = np.ones((3, 3), np.uint8) # mask 내 노이즈 제거
            self.masks = cv2.morphologyEx(self.masks, cv2.MORPH_OPEN, kernel)

    def inpainting(self, image_rgb):
        if self.masks is not None and isinstance(self.masks, np.ndarray):
            inpainted_image = image_rgb.copy()
            inpainted_image = cv2.inpaint(inpainted_image, self.masks, 3, cv2.INPAINT_TELEA)
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

        self.masking(image_rgb)  # masking
        image_inpainted = self.inpainting(image_rgb)  # inpainting

        # image_input = self.background_filtering(image_rgb, extension)  # input filtering
        image_output = self.background_filtering(image_inpainted, extension)  # output filtering

        image_input = self.combine_alpha(image_rgb)
        image_output = self.combine_alpha(image_output)

        _, input_bytes = cv2.imencode("." + extension, image_input)
        _, mask_bytes = cv2.imencode("." + extension, self.masks.astype(np.uint8))
        _, result_bytes = cv2.imencode("." + extension, image_output)

        return io.BytesIO(input_bytes), io.BytesIO(mask_bytes), io.BytesIO(result_bytes)




