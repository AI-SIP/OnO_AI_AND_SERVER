import cv2
import numpy as np
import io


def rgb_to_hsv(r, g, b):
    color = np.uint8([[[r, g, b]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color[0][0]


def to_hsv(image_rgb):
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
    return image_hsv


class ColorRemover:
    def __init__(self, target_rgb=(34, 30, 235), tolerance=60):
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
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        hist_s, _ = np.histogram(s, bins=256, range=(0, 256))
        dominant_s = np.argmax(hist_s)
        hist_v, _ = np.histogram(v, bins=50, range=(0, 256))
        dominant_v = np.argmax(hist_v) * 256 / 50
        gray_mask = (s < dominant_s + 15) & (v > dominant_v - 70)
        v[gray_mask] = 255
        s[gray_mask] = 0
        image_hsv = cv2.merge([h, s, v])
        image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

        image_rgb = cv2.GaussianBlur(image_rgb, (21, 21), 0)

        '''sharpening_mask = np.array([[0, -1, 0], [-1, 9, -1], [0, -1, 0]])
        image_rgb = cv2.filter2D(image_rgb, -1, sharpening_mask)'''

        image_processed = self.combine_alpha(image_rgb)
        _, filtered_bytes = cv2.imencode("."+extension, image_processed)
        return filtered_bytes

    def mask(self, image_rgb, mask_path=None):
        image_hsv = to_hsv(image_rgb)
        lower_bound = np.array([self.target_hsv[0] - self.tolerance, 50, 50])
        upper_bound = np.array([self.target_hsv[0] + self.tolerance, 255, 255])

        self.masks = cv2.inRange(image_hsv, lower_bound, upper_bound)

        image_rgb[self.masks != 0] = [255, 255, 255]
        return image_rgb

    def inpaint(self, image_rgb):
        if self.masks is not None and isinstance(self.masks, np.ndarray):
            inpainted_image = cv2.inpaint(image_rgb, self.masks, 35, cv2.INPAINT_TELEA)
            return inpainted_image
        else:
            raise ValueError("Mask is not properly defined or is not a numpy array.")

    def remove_alpha(self, image):
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
        input_bytes = self.background_filtering(image_rgb, extension)  # input filtering

        image_masked = self.mask(image_rgb)  # masking
        _, mask_bytes = cv2.imencode("." + extension, self.masks.astype(np.uint8))
        image_inpainted = self.inpaint(image_masked)  # inpainting
        result_bytes = self.background_filtering(image_inpainted, extension)  # output filtering

        return io.BytesIO(input_bytes), io.BytesIO(mask_bytes), io.BytesIO(result_bytes)


