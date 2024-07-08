import cv2
import numpy as np

def rgb_to_hsv(r, g, b):
    color = np.uint8([[[r, g, b]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color[0][0]

def to_hsv(image_rgb):
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
    return image_hsv

class ColorRemover:
    def __init__(self, target_rgb, tolerance=40):
        self.target_rgb = target_rgb
        self.tolerance = tolerance
        self.target_hsv = rgb_to_hsv(*target_rgb)
        self.alpha_channel = None
        self.masks = None
        print(cv2.version)
        print(np.version)

    def remove_alpha(self, image):
        if image.shape[2] == 4:  # RGBA
            self.alpha_channel = image[:, :, 3]
            image_rgb = image[:, :, :3]
        else:
            image_rgb = image

        return image_rgb

    def combine_alpha(self, image_rgb):
        if self.alpha_channel is not None:  # RGBA
            image_rgba = cv2.merge([image_rgb, self.alpha_channel])
            return image_rgba
        else:
            return image_rgb

    def open(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_rgb = self.remove_alpha(image)

        return image_rgb

    def mask(self, image_rgb, mask_path):
        image_hsv = to_hsv(image_rgb)
        lower_bound = np.array([self.target_hsv[0] - self.tolerance, 50, 50])
        upper_bound = np.array([self.target_hsv[0] + self.tolerance, 255, 255])

        self.masks = cv2.inRange(image_hsv, lower_bound, upper_bound)
        cv2.imwrite(mask_path, self.masks)

    def fill(self, image_rgb):
        image_rgb[self.masks != 0] = [255, 255, 255]

        return image_rgb

    def blur(self, image_rgb, kernel_size=(601, 601)):
        mask_3d = self.masks[:, :, None] == 0
        blurred_image = cv2.GaussianBlur(image_rgb, kernel_size, 0)
        image_rgb = np.where(mask_3d, image_rgb, blurred_image)

        return image_rgb

    def inpaint(self, image_rgb):
        if self.masks is not None and isinstance(self.masks, np.ndarray):
            inpainted_image = cv2.inpaint(image_rgb, self.masks, 15, cv2.INPAINT_TELEA)
            return inpainted_image
        else:
            raise ValueError("Mask is not properly defined or is not a numpy array.")

    def save(self, image_rgb, output_path):
        image = self.combine_alpha(image_rgb)
        cv2.imwrite(output_path, image)

        return image

    def process(self, image_path, output_path, mask_path):
        image_rgb = self.open(image_path)
        self.mask(image_rgb, mask_path)
        image_filled = self.fill(image_rgb)
        # image_blured = self.blur(image_filled)
        image_inpainted = self.inpaint(image_filled)
        image_processed = self.save(image_inpainted, output_path)

        return image_processed

    def processWithCheck(self, image_path, output_path, mask_path):
        """
        Execution time: 1.48 seconds
        CPU Usage After: 19.8%
        Memory Usage: 2.0%
        """
        import time
        import psutil

        cpu_usage_before = psutil.cpu_percent(interval=None)
        memory_usage_before = psutil.virtual_memory().percent
        start_time = time.time()

        image_processed = self.process(self, image_path, output_path, mask_path) # 실행

        end_time = time.time()
        cpu_usage_after = psutil.cpu_percent(interval=1)
        memory_usage_after = psutil.virtual_memory().percent

        print(f"Execution time: {end_time - start_time:.2f} seconds")
        print(f"CPU Usage After: {cpu_usage_after - cpu_usage_before}%")
        print(f"Memory Usage: {memory_usage_after - memory_usage_before}%")

        return image_processed
