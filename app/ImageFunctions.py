from PIL.Image import Image
from fastapi import HTTPException, UploadFile, status


class Images:
    async def validate_type(file: UploadFile) -> UploadFile:
        if file.content_type not in ["image/jpg", "image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="지원하지 않는 이미지 형식",
            )
        if not file.content_type.startswith("image"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미지가 아닌 파일",
            )

        return file


    async def validate_size(file: UploadFile) -> UploadFile:
        if len(await file.read()) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="10MB 초과 - 너무 큰 이미지 용량",
            )

        return file

    def scanning(self):
        import cv2
        import numpy as np

        def auto_scan_image(image_path):
            # 이미지를 읽어온 후 그레이스케일로 변환
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)

            # 윤곽을 찾고, 가장 큰 윤곽을 문서의 경계로 간주
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # 근사 윤곽이 4개의 점을 갖고 있을 때 (즉, 사각형일 때)
                if len(approx) == 4:
                    screenCnt = approx
                    break

            # 왜곡 보정
            pts = screenCnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            # 좌상, 우상, 우하, 좌하 순서로 점을 정렬
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            # 좌상, 우상, 우하, 좌하 점들을 사용하여 투시 변환을 수행
            (tl, tr, br, bl) = rect
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
                [0, maxHeight - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

            # 배경을 밝게 만들기 위한 색상 보정
            warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            warp = cv2.threshold(warp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            return warp

        # 이미지 경로
        image_path = 'path_to_your_document_image.jpg'
        scanned_image = auto_scan_image(image_path)

        # 결과 이미지를 보여주거나 저장
        cv2.imshow('Scanned Image', scanned_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def analyze_colors(image_path): # 색상 인식
        import cv2
        import numpy as np

        image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 색상과 채도의 평균과 표준편차 계산
        hue_mean, hue_std = np.mean(hsv_image[:, :, 0]), np.std(hsv_image[:, :, 0])
        sat_mean, sat_std = np.mean(hsv_image[:, :, 1]), np.std(hsv_image[:, :, 1])

        print(f"Hue: Mean = {hue_mean}, Std = {hue_std}")
        print(f"Saturation: Mean = {sat_mean}, Std = {sat_std}")

        # 색상과 채도에 따라 필기와 인쇄된 글 분리
        if sat_std > 15 and hue_std > 20:
            print("이미지는 학생의 샤프펜슬 필기일 가능성이 높습니다.")
        else:
            print("이미지는 문항일 가능성이 높습니다.")

    def save_img(image: Image, file_path: str):
        image.save(file_path)
        return file_path
