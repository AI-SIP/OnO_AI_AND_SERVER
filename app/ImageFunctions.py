import io
from PIL import Image
from PIL import ExifTags
from fastapi import HTTPException, UploadFile, status
import logging


async def validate_type(file: UploadFile):
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
    return file, file.content_type.split("/")[-1]


async def validate_size(file: UploadFile) -> UploadFile:
    if len(await file.read()) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="10MB 초과 - 너무 큰 이미지 용량",
        )
    file.file.seek(0)
    return file


def correct_rotation(img_data: bytes, extension="jpeg") -> bytes:
    """EXIF 정보에 따라 이미지를 올바른 방향으로 회전시키는 함수"""
    logging.info(f"format is {extension}")
    if extension == "png":
        return img_data

    else:  # "jpg" or "jpeg":
        extension = 'jpeg'
        image = Image.open(io.BytesIO(img_data))
        logging.info("opened format")
        try:
            exif = image.getexif()
            if "exif" is not None:
                logging.info("exif exists: {!r}".format(exif))
                orientation_key = next((key for key, value in ExifTags.TAGS.items() if value == 'Orientation'), None)
                if orientation_key and orientation_key in exif:
                    orientation = exif[orientation_key]
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
            else:
                logging.info("No EXIF information available.")

        except AttributeError:  # 이미지에 EXIF 정보가 없는 경우
            logging.info("No EXIF information available.")

        # 수정된 이미지를 바이트로 변환하여 반환
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=extension)
        img_byte_arr.seek(0)

        return img_byte_arr.read()








