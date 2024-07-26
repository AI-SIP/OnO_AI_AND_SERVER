import io
from uuid import uuid4
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from starlette import status
from starlette.responses import StreamingResponse, JSONResponse
from ColorRemover import ColorRemover
import ImageFunctions as ImageManager
import logging

# 로깅 추가
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
s3_client = boto3.client('s3')
try:
    response = s3_client.list_buckets()
    BUCKET_NAME = response['Buckets'][0]['Name']  # 첫 번째 버킷의 이름
except Exception as e:
    raise HTTPException(status_code=404, detail='No Buckets found in S3')


def create_file_path(obj_path, extension):
    file_id = uuid4()  # 각 클라이언트마다 고유한 파일 ID 생성
    dir_path = obj_path.rsplit('/', 1)[0]
    paths = {"input_path": f"{dir_path}/{file_id}.input.{extension}",
             "output_path": f"{dir_path}/{file_id}.output.{extension}",
             "mask_path": f"{dir_path}/{file_id}.mask.{extension}",
             "extension": extension}
    return paths


def parse_s3_url(full_url: str):
    """ URL에서 S3 키를 추출 """
    parsed_url = urlparse(full_url)
    return parsed_url.path.lstrip('/')


def download_image_from_s3(s3_key: str):
    """ S3에서 이미지를 동기적으로 다운로드 """
    try:
        img_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        return img_obj['Body'].read()
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Image not found in S3 : {s3_key}.")
    except Exception as de:
        raise HTTPException(status_code=500, detail=str(de))


def upload_image_to_s3(file_bytes, file_path):
    """ 이미지를 S3에 동기적으로 업로드 """
    s3_client.upload_fileobj(file_bytes, BUCKET_NAME, file_path)


@app.get("/", status_code=status.HTTP_200_OK)
def greeting():
    return JSONResponse(content={"message": "Hello!"})


@app.get("/process-color")
def processColor(full_url: str):
    """ color-based handwriting detection & Telea Algorithm-based inpainting """
    logger.info("Processing color for URL: %s", full_url)
    try:
        s3_key = parse_s3_url(full_url)
        paths = create_file_path(s3_key, s3_key.split(".")[-1])
        img_bytes = download_image_from_s3(s3_key)  # download from S3
        logger.info("Key is : %s and Start processing", s3_key)

        target_rgb = (34, 30, 235)  # 221EEB in RGB
        color_remover = ColorRemover(target_rgb, tolerance=20)
        img_input_bytes, img_mask_bytes, img_output_bytes = color_remover.process(img_bytes, paths['extension'])
        logger.info("Finished Processing, and Start Uploading Image")

        upload_image_to_s3(img_input_bytes, paths["input_path"])
        upload_image_to_s3(img_mask_bytes, paths["mask_path"])
        upload_image_to_s3(img_output_bytes, paths["output_path"])

        logger.info("All finished Successfully")
        return JSONResponse(content={"message": "File processed successfully", "path": paths})

    except Exception as pe:
        logger.error("Error during processing: %s", pe)
        raise HTTPException(status_code=500, detail="Error processing the image.")


@app.get("/show-url")
def showByUrl(full_url: str):
    s3_key = parse_s3_url(full_url)
    try:
        img_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    except ClientError as ce:
        error_code = ce.response['Error']['Code']
        if error_code == 'NoSuchKey':
            raise HTTPException(status_code=404, detail="File not found")
        else:
            raise HTTPException(status_code=500, detail=f"500 error")

    return StreamingResponse(content=img_obj['Body'], media_type="image/" + s3_key.split('.')[-1])


@app.post("/direct/upload")
async def upload_directly(upload_file: UploadFile = File(...)):
    try:
        if upload_file is None:
            raise HTTPException(status_code=400, detail="No input file")
        upload_file, extension = await ImageManager.validate_type(upload_file)
        upload_file = await ImageManager.validate_size(upload_file)
        paths = create_file_path('images/', extension)  # 경로 설정
        s3_client.upload_fileobj(upload_file.file, BUCKET_NAME, paths["input_path"])
    except Exception:
        raise HTTPException(status_code=500, detail="File upload failed")
    finally:
        upload_file.file.close()

    return {"message": f"File {upload_file.filename} uploaded successfully",
            "path": paths["input_path"]}
