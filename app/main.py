import io
from uuid import uuid4
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from starlette.responses import StreamingResponse, JSONResponse
from app.ColorRemover import ColorRemover
import app.ImageFunctions as ImageManager
from boto3 import client

app = FastAPI()
paths = dict()

'''
보안 주의
'''
BUCKET_NAME = "myawsbucket-mvp"
s3_client = client(
        "s3",
        aws_access_key_id="",
        aws_secret_access_key="",
        region_name="ap-northeast-2",
)


def create_file_path(obj_path, extension):
    global paths
    file_id = uuid4()  # 각 클라이언트마다 고유한 파일 ID 생성
    dir_path = obj_path.rsplit('/', 1)[0]
    paths = {"input_path": f"{dir_path}/{file_id}.input.{extension}",
             "output_path": f"{dir_path}/{file_id}.output.{extension}",
             "mask_path": f"{dir_path}/{file_id}.mask.{extension}",
             "extension": extension}
    print(paths)


def get_file_path():
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def upload_image_to_s3(file_bytes, file_path):
    """ 이미지를 S3에 동기적으로 업로드 """
    s3_client.upload_fileobj(file_bytes, BUCKET_NAME, file_path)


@app.get("/process-color/{full_url:path}")
def process_color(full_url: str):
    """ color-based handwriting detection & Telea Algorithm-based inpainting """
    print(full_url)
    s3_key = parse_s3_url(full_url)
    create_file_path(s3_key, s3_key.split(".")[-1])
    img_bytes = download_image_from_s3(s3_key)  # download from S3

    target_rgb = (34, 30, 235)  # 221EEB in RGB
    color_remover = ColorRemover(target_rgb, tolerance=20)
    img_mask_bytes, img_output_bytes = color_remover.process(img_bytes, paths['extension'])

    upload_image_to_s3(io.BytesIO(img_bytes), paths["input_path"])
    upload_image_to_s3(img_mask_bytes, paths["mask_path"])
    upload_image_to_s3(img_output_bytes, paths["output_path"])

    return JSONResponse(content={"message": "File processed successfully", "output": paths['output_path']})


@app.get("/show-output")
def showByOutput():
    try:
        img_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=paths["output_path"])
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            raise HTTPException(status_code=404, detail="File not found")
        else:
            raise HTTPException(status_code=500, detail=e)

    return StreamingResponse(content=img_obj['Body'], media_type="image/" + paths["extension"])


@app.get("/show-url")
def showByUrl(full_url: str):
    s3_key = parse_s3_url(full_url)
    try:
        img_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    except ClientError as e:
        error_code = e.response['Error']['Code']
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
        create_file_path(upload_file.filename, extension)  # 경로 설정
        s3_client.upload_fileobj(upload_file.file, BUCKET_NAME, paths["input_path"])
    except Exception:
        raise HTTPException(status_code=500, detail="File upload failed")
    finally:
        upload_file.file.close()

    return {"message": f"File {upload_file.filename} uploaded successfully",
            "path": paths["input_path"]}

