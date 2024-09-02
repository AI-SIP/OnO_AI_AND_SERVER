import os
import io
from uuid import uuid4
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
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
    try:
        img_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        return img_obj['Body'].read()
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Image not found in S3 : {s3_key}.")
    except Exception as de:
        raise HTTPException(status_code=500, detail=str(de))


def upload_image_to_s3(file_bytes, file_path):
    s3_client.upload_fileobj(file_bytes, BUCKET_NAME, file_path)


@app.get("/", status_code=status.HTTP_200_OK)
def greeting():
    return JSONResponse(content={"message": "Hello! Let's start image processing"})


@app.post("/process-color")
async def processColor(request: Request):
    """ color-based handwriting detection & Telea Algorithm-based inpainting """
    data = await request.json()
    full_url = data['fullUrl']
    colors_list = data['colorsList']
    tolerance = data.get('tolerance')  # value or None

    try:
        target_rgb_list = []
        for color in colors_list:
            if color is None:
                continue
            target_rgb = (color['red'], color['green'], color['blue'])
            target_rgb_list.append(target_rgb)
        logger.info("Target rgb list is %s", target_rgb_list)

        s3_key = parse_s3_url(full_url)
        paths = create_file_path(s3_key, s3_key.split(".")[-1])
        img_bytes = download_image_from_s3(s3_key)  # download from S3
        corrected_img_bytes = ImageManager.correct_rotation(img_bytes, paths['extension'])
        logger.info("Key is : %s and Start processing", s3_key)

        if tolerance is not None:
            color_remover = ColorRemover(target_rgb_list, tolerance)
        else:
            color_remover = ColorRemover(target_rgb_list)
        img_input_bytes, img_mask_bytes, img_output_bytes = color_remover.process(corrected_img_bytes, paths['extension'])
        logger.info("Finished Processing, and Start Uploading Image")

        upload_image_to_s3(img_input_bytes, paths["input_path"])
        upload_image_to_s3(img_mask_bytes, paths["mask_path"])
        upload_image_to_s3(img_output_bytes, paths["output_path"])

        logger.info("All finished Successfully")
        return JSONResponse(content={"message": "File processed successfully", "path": paths})

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e.args[0]}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Error during processing: %s", e)
        raise HTTPException(status_code=500, detail="Error processing the image.")


@app.get("/analysis")
async def analyzeProblem(problem_url: str):
    """ Curriculum-based Problem Analysis API with CLOVA OCR & ChatGPT  """
    import requests
    import uuid
    import time
    import json

    logger.info("Analyzing problem from this image URL: %s", problem_url)
    try:
        s3_key = parse_s3_url(problem_url)
        img_bytes = download_image_from_s3(s3_key)  # download from S3
        extension = s3_key.split(".")[-1]
        logger.info("Completed Download & Sending Requests... '%s'", s3_key)

        api_url = os.getenv("CLOVA_API_URL")
        secret_key = os.getenv("CLOVA_SECRET_KEY")
        image_file = ImageManager.correct_rotation(img_bytes, extension)  # rotating correction

        headers = {
            'X-OCR-SECRET': secret_key
        }
        request_json = {
            'images': [
                {
                    'format': extension,
                    'name': 'ocr_sample'
                }
            ],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000)),
            'enableTableDetection': False
        }
        payload = {'message': json.dumps(request_json).encode('utf-8')}
        files = [
            ('file', image_file)
        ]
        logger.info("Processing OCR & Receiving Responses...")

        ocr_response = requests.request("POST", api_url, headers=headers, data=payload, files=files).text
        ocr_response_json = json.loads(ocr_response)
        logger.info("***** Finished Analyzing Successfully *****")

        infer_texts = []
        for image in ocr_response_json["images"]:
            for field in image["fields"]:
                infer_texts.append(field["inferText"])
        result = ' '.join(infer_texts)
        print(result)

        return JSONResponse(content={"message": "OCR Finished Successfully", "result": result})

    except Exception as pe:
        logger.error("Error during OCR: %s", pe)
        raise HTTPException(status_code=500, detail="Error during OCR.")


@app.get("/show-url")
def showByUrl(full_url: str):
    s3_key = parse_s3_url(full_url)
    try:
        img_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        img_bytes = img_obj['Body'].read()
        corrected_img_bytes = ImageManager.correct_rotation(img_bytes, s3_key.split(".")[-1])
    except ClientError as ce:
        error_code = ce.response['Error']['Code']
        if error_code == 'NoSuchKey':
            raise HTTPException(status_code=404, detail="File not found")
        else:
            raise HTTPException(status_code=500, detail=f"500 error")

    return StreamingResponse(content=io.BytesIO(corrected_img_bytes), media_type="image/" + s3_key.split('.')[-1])


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


@app.get("/scaling")
def scaling(full_url: str):
    logger.info("Processing scaling for URL: %s", full_url)
    try:
        s3_key = parse_s3_url(full_url)
        paths = create_file_path(s3_key, s3_key.split(".")[-1])
        img_bytes = download_image_from_s3(s3_key)  # download from S3
        logger.info("Key is : %s and Start processing", s3_key)

        color_remover = ColorRemover()
        img_output_bytes = color_remover.scaling(img_bytes, 'jpg')
        logger.info("Finished Scaling, and Start Uploading Image")

        upload_image_to_s3(img_output_bytes, "images/scaled.jpg")

        logger.info("All finished Successfully")
        return JSONResponse(content={"message": "File processed successfully"})

    except Exception as pe:
        logger.error("Error during processing: %s", pe)
        raise HTTPException(status_code=500, detail="Error processing the image.")