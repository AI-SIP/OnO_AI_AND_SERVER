from uuid import uuid4
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from starlette.responses import StreamingResponse
from app.ColorRemover import ColorRemover
import app.ImageFunctions as ImageManager
from boto3 import client

app = FastAPI()
paths = dict()

'''
보안 주의
'''
BUCKET_NAME = "your bucket name"
s3_client = client(
        "s3",
        aws_access_key_id="access key id",
        aws_secret_access_key="secret access key",
        region_name="region",
    )


def get_file_path():
    return paths


def create_file_path(file_name, extension):
    global paths
    file_id = uuid4()  # 각 클라이언트마다 고유한 파일 ID 생성
    extension = file_name.split('.')[-1]
    paths = {"input_path": f"./app/imgs/{file_id}.input.{extension}",
             "output_path": f"./app/imgs/{file_id}.output.{extension}",
             "mask_path": f"./app/imgs/{file_id}.mask.{extension}",
             "extension": extension}
    return paths


@app.post("/upload")
async def upload(upload_file: UploadFile = File(...)):
    try:
        if upload_file is None:
            raise HTTPException(status_code=400, detail="No input file")
        upload_file, extension = await ImageManager.validate_type(upload_file)
        upload_file = await ImageManager.validate_size(upload_file)
        file_paths = create_file_path(upload_file.filename, extension)  # 경로 설정
        s3_client.upload_fileobj(upload_file.file, BUCKET_NAME, file_paths["input_path"])
    except Exception:
        raise HTTPException(status_code=500, detail="File upload failed")
    finally:
        upload_file.file.close()

    return {"message": f"File {upload_file.filename} uploaded successfully",
            "path": file_paths["input_path"]}

@app.get("/process-color")
def process_color(file_paths: dict = Depends(get_file_path)):
    """
        color-based handwriting detection & Telea Algorithm-based inpainting
    """
    img_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_paths['input_path'])  # download from S3
    img_bytes = img_obj['Body'].read()

    target_rgb = (34, 30, 235)  # 221EEB in RGB
    color_remover = ColorRemover(target_rgb, tolerance=20)
    img_mask_bytes, img_output_bytes = color_remover.process(img_bytes, file_paths['extension'])

    s3_client.upload_fileobj(img_mask_bytes, BUCKET_NAME, file_paths["mask_path"])
    s3_client.upload_fileobj(img_output_bytes, BUCKET_NAME, file_paths["output_path"])

    return {"message": "File processed successfully", "output": file_paths['output_path']}

@app.get("/show/{file_type}")
def show(file_type: str, file_paths: dict = Depends(get_file_path)):
    """
       Show images from S3 based on the file type: input, output, or mask.
    """
    if file_type not in ['input', 'output', 'mask']:
        raise HTTPException(status_code=400, detail="Invalid file type specified.")
    key_path = file_paths.get(f"{file_type}_path")

    try:
        img_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key_path)
    except Exception:
        raise HTTPException(status_code=500, detail=f"File {file_type} does not exist in here: {key_path}")

    return StreamingResponse(content=img_obj['Body'], media_type="image/"+file_paths['extension'])
