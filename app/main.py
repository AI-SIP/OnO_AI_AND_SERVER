from uuid import uuid4
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from pathlib import Path
from starlette.responses import FileResponse
import shutil
from app.ColorRemover import ColorRemover
from app.ImageFunctions import Images


def get_file_path():
    return paths


def create_file_path(file_name):
    global paths

    file_id = uuid4() # 각 클라이언트마다 고유한 파일 경로 생성
    extension=file_name.split('.')[-1]
    paths = {"input_path": f"./imgs/{file_id}.input.{extension}",
             "output_path": f"./imgs/{file_id}.output.{extension}",
             "mask_path": f"./imgs/{file_id}_mask.{extension}",}

    return paths


app = FastAPI()
paths = dict()

@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.post("/upload")
async def upload(img: UploadFile = File(...)):
    if img is None:
        raise HTTPException(status_code=400, detail="No input file")

    validated_file = await Images.validate_type(img)
    validated_file = await Images.validate_size(img)
    validated_file.file.seek(0)

    file_paths = create_file_path(img.filename)
    input_img_path = file_paths['input_path']

    with open(input_img_path, "wb") as buffer:
        shutil.copyfileobj(validated_file.file, buffer)

    return {"message": "File uploaded successfully", "path": input_img_path}


@app.get("/process-color")
def process_color(file_paths: dict = Depends(get_file_path)):
    input_img_path = file_paths['input_path']
    output_img_path = file_paths['output_path']
    mask_path = file_paths['mask_path']

    if not Path(input_img_path).exists():
        raise HTTPException(status_code=404, detail=f"Input image not found")

    target_rgb = (34, 30, 235)  # 221EEB in RGB
    color_remover = ColorRemover(target_rgb, tolerance=20)
    color_remover.process(input_img_path, output_img_path, mask_path)

    return FileResponse(output_img_path, media_type="image/" + output_img_path.split('.')[-1])
