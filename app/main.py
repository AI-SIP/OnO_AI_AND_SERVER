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

from openai import OpenAI
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 로깅 추가
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

#  클라이언트 생성
s3_client = boto3.client( "s3",
                          region_name="ap-northeast-2")
ssm_client = boto3.client('ssm',
                          region_name='ap-northeast-2')
# s3 버킷 연결
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

@app.get("/analysis/ocr")
async def ocr(problem_url: str):
    """ OCR with Naver Clova OCR API"""
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

        ssm = boto3.client('ssm',
                           region_name='ap-northeast-2')
        api_url = ssm.get_parameter(
            Name='/ono/dev/fastapi/CLOVA_API_URL',
            WithDecryption=False
        )['Parameter']['Value']
        secret_key = ssm.get_parameter(
            Name='/ono/dev/fastapi/CLOVA_SECRET_KEY',
            WithDecryption=False
        )['Parameter']['Value']

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

# OpenAI 연결
openai_secret_key = ssm_client.get_parameter(
    Name='/ono/new_dev/fastapi/OPENAI_API_KEY',
    WithDecryption=False
)['Parameter']['Value']
openai_client = OpenAI(api_key=openai_secret_key)

# Mivlus DB 연결
MILVUS_HOST = 'localhost'
MILVUS_PORT = 19530
DB_NAME = "ono_dev"
COLLECTION_NAME = 'Math2015Curriculum'
DIMENSION = 1536
INDEX_TYPE = "IVF_FLAT"

@app.get("/milvus/connect")
async def connect_milvus():
    # Milvus 서버 연결
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, db_name=DB_NAME)
    logger.info(f"* log >> Milvus Server is connected to {MILVUS_HOST}:{MILVUS_PORT}")

    # 컬렉션의 스키마 출력
    collection = Collection(COLLECTION_NAME)
    logger.info("* Collection Schema:")
    for field in collection.schema.fields:
        logger.info(f"    - Field Name: {field.name}, Data Type #: {field.dtype}")


@app.get("/milvus/create")
async def create_milvus():
    await connect_milvus()  # milvus 서버 연결

    # 스키마 및 컬렉션 생성
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name='content_embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]
    schema = CollectionSchema(fields=fields, description='Math2015Curriculum embedding collection')
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    logger.info(f"* log >> Collection [{COLLECTION_NAME}] is created.")

    # 인덱스 생성
    # 스칼라 인덱스
    collection.create_index(
        field_name="id"
    )
    # 벡터 인덱스
    index_params = {
        'index_type': INDEX_TYPE,
        'metric_type': 'COSINE',
        'params': {
            'nlist': 128
        }
    }
    collection.create_index(
        field_name="content_embedding",
        index_params=index_params
    )
    logger.info(f"* log >> 인덱스 생성 결과: {[idx.index_name for idx in collection.indexes]}")  # True

    # 컬렉션의 스키마 출력
    collection = Collection(COLLECTION_NAME)
    logger.info("* Collection Schema:")
    for field in collection.schema.fields:
        logger.info(f"    - Field Name: {field.name}, Data Type #: {field.dtype}")


def get_embedding(client, text_list):
    try:
        embedding_response = client.embeddings.create(
            input=text_list,  # 배열 input
            model="text-embedding-3-small"
        )
        vectors = [d.embedding for d in embedding_response.data]
        return vectors
    except Exception as e:
        logger.info(f"임베딩 생성 중 오류 발생: {e}")
        return [None] * len(text_list)


@app.get("/milvus/insert")
async def insert_curriculum_embeddings():
    """ s3에서 교과과정을 읽고 임베딩하여 Milvus에 삽입 """
    # Milvus 연결
    await connect_milvus()
    collection = Collection(COLLECTION_NAME)

    # S3 내 커리큘럼 데이터 로드
    texts = []
    prefix = 'curriculum/math2015/'  # 경로
    try:
        # 버킷에서 파일 목록 가져오기
        s3_curriculum_response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        for item in s3_curriculum_response.get('Contents', []):
            s3_key = item['Key']
            if s3_key.endswith('.txt'):
                # S3 객체 가져오기
                obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
                # 텍스트 읽기
                text = obj['Body'].read().decode('utf-8')
                texts.append(text)
        logging.info(f"* log >> read {len(texts)} texts from S3")
    except Exception as e:
        logging.error(f"Error reading curriculum from S3: {e}")

    # 데이터 임베딩
    content_embeddings = get_embedding(openai_client, texts)
    logging.info(f"* log >> embedding 완료. dimension: {DIMENSION}")

    # 데이터 삽입
    data = [
        texts,  # content 필드
        content_embeddings  # content_embedding 필드
    ]
    status = collection.insert(data)
    print(f"* log >> 데이터 삽입 완료")
    return {"status": status, "ids": status.primary_keys}


@app.get("/analysis/retrieve")
async def retrieve(problem_text: str):
    try:
        # Milvus 연결
        await connect_milvus()

        # 컬렉션의 스키마 출력
        collection = Collection(COLLECTION_NAME)
        logger.info("* Collection Schema:")
        for field in collection.schema.fields:
            logger.info(f"    - Field Name: {field.name}, Data Type #: {field.dtype}")

        # 검색 테스트
        query = problem_text
        query_embeddings = [get_embedding(openai_client, [query])]
        if not query_embeddings or query_embeddings[0] is None:
            raise ValueError("Embedding generation failed")
        logger.info(f"* log >> Query embedding 완료")

        search_params = {
            'metric_type': 'COSINE',
            'params': {
                'probe': 20
            },
        }
        results = collection.search(
            data=query_embeddings[0],
            anns_field='content_embedding',
            param=search_params,
            limit=3,
            expr=None,
            output_fields=['content']
        )
        context = ' '.join([result.entity.get('content') for result in results[0]])
        logger.info(f"* log >> context found")

        # 결과 확인
        '''logger.info(f"* log >> 쿼리 결과")
        for result in results[0]:
            logger.info("\n-------------------------------------------------------------------")
            logger.info(f"Score : {result.distance}, \nText : \n{result.entity.get('content')}")'''

        return context
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/augmentation")
async def augment(curriculum_context, query):
    prompt = "교과과정에 기반하여 이 문제에 필요한 개념을 말해줘. 응답은 자연어처럼 제공해줘. \n"
    context = curriculum_context
    passage = query
    augmented_query = prompt + context+ passage
    return augmented_query

@app.get("/analysis/generation")
async def generate(question):
    def get_chatgpt_response(client, question, model="gpt-4o-mini"):
        try:
            gpt_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user",
                     "content": question
                     }
                ],
                temperature=0.5
            )
            return gpt_response.choices[0].message.content
        except Exception as e:
            print(f"Error during GPT querying: {e}")
            return None

    chatgpt_response = get_chatgpt_response(openai_client, question)
    logging.info(f"* log >> ChatGPT Response: {chatgpt_response}")
    return chatgpt_response

