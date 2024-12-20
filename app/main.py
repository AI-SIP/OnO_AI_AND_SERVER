import io
from uuid import uuid4
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from starlette import status
from starlette.responses import StreamingResponse, JSONResponse
from ColorRemover import ColorRemover
from AIProcessor import AIProcessor
import ImageFunctions as ImageManager
import logging
import os
from openai import OpenAI
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import time
from datetime import datetime

# 로깅 추가
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

#  클라이언트 생성
s3_client = boto3.client("s3",
                          region_name="ap-northeast-2",)
ssm_client = boto3.client('ssm',
                          region_name="ap-northeast-2",)

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
             "mask_path": f"{dir_path}/{file_id}.mask.{extension}",
             "output_path": f"{dir_path}/{file_id}.output.{extension}",
             "one": f"{dir_path}/{file_id}.mask_b.{extension}",
             "two": f"{dir_path}/{file_id}.mask_p.{extension}",
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


def download_model_from_s3(yolo_path: str = 'models/yolo11_best.pt', sam_path: str = "models/mobile_sam.pt"):  # models/sam_vit_h_4b8939.pth
    dest_dir = f'../'  # 모델을 저장할 컨테이너 내 경로
    try:
        yolo_full_path = dest_dir+yolo_path
        sam_full_path = dest_dir+sam_path
        if not os.path.exists(yolo_full_path):
            s3_client.download_file(BUCKET_NAME, yolo_path, yolo_full_path)
            logger.info(f'YOLOv11 & SAM models downloaded successfully to {dest_dir}')
        else:
            logger.info(f'YOLOv11 already exists at {yolo_full_path}')
        if not os.path.exists(sam_full_path):
            s3_client.download_file(BUCKET_NAME, sam_path, sam_full_path)
            logger.info(f'SAM models downloaded successfully to {dest_dir}')
        else:
            logger.info(f'SAM models already exists at {dest_dir}')

        logger.info(f"Files in 'models' Dir: {os.listdir(dest_dir)}")
    except Exception as e:
        print(f'Failed to download model: {e}')


@app.get("/", status_code=status.HTTP_200_OK)
def greeting():
    return JSONResponse(content={"message": "Hello! Welcome to OnO's FastAPI Server!"})


@app.get("/load-models", status_code=status.HTTP_200_OK)
async def get_models():
    try:
        download_model_from_s3()
    except Exception as e:
        logger.error("Error with Download & Saving AIs: %s", e)
        raise HTTPException(status_code=500, detail="Error with Download & Saving AIs")


get_models()


@app.post("/process-shape")
async def processShape(request: Request):
    """ AI handwriting detection & Telea Algorithm-based inpainting """
    data = await request.json()
    full_url = data['fullUrl']
    point_list = data.get('points')
    label_list = data.get('labels')  # value or None
    logger.info(f"사용자 입력 포인트: {point_list}")
    logger.info(f"사용자 입력 라벨: {label_list}")

    try:
        s3_key = parse_s3_url(full_url)
        paths = create_file_path(s3_key, s3_key.split(".")[-1])
        img_bytes = download_image_from_s3(s3_key)  # download from S3
        corrected_img_bytes = ImageManager.correct_rotation(img_bytes, paths['extension'])
        logger.info(f"시용자 입력 이미지({s3_key}) 다운로드 및 전처리 완료")

        # aiProcessor = AIProcessor(yolo_path='/Users/semin/models/yolo11_best.pt', sam_path='/Users/semin/models/mobile_sam.pt')  # local
        aiProcessor = AIProcessor(yolo_path="../models/yolo11_best.pt", sam_path="../models/mobile_sam.pt")  # server
        img_input_bytes, img_mask_bytes, img_output_bytes, one, two = aiProcessor.process(img_bytes=corrected_img_bytes,
                                                                                          user_inputs=point_list)
        logger.info("AI 필기 제거 프로세스 완료")

        upload_image_to_s3(img_input_bytes, paths["input_path"])
        upload_image_to_s3(img_mask_bytes, paths["mask_path"])
        upload_image_to_s3(img_output_bytes, paths["output_path"])
        if one is not None:
            upload_image_to_s3(one, paths["one"])
        if two is not None:
            upload_image_to_s3(two, paths["two"])

        logger.info("AI 필기 제거 결과 이미지 업로드 완료")
        return JSONResponse(content={"message": "File processed successfully", "path": paths})

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e.args[0]}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Error during processing: %s", e)
        raise HTTPException(status_code=500, detail="Error processing the image.")


@app.post("/process-color")
async def processColor(request: Request):
    """ color-based handwriting detection & Telea Algorithm-based inpainting """
    data = await request.json()
    full_url = data['fullUrl']
    colors_list = data['colorsList']
    intensity = data.get('intensity')  # value or None

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

        if intensity is None:
            intensity = 1  # default: weak
        color_remover = ColorRemover(target_rgb_list, intensity)
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


@app.get("/show/image")
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


@app.post("/upload/image")
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


@app.get("/analysis/whole")
async def analysis(problem_url=None):
    """ Curriculum-based Chat Completion API with CLOVA OCR & ChatGPT  """
    await connect_milvus()  # milvus 서버 연결

    if problem_url is None:
        problem_text = '확률변수 X는 평균이 m, 표준편차가 5인 정규분포를 따르고, 확률변수 X의 확률밀도함수 f(x)가 다음 조건을 만족시킨다. m이 자연수일 때 P(17<=X<=18)=a이다. 1000a의 값을 오른쪽 표준정규분포표를 이용하여 구하시오.'
    else:
        problem_text = await ocr(problem_url)

    retrieving_result, subjects, units, concepts = await retrieve(problem_text)
    question = await augment(retrieving_result, problem_text)
    answer = await generate(question)

    return JSONResponse(content={"message": "Problem Analysis Finished Successfully",
                                 "subject": list(set(subjects)),
                                 "unit": list(set(units)),
                                 "key_concept": list(set(concepts)),
                                 "answer": answer})


@app.get("/analysis/ocr")
async def ocr(problem_url: str):
    """ OCR with Naver Clova OCR API"""
    import requests
    import uuid
    import time
    import json

    try:
        dt3 = datetime.fromtimestamp(time.time())
        s3_key = parse_s3_url(problem_url)
        img_bytes = download_image_from_s3(s3_key)  # download from S3
        extension = s3_key.split(".")[-1]
        dt4 = datetime.fromtimestamp(time.time())
        logger.info(f"{dt3}~{dt4}: 이미지 다운로드 완료")

        clova_api_url = ssm_client.get_parameter(
            Name='/ono/fastapi/CLOVA_API_URL',
            WithDecryption=False
        )['Parameter']['Value']
        clova_secret_key = ssm_client.get_parameter(
            Name='/ono/fastapi/CLOVA_SECRET_KEY',
            WithDecryption=False
        )['Parameter']['Value']

        image_file = ImageManager.correct_rotation(img_bytes, extension)  # rotating correction

        headers = {
            'X-OCR-SECRET': clova_secret_key
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

        dt3 = datetime.fromtimestamp(time.time())
        ocr_response = requests.request("POST", clova_api_url, headers=headers, data=payload, files=files).text
        ocr_response_json = json.loads(ocr_response)
        dt4 = datetime.fromtimestamp(time.time())
        logger.info(f"{dt3}~{dt4}: 이미지 OCR 완료")

        infer_texts = []
        for image in ocr_response_json["images"]:
            for field in image["fields"]:
                infer_texts.append(field["inferText"])
        result = ' '.join(infer_texts)
        return result

    except Exception as pe:
        logger.error("Error during OCR: %s", pe)
        raise HTTPException(status_code=500, detail="Error during OCR.")


@app.post("/upload/curriculum")
async def upload_curriculum_txt(upload_file: UploadFile = File(...)):
    extension = 'txt'
    try:
        if upload_file is None:
            raise HTTPException(status_code=400, detail="No input file")
        path = f'curriculum/math2015/{upload_file.filename}.{extension}'  # 경로 설정
        s3_client.upload_fileobj(upload_file.file, BUCKET_NAME, path)
    except Exception:
        raise HTTPException(status_code=500, detail="File upload failed")
    finally:
        upload_file.file.close()
        logger.info(f"커리큘럼 {upload_file.filename}이 정상적으로 업로드되었습니다.")
    return {"message": f"File {upload_file.filename} uploaded successfully",
            "path": path}


# OpenAI 연결
openai_secret_key = ssm_client.get_parameter(
    Name='/ono/fastapi/OPENAI_API_KEY',
    WithDecryption=False
)['Parameter']['Value']
openai_client = OpenAI(api_key=openai_secret_key)

# Mivlus DB 연결
SERVER = os.getenv('SERVER')
logger.info(f"* log >> 환경변수를 SERVER({SERVER})로 받아왔습니다.")

MILVUS_HOST = ssm_client.get_parameter(
    Name=f'/ono/{SERVER}/fastapi/MILVUS_HOST_NAME',
    WithDecryption=False
)['Parameter']['Value']
MILVUS_PORT = 19530
COLLECTION_NAME = 'Curriculum2015'
DIMENSION = 1536
INDEX_TYPE = "IVF_FLAT"


@app.get("/milvus/connect")
async def connect_milvus():
    try:
        # Milvus 서버 연결
        dt1 = str(datetime.fromtimestamp(time.time()))
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)  # server 용
        # connections.connect(host="127.0.0.1", port=19530, db="default")  # localhost 용
        dt2 = str(datetime.fromtimestamp(time.time()))
        logger.info(f"{dt1} ~ {dt2}: Milvus 서버 {MILVUS_HOST}:{MILVUS_PORT}에 연결 완료")

        # 컬렉션의 스키마 출력
        if utility.has_collection(COLLECTION_NAME):
            collection = Collection(COLLECTION_NAME)
            logger.info(f"* 존재하는 Collection {COLLECTION_NAME} Schema:")
            for field in collection.schema.fields:
                logger.info(f"    - Field Name: {field.name}, Data Type #: {field.dtype}")

    except Exception as e:
        logger.error(f"Failed to connect to Milvus server: {str(e)}")


@app.get("/milvus/create")
async def create_milvus():
    await connect_milvus()  # milvus 서버 연결

    # 스키마 및 컬렉션 생성
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name='content_embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name='subject_name', dtype=DataType.VARCHAR, max_length=100),  # Meta Data1
        FieldSchema(name='unit_name', dtype=DataType.VARCHAR, max_length=100),  # Meta Data2
        FieldSchema(name='main_concept', dtype=DataType.VARCHAR, max_length=100),  # Meta Data3
    ]
    schema = CollectionSchema(fields=fields, description='2015 Korean High School Curriculum Collection')
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    logger.info(f"* log >> New Collection [{COLLECTION_NAME}] is created.")

    # 인덱스 생성
    index_params = {  # 벡터 인덱스
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
    logger.info("* 생성할 Collection Schema:")
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
async def insert_curriculum_embeddings(subject: str):
    """ s3에서 교과과정을 읽고 임베딩하여 Milvus에 삽입 """
    # Milvus 연결
    await connect_milvus()
    collection = Collection(COLLECTION_NAME)

    # S3 내 커리큘럼 데이터 로드
    texts, subject_names, unit_names, main_concepts = [], [], [], []
    prefix = f'curriculum/{subject}2015/'  # 경로
    try:
        # 버킷에서 파일 목록 가져오기
        s3_curriculum_response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        for item in s3_curriculum_response.get('Contents', []):
            s3_key = item['Key']
            if s3_key.endswith('.txt'):
                # S3 객체 가져오기
                obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
                # 텍스트 읽기
                data = obj['Body'].read().decode('utf-8')
                lines = data.splitlines()
                # 메타 데이터 추출
                meatdata_lines = [line.strip('#').strip() for line in lines[:3]]
                subject_name = meatdata_lines[0]
                subject_names.append(subject_name)
                unit_name = meatdata_lines[1]
                unit_names.append(unit_name)
                main_concept = meatdata_lines[2]
                main_concepts.append(main_concept)
                # 교과과정 내용 추출
                text = '\n'.join(lines[3:]).strip()
                texts.append(text)
        logger.info(f"* log >> read {len(texts)} texts from S3")
    except Exception as e:
        logger.error(f"Error reading curriculum from S3: {e}")

    # 교과과정 내용 임베딩
    content_embeddings = get_embedding(openai_client, texts)
    logger.info(f"* log >> embedding 완료. dimension: {DIMENSION}")

    # 데이터 삽입
    data = [
        texts,  # content 필드
        content_embeddings,  # content_embedding 필드
        subject_names,  # subject_name 필드
        unit_names,  # unit_name 필드
        main_concepts  # main_concept 필드
    ]
    collection.insert(data)

    logger.info(f"* log >> 데이터 삽입 완료")
    return {"message": f"Curriculum inserted successfully"}


@app.get("/analysis/retrieve")
async def retrieve(problem_text: str):
    try:
        # collection을 메모리에 로드
        collection = Collection(COLLECTION_NAME)
        collection.load()

        # 검색 테스트
        query = problem_text
        dt5 = str(datetime.fromtimestamp(time.time()))
        query_embeddings = [get_embedding(openai_client, [query])]
        if not query_embeddings or query_embeddings[0] is None:
            raise ValueError("Embedding generation failed")
        dt6 = str(datetime.fromtimestamp(time.time()))
        logger.info(f"{dt5} ~ {dt6}: 쿼리 임베딩 완료")

        search_params = {
            'metric_type': 'COSINE',
            'params': {
                'radius': 0.35,
                'range_filter': 0.9
            },
        }
        dt5 = str(datetime.fromtimestamp(time.time()))
        results = collection.search(
            data=query_embeddings[0],
            anns_field='content_embedding',
            param=search_params,
            limit=2,
            expr=None,
            output_fields=['content', 'subject_name', 'unit_name', 'main_concept']
        )
        dt6 = str(datetime.fromtimestamp(time.time()))
        context = ' '.join([result.entity.get('content') for result in results[0]])
        subjects_list = [result.entity.get('subject_name') for result in results[0]]
        unit_list = [result.entity.get('unit_name') for result in results[0]]
        main_concept_list = [result.entity.get('main_concept') for result in results[0]]
        logger.info(f"{dt5} ~ {dt6}: 검색 완료")
        logs = ""
        for result in results[0]:
            logs += ("\n"+f"Score : {result.distance}, \
            \nInfo: {result.entity.get('subject_name')}\
             > {result.entity.get('unit_name')}\
            > {result.entity.get('main_concept')}, \
            \nText : {result.entity.get('content')}"+"\n\n")
        logger.info(f"* log >> 검색 결과: {logs}")

        return context, subjects_list, unit_list, main_concept_list
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/augmentation")
async def augment(curriculum_context, query):
    prompt = ("너는 고등학생의 오답 문제를 통해 약점을 보완해주는 공책이야. 문제 풀이하면 안돼. \
              교육과정을 참고해서 이 문제의 핵심 의도를 바탕으로 문제에서 헷갈릴만한 요소, \
              이 문제를 틀렸다면 놓쳤을 수 있는 중요한 개념을 유추해서 그 개념에 대해 4줄 이내로 설명해. \
              그 개념을 습득하기에 필요한 관련 개념들도 몇 개 소개해줘.  \
              만약 오답 문제와 교과과정이 관련이 없는 과목 같다고 판단되면, 교육과정은 참고하지 않으면 돼. \n\n\n")
    passage = f"오답 문제 : {query} \n\n\n"
    context = f"교과과정 : {curriculum_context} \n\n\n"
    augmented_query = prompt + passage + context
    return augmented_query


@app.get("/analysis/generation")
async def generate(question):
    def get_chatgpt_response(client, question, model="gpt-4o-mini"):
        try:
            dt7 = str(datetime.fromtimestamp(time.time()))
            gpt_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user",
                     "content": question
                     }
                ],
                temperature=0.6
            )
            dt8 = str(datetime.fromtimestamp(time.time()))
            logger.info(f"{dt7} ~ {dt8}: LLM 응답 완료")
            return gpt_response.choices[0].message.content
        except Exception as e:
            logger.info(f"Error during GPT querying: {e}")
            return None

    chatgpt_response = get_chatgpt_response(openai_client, question)
    logger.info(f"* log >> 응답 결과 \n {chatgpt_response}")
    return chatgpt_response
