# 파이썬 이미지 사용
FROM python:3.12-slim

# 라벨 추가
LABEL org.opencontainers.image.source="https://github.com/AI-SIP/MVP_CV"

# 필요한 시스템 패키지 먼저 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    python3-dev && \
    rm -rf /var/lib/apt/lists/* \

# 작업 디렉토리 설정
WORKDIR /test

# 의존성 파일 복사 및 설치
COPY ./requirements.txt /test/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /test/requirements.txt

# 애플리케이션 코드 복사
COPY ./app /test/app/

# 실행할 위치
WORKDIR /test/app

# 컨테이너 실행 시 실행할 명령어
CMD ["bash", "-c", "mkdir -p /test/models && uvicorn main:app --host 0.0.0.0 --port 8000"]