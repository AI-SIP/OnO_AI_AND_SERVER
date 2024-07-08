FROM ubuntu:latest
LABEL authors="semin"
ENTRYPOINT ["top", "-b"]

#
FROM python:3.12

# 작업 디렉토리 설정
WORKDIR /ono

# 의존성 파일 복사
COPY ./requirements.txt /ono/requirements.txt

# 의존성 설치
RUN pip install --no-cache-dir --upgrade -r /ono/requirements.txt

# 현재 디렉토리의 모든 파일을 컨테이너의 작업 디렉토리로 복사
COPY ./app /ono/app/

# 컨테이너 실행 시 실행할 명령어
CMD ["uvicorn", "app.main:app","--proxy-headers", "--host", "0.0.0.0", "--port", "80"]