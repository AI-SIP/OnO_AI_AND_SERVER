from fastapi import HTTPException, UploadFile, status


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
