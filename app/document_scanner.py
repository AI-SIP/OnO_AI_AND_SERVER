import cv2

def detect_and_draw_box(frame):
    # 이미지를 흑백으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Canny 에지 검출기를 사용하여 에지 검출
    edged = cv2.Canny(gray, 30, 50)
    cv2.imshow('Edged', edged)

    # 외부 윤곽선만 검출
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    for contour in contours:
        # 윤곽선을 근사화
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # 근사화된 윤곽선이 4개의 점을 가진 경우 (사각형)
        if len(approx) == 4:
            # 윤곽선 그리기
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 6)
            # 사각형 내부를 연두색으로 채우기
            cv2.fillPoly(frame, [approx], (0, 255, 0))
            detected = True
            break

    # 사각형이 감지되지 않은 경우, 메시지 출력
    if not detected:
        cv2.putText(frame, "No square detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def main():
    # 카메라 캡처 객체 생성
    cap = cv2.VideoCapture(1)

    while True:
        # 프레임 별로 읽기
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 사각형 검출 및 그리기
        frame = detect_and_draw_box(frame)
        # 결과 이미지 표시
        cv2.imshow('Frame', frame)

        # ESC 키를 누르면 종료
        key = cv2.waitKey(1)
        if key == 27:
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
