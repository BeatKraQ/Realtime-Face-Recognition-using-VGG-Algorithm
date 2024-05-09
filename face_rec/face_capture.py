import os
import sys
import glob
import cv2

def save_face(frame, p1, p2, filename):
    cp = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)  # 중심점 계산

    w = p2[0] - p1[0]  # 너비 계산
    h = p2[1] - p1[1]  # 높이 계산

    # 얼굴 영역의 비율 조정
    if h * 3 > w * 4:
        w = round(h * 3 / 4)
    else:
        h = round(w * 4 / 3)

    # 얼굴 영역 추출
    x1 = cp[0] - w // 2
    y1 = cp[1] - h // 2
    if x1 < 0 and y1 < 0:
        return
    if x1 + w >= frame.shape[1] or y1 + h >= frame.shape[0]:
        return

    crop = frame[y1:y1+h, x1:x1+w]
    crop = cv2.resize(crop, dsize=(150, 200), interpolation=cv2.INTER_CUBIC)  # 크기 조정
    cv2.imwrite(filename, crop)  # 파일 저장

# 비디오 캡처 시작
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera open failed!')  # 카메라 열기 실패
    sys.exit()

# 네트워크 로드
model = '../opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = '../opencv_face_detector/deploy.prototxt'

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')  # 네트워크 로드 실패
    sys.exit()

# 출력 디렉토리 설정 및 파일 인덱스
outdir = 'output'
prefix = outdir + '/face_'
file_idx = 1

try:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
except OSError:
    print('Output folder create failed!')  # 출력 폴더 생성 실패

png_list = glob.glob(prefix + '*.png')
if len(png_list) > 0:
    png_list.sort()
    last_file = png_list[-1]
    file_idx = int(last_file[-8:-4]) + 1

# 프레임 읽기
cnt = 0
while True:
    _, frame = cap.read()
    if frame is None:
        break

    # 얼굴 인식
    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detect = net.forward()

    detect = detect[0, 0, :, :]
    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.8:
            break

        # 얼굴 발견!
        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        # 얼굴 이미지를 png 파일로 저장
        cnt += 1

        if cnt % 10 == 0:
            filename = '{0}{1:04d}.png'.format(prefix, file_idx)
            save_face(frame, (x1, y1), (x2, y2), filename)
            file_idx += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))  # 얼굴 표시

        label = 'Face: %4.3f' % confidence
        cv2.putText(frame, label, (x1, y1 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)  # 프레임 표시

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()  # 창 닫기
