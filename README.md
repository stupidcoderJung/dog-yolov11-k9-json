# DogYOLOv11 K9 JSON (초보자용 가이드)

YOLO를 처음 커스텀하는 분을 위한, 강아지 전용 모델 베이스 코드입니다.

이 저장소는 다음 목표를 가집니다.

- 한 이미지에서 여러 마리 강아지 탐지
- `label=견종` 분류
- `headbndbox` 예측
- `emotional`, `action` 분류
- 학습/추론 모두 동일한 JSON 포맷 사용

## 1) 이 프로젝트에서 중요한 전제

### `label`의 의미
`label`은 견종입니다.  
현재 설계는 **Stanford Dogs 120종** 사용을 전제로 합니다.

- 데이터셋 링크: [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- 모델 초기값: `num_breeds=120` (`dog_yolov11.py`)
- 기본 경량 설정: `width_mult=0.23` (약 1.45M params 목표)

### 현재 코드의 범위
현재 저장소는 **모델 코어와 변환 유틸 중심**입니다.

- 있음: 모델/로스/JSON 변환/디코더
- 아직 없음: `train.py`, `val.py`, 데이터셋 클래스
- 포함됨: 기본 NMS 후처리(`decode_dog_predictions()` 옵션)

로드맵은 이슈에서 관리 중입니다.

- 로드맵: [#1](https://github.com/stupidcoderJung/dog-yolov11-k9-json/issues/1)

## 2) 폴더 구조

```text
.
├── dog_yolov11.py                      # 모델, 로스, JSON 변환/디코더
├── smoke_test.py                       # 파라미터/forward/loss/decode 검증
├── docs/adr/0001-dog-yolov11-json-contract.md
└── README.md
```

## 3) JSON 어노테이션 계약

입력/출력 공통으로 아래 키를 씁니다.

- `label` (견종)
- `bodybndbox` (`[x1, y1, x2, y2]`)
- `bodybndbox_coco` (`[x, y, w, h]`)
- `headbndbox` (`[x1, y1, x2, y2]`, 없으면 `[0,0,0,0]`)
- `headbndbox_coco` (`[x, y, w, h]`)
- `emotional`
- `action`

예시:

```json
{
  "label": "Border Collie",
  "bodybndbox": [366, 750, 503, 911],
  "bodybndbox_coco": [366, 750, 137, 161],
  "headbndbox": [403, 750, 462, 820],
  "headbndbox_coco": [403, 750, 59, 70],
  "emotional": "excited",
  "action": "running"
}
```

## 4) 설치

```bash
cd /Users/jipibe.j/Documents/insta-crawl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install torch torchvision
```

## 5) 빠른 개념 정리

`dog_yolov11.py`의 핵심 구성:

- `DogYOLOv11`
  - 출력: stride 8/16/32 3개 스케일
  - 한 셀당 예측: `obj + body(4) + head(4) + breed + emotion + action`
  - 기본 `width_mult=0.23`로 1.5M 미만 파라미터를 목표로 함

- `DogYOLOLoss`
  - 픽셀 좌표 박스를 그리드 타깃으로 변환
  - `Unknown` 라벨은 `ignore_index=-100`으로 loss 제외 가능
  - head 박스가 없는 샘플(`0,0,0,0`)은 head loss 제외

- `annotations_to_target()`
  - JSON 어노테이션을 학습용 tensor 딕셔너리로 변환

- `decode_dog_predictions()`
  - 모델 raw output을 다시 JSON 계약 포맷으로 변환
  - 기본 NMS 지원 (`apply_nms`, `iou_thres`, `class_agnostic`)
  - 점수 계산: `objectness * breed_confidence`

- `model_size_report()`
  - 현재 설정의 파라미터 수를 확인하는 유틸

## 6) 스모크 테스트

아래 명령은 다음을 한 번에 확인합니다.

- 모델 forward
- loss 계산
- backward
- 디코딩 결과(JSON 형태) 생성
- 파라미터 수 출력

```bash
cd /Users/jipibe.j/Documents/insta-crawl
python smoke_test.py
```

정상이라면:

- `pred_shapes`가 3개 스케일로 출력
- `loss`가 숫자로 출력
- `backward: ok` 출력
- `decoded_count`가 0 이상으로 출력

원하는 설정으로 테스트:

```bash
python smoke_test.py --num-breeds 120 --num-emotions 5 --num-actions 5 --width-mult 0.23
```

## 7) 120 견종으로 확장하는 방법

핵심은 **클래스 순서 고정**입니다.

1. 견종 이름 리스트(`breed_names`)를 길이 120으로 고정
2. `breed_to_idx = {name: idx}` 매핑 생성
3. 모델/로스 생성 시 `num_breeds=120` 사용
4. 추론 디코딩 시 같은 `breed_names` 전달

예시:

```python
breed_names = [...]  # 길이 120, 순서 고정
breed_to_idx = {name: i for i, name in enumerate(breed_names)}

model = DogYOLOv11(num_breeds=120, num_emotions=5, num_actions=5, width_mult=0.23)
loss_fn = DogYOLOLoss(num_breeds=120, num_emotions=5, num_actions=5)
```

## 8) 최소 학습 루프 예시

```python
model.train()
for images, ann_lists in loader:
    preds = model(images)
    targets = [
        annotations_to_target(
            anns, breed_to_idx, emotion_to_idx, action_to_idx,
            unknown_breed_policy="ignore"
        )
        for anns in ann_lists
    ]
    loss = loss_fn(preds, targets, img_size=(images.shape[2], images.shape[3]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 9) 자주 하는 실수

- `img_size`를 `(H, W)`가 아니라 단일값으로 넘김
- `bodybndbox`가 `x2 <= x1` 또는 `y2 <= y1`
- 클래스 매핑 순서가 학습/추론에서 다름
- `Unknown` 정책을 정하지 않고 그대로 학습
- head가 없는 샘플에 임의 head 박스를 넣어 노이즈 증가

## 10) 참고 문서

- ADR: `docs/adr/0001-dog-yolov11-json-contract.md`
- 이슈: [#1 Roadmap](https://github.com/stupidcoderJung/dog-yolov11-k9-json/issues/1)
