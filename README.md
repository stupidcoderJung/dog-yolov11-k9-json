# DogYOLOv11 K9 JSON

강아지 탐지/속성 분류를 위한 경량 YOLO 커스텀 프로젝트입니다.

이 저장소는 아래 JSON 계약을 기준으로 동작합니다.

- `label` (견종)
- `bodybndbox` / `bodybndbox_coco`
- `headbndbox` / `headbndbox_coco`
- `emotional`
- `action`

기본 목표:

- 한 이미지에서 여러 마리 강아지 탐지
- `label=견종` 분류 (Stanford Dogs 120종 확장 가능)
- `headbndbox` + 감정/행동 분류
- 학습/추론 출력을 같은 JSON 형태로 유지

## 1) 현재 포함 기능

- 모델/로스: `dog_yolov11.py`
- 디코더 + NMS: `decode_dog_predictions(...)`
- 데이터셋 로더: `dataset.py` (`DogJsonDataset`, `dog_collate_fn`)
- 라벨 검증기: `label_validator.py`
- 학습 스크립트: `train.py`
- 검증/추론 스크립트: `val.py`
- 빠른 스모크: `smoke_test.py`
- CI 스모크 테스트: `.github/workflows/smoke-test.yml`

## 2) 설치

```bash
git clone https://github.com/stupidcoderJung/dog-yolov11-k9-json.git
cd dog-yolov11-k9-json
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install "numpy<2"
pip install torch torchvision
```

## 3) 먼저 스모크 테스트

```bash
python smoke_test.py --img-h 640 --img-w 640 --batch 1 --num-breeds 120 --num-emotions 5 --num-actions 5 --width-mult 0.23
```

정상 출력 포인트:

- `pred_shapes: [...]`
- `loss: <float>`
- `backward: ok`
- `nms_before_after_image0: <before> -> <after>`

## 4) 데이터 포맷 (Manifest)

`manifest.json`은 리스트(또는 `{ "samples": [...] }`)를 받습니다.

```json
[
  {
    "image": "images/sample_0001.jpg",
    "annotations": [
      {
        "label": "Border Collie",
        "bodybndbox": [366, 750, 503, 911],
        "bodybndbox_coco": [366, 750, 137, 161],
        "headbndbox": [403, 750, 462, 820],
        "headbndbox_coco": [403, 750, 59, 70],
        "emotional": "excited",
        "action": "running"
      }
    ]
  }
]
```

경로 규칙:

- `image`는 manifest 기준 상대경로 또는 절대경로
- 학습 시 지정한 `--img-h`, `--img-w`에 맞춰 이미지/박스가 자동 리사이즈

## 5) 라벨 검증기 사용

실제 학습 전, 어노테이션 품질 점검:

```bash
python label_validator.py \
  --manifest /path/to/manifest.json \
  --breed-names /path/to/breeds_120.txt \
  --emotion-names /path/to/emotions.txt \
  --action-names /path/to/actions.txt \
  --allow-unknown-breed
```

검증 내용:

- bbox 구조/면적 유효성
- 이미지 범위 초과 여부
- 클래스 사전에 없는 라벨
- head가 body 밖에 있는 경우 경고

## 6) 학습 실행 (`train.py`)

### 6-1) 실제 데이터로 학습

```bash
python train.py \
  --manifest /path/to/manifest.json \
  --breed-names /path/to/breeds_120.txt \
  --emotion-names /path/to/emotions.txt \
  --action-names /path/to/actions.txt \
  --validate-labels \
  --allow-unknown-breed \
  --epochs 1 \
  --batch-size 2 \
  --img-h 640 --img-w 640 \
  --width-mult 0.23
```

### 6-2) 샘플(합성) 데이터로 1 epoch 빠르게 확인

```bash
python train.py \
  --synthetic-samples 8 \
  --epochs 1 \
  --batch-size 2 \
  --img-h 640 --img-w 640 \
  --width-mult 0.23 \
  --max-steps-per-epoch 2
```

학습 결과:

- `runs/train/<run-name>-<timestamp>/last.pt`
- `runs/train/<run-name>-<timestamp>/class_names.json`
- `runs/train/<run-name>-<timestamp>/train_config.json`
- synthetic 사용 시 `synthetic_dataset/manifest.json`도 생성

## 7) 검증/추론 실행 (`val.py`)

```bash
python val.py \
  --manifest /path/to/manifest.json \
  --checkpoint /path/to/last.pt \
  --img-h 640 --img-w 640 \
  --conf-thres 0.25 \
  --iou-thres 0.50 \
  --output-json runs/val/predictions.json
```

출력:

- 평균 loss
- NMS 전/후 총 박스 수 (`nms_before_total`, `nms_after_total`)
- 이미지별 JSON 예측 결과 파일

## 8) 클래스 파일 예시

`breeds_120.txt` (한 줄에 하나):

```text
Border Collie
Poodle
...
```

`emotions.txt`:

```text
excited
curious
calm
resting
other
```

`actions.txt`:

```text
running
standing
resting
walking
playing
```

## 9) NMS 간단 설명

NMS(Non-Maximum Suppression)는 같은 객체에 대해 중복 예측된 박스 중 점수가 낮은 박스를 제거하는 후처리입니다.

- `apply_nms=True/False`로 on/off
- `class_agnostic=False`면 클래스별 NMS
- `iou_thres`로 중복 판정 강도 조절

## 10) 1.5M 미만 파라미터 목표

기본값 `width_mult=0.23` 기준으로 약 1.5M 미만 구성을 목표로 했습니다.

확인:

```bash
python smoke_test.py --width-mult 0.23 --num-breeds 120 --num-emotions 5 --num-actions 5
```

## 11) 로드맵

- 이슈: [#1](https://github.com/stupidcoderJung/dog-yolov11-k9-json/issues/1)
- ADR: `docs/adr/0001-dog-yolov11-json-contract.md`
