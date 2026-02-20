# DogYOLOv11 K9 JSON

강아지 인지 모델(`DogYOLOv11`)과 JSON 어노테이션 계약을 맞춘 학습/추론 베이스 코드입니다.

## 현재 상태
- 모델 구조: `DogYOLOv11` (stride 8/16/32 head)
- 손실 함수: `DogYOLOLoss` (body box + head relative box + breed/emotion/action)
- 타깃 변환: `annotations_to_target()` (JSON -> 학습 포맷)
- 결과 디코딩: `decode_dog_predictions()` (예측 -> JSON 포맷)
- ADR: `docs/adr/0001-dog-yolov11-json-contract.md`
- 로드맵 이슈: [#1](https://github.com/stupidcoderJung/dog-yolov11-k9-json/issues/1)

## JSON 계약 포맷
입력/출력 공통 포맷은 아래 키를 사용합니다.

- `label`
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

## 빠른 시작
`torch` 환경이 없다면 먼저 가상환경을 구성하세요.

```bash
cd /Users/jipibe.j/Documents/insta-crawl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install torch torchvision
```

## 스모크 테스트
아래 스크립트로 모델 forward/loss/decode를 한 번에 확인할 수 있습니다.

```bash
cd /Users/jipibe.j/Documents/insta-crawl
python - <<'PY'
import torch
from dog_yolov11 import DogYOLOv11, DogYOLOLoss, annotations_to_target, decode_dog_predictions

annotations = [
    {"label":"Border Collie","bodybndbox":[366,750,503,911],"headbndbox":[403,750,462,820],"emotional":"excited","action":"running"},
    {"label":"Poodle","bodybndbox":[662,794,756,915],"headbndbox":[662,811,702,860],"emotional":"curious","action":"standing"},
    {"label":"Unknown","bodybndbox":[177,648,230,701],"headbndbox":[0,0,0,0],"emotional":"calm","action":"resting"},
]

breed_to_idx = {"Border Collie":0, "Poodle":1}
emotion_to_idx = {"excited":0, "curious":1, "calm":2, "resting":3}
action_to_idx = {"running":0, "standing":1, "resting":2}

target = annotations_to_target(
    annotations,
    breed_to_idx=breed_to_idx,
    emotion_to_idx=emotion_to_idx,
    action_to_idx=action_to_idx,
    unknown_breed_policy="ignore"
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = DogYOLOv11(num_breeds=2, num_emotions=4, num_actions=3).to(device)
loss_fn = DogYOLOLoss(num_breeds=2, num_emotions=4, num_actions=3).to(device)

x = torch.randn(1, 3, 1024, 768, device=device)
preds = model(x)
loss = loss_fn(preds, [target], img_size=(1024, 768))
decoded = decode_dog_predictions(
    preds,
    image_size=(1024, 768),
    breed_names=["Border Collie", "Poodle"],
    emotion_names=["excited", "curious", "calm", "resting"],
    action_names=["running", "standing", "resting"],
    conf_thres=0.95,
    max_det=5
)

print("pred_shapes:", [tuple(p.shape) for p in preds])
print("loss:", float(loss))
print("decoded_count:", len(decoded[0]))
PY
```

## 사용 흐름
학습 시:

1. 원본 JSON 어노테이션 로드
2. `annotations_to_target()`으로 모델 타깃 변환
3. `loss_fn(preds, targets, img_size=(H, W))` 계산

추론 시:

1. `preds = model(images)`
2. `decode_dog_predictions(...)` 호출
3. JSON 계약 포맷으로 결과 저장

## 주의사항
- `Unknown` 라벨은 기본적으로 분류 loss에서 제외(`ignore`)됩니다.
- head 박스가 없으면(`0,0,0,0`) head loss는 자동 제외됩니다.
- 현재 디코더는 NMS를 포함하지 않습니다.
