# Issue 8 ROI Attribute Experiments

이 디렉터리는 이슈 #8(ROIAlign 기반 Object-Centric Attribute Head) 실험용 코드입니다.
기존 코어 파일(`dog_yolov11.py`, `train.py`, `val.py`)을 직접 수정하지 않고,
별도 모듈로 다양한 조합을 검증할 수 있도록 구성했습니다.

## 실행 환경 메모

- 운영 대상: GPU 서버(CUDA) 기준
- 로컬 검증: `--device cpu` 스모크 테스트만 수행
- 본 실험 폴더는 MPS 최적화/호환을 목표로 하지 않습니다.

## 파일 구성

- `roi_attr_head.py`
  - `DogRoiAttrHead`
  - body/head box를 받아 ROIAlign 후 emotion/action(옵션 breed) 분류
  - 옵션:
    - `fusion`: `concat` | `xattn`
    - `use_multiscale`: p3/p4/p5 레벨 선택

- `roi_attr_adapter.py`
  - `DogYoloWithFeatures`: 기존 `DogYOLOv11`에서 `preds + (p3,p4,p5)`를 반환하는 어댑터
  - `RoiAttrExperimentModel`: detection + roi-attr head 결합
  - `infer_with_roi_attributes(...)`: NMS 결과에 ROI 분류를 덮어쓰는 실험용 경로

- `roi_attr_loss.py`
  - `RoiAttributeLoss`
  - ROI 분류 로스(emotion/action + optional breed)
  - `combine_grid_and_roi_attr_loss(...)` 유틸

- `smoke_roi_attr.py`
  - synthetic 배치로 `forward -> loss -> backward` 확인
  - 실험 옵션 다수 제공

- `infer_roi_attr_demo.py`
  - 랜덤 입력에서 `decode_dog_predictions + ROI attribute override` 경로 점검
  - JSON 계약 키(`label/bodybndbox/headbndbox/emotional/action`) 유지 확인

## 빠른 실행

```bash
python3 experiments/issue8_roi_attr/smoke_roi_attr.py \
  --batch 2 \
  --img-h 640 --img-w 640 \
  --roi-fusion concat \
  --lambda-attr-grid 0.0 \
  --lambda-attr-roi 1.0
```

멀티스케일 + xattn 실험 예시:

```bash
python3 experiments/issue8_roi_attr/smoke_roi_attr.py \
  --roi-use-multiscale \
  --roi-fusion xattn \
  --with-breed-head \
  --roi-output-size 5 \
  --roi-hidden-dim 192
```

추론 경로 데모:

```bash
python3 experiments/issue8_roi_attr/infer_roi_attr_demo.py \
  --roi-fusion concat \
  --obj-thres 0.01 \
  --conf-thres 0.01
```

## 실험 축

- 구조
  - `--roi-fusion concat|xattn`
  - `--roi-use-multiscale`
  - `--roi-output-size`
  - `--roi-hidden-dim`

- 로스 가중치
  - `--lambda-attr-grid`
  - `--lambda-attr-roi`

- 헤드 출력
  - `--with-breed-head`

## 주의사항

- ROIAlign는 `torchvision.ops.roi_align` 사용합니다.
- head box가 무효(`x2<=x1` 또는 `y2<=y1`)이면 head token은 0-vector로 처리합니다.
