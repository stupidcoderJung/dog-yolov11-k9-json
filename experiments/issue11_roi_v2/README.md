# Issue 11 ROI v2 Hybrid Experiments

이 디렉터리는 이슈 #11(품종/confidence 보강 ROI v2) 실험용 모델 모듈입니다.
기존 코어 파일(`dog_yolov11.py`, `train.py`, `val.py`)을 수정하지 않고,
실험 모델/캘리브레이션/데모를 독립 경로로 검증할 수 있도록 구성했습니다.

## 파일 구성

- `roi_v2_adapter.py`
  - `RoiV2HybridExperimentModel`
  - detector + ROI attribute head 결합
  - score 정책 비교 지원
    - `obj_x_breed`
    - `calibrated_obj_x_breed`
    - `breed_only`
  - 추론 출력에 `objectness`, `breed_confidence`, `final_confidence`를 명시 기록

- `calibration.py`
  - `fit_binary_temperature(...)`
  - `apply_temperature_to_probability(...)`
  - `expected_calibration_error(...)`, `brier_score(...)`

- `smoke_roi_v2.py`
  - synthetic 배치로 `forward -> loss -> backward` 검증
  - score policy/temperature/lambda 실험 인자 제공

- `infer_roi_v2_demo.py`
  - 랜덤 입력에서 ROI v2 추론 경로 점검
  - confidence 구성요소 출력 확인

## 빠른 실행

```bash
python3 experiments/issue11_roi_v2/smoke_roi_v2.py \
  --batch 2 \
  --img-h 640 --img-w 640 \
  --lambda-attr-grid 0.35 \
  --lambda-attr-roi 1.0 \
  --score-policy obj_x_breed
```

캘리브레이션 score 정책 실험 예시:

```bash
python3 experiments/issue11_roi_v2/infer_roi_v2_demo.py \
  --score-policy calibrated_obj_x_breed \
  --temperature 1.8 \
  --obj-thres 0.01 \
  --conf-thres 0.01
```

## 참고

- ROI head/loss는 `experiments/issue8_roi_attr` 모듈을 재사용합니다.
- 본 디렉터리는 실험용이므로, 학습 파이프라인 본선 통합 전 검증 단계에 사용합니다.
