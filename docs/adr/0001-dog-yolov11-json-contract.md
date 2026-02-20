# ADR 0001: DogYOLOv11 JSON 입출력 정합 설계

- 상태: Accepted
- 날짜: 2026-02-20
- 작성자: Codex

## 배경
강아지 데이터셋 어노테이션 포맷은 아래 JSON 구조를 기준으로 정의되어 있다.

- `label`
- `bodybndbox` (xyxy)
- `bodybndbox_coco` (xywh)
- `headbndbox` (xyxy)
- `headbndbox_coco` (xywh)
- `emotional`
- `action`

모델 학습/추론 파이프라인이 이 계약을 일관되게 따르지 않으면 학습 오류와 품질 저하가 발생한다.

## 문제
초기 구현에서 다음 이슈가 있었다.

1. 모델 출력 스케일 순서와 loss의 stride 해석 불일치
2. loss가 픽셀 좌표 입력을 직접 처리하지 못함
3. `headbndbox=[0,0,0,0]` 같은 무효 head 케이스 미처리
4. `label=Unknown` 같은 미정 라벨 처리 기준 부재
5. 추론 결과를 JSON 계약 포맷으로 내보내는 표준 디코더 부재

## 결정
아래 설계로 정합성을 맞춘다.

1. 모델 출력을 stride 기준 `o8 -> o16 -> o32` 순서로 고정한다.
2. loss는 `img_size=(H,W)`와 feature map 크기를 이용해 픽셀 좌표를 직접 grid/offset으로 변환한다.
3. head 박스 유효성 마스크(`head_valid`)를 도입해 무효 head는 head loss에서 제외한다.
4. 분류 loss는 `ignore_index=-100`을 지원하고 `Unknown`은 정책(`ignore` 또는 `class`)으로 매핑한다.
5. 입력 JSON을 학습 타깃으로 변환하는 `annotations_to_target()`을 제공한다.
6. 모델 raw output을 JSON 계약 포맷으로 변환하는 `decode_dog_predictions()`을 제공한다.

## 구현 위치
- `/Users/jipibe.j/Documents/insta-crawl/dog_yolov11.py`
  - `DogYOLOv11.forward()` 출력 순서 정렬
  - `DogYOLOLoss` 좌표/마스크/ignore 처리
  - `annotations_to_target()` 추가
  - `decode_dog_predictions()` 추가

## 결과
이제 학습 입력과 추론 출력이 동일한 JSON 계약을 공유한다.

- 입력: 어노테이션 JSON -> `annotations_to_target()` -> `DogYOLOLoss`
- 출력: 모델 예측 -> `decode_dog_predictions()` -> 어노테이션 호환 JSON

## 트레이드오프
1. IoU 기반 매칭이 아닌 단일 grid 매칭 방식이므로 군집 객체 밀집 상황에서 recall 저하 가능성이 있다.
2. NMS를 포함하지 않은 디코더이므로 후처리 단계에서 추가 정제가 필요할 수 있다.

## 후속 작업
1. NMS(클래스별/통합) 추가
2. `smoke_test.py` 고정 파일화 및 CI 연결
3. 데이터 검증기(박스 범위/라벨 사전 검증) 추가
