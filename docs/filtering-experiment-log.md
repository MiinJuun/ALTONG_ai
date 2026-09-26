# 필터링 모델 실험 기록

각 실험은 같은 형식으로 아래에 추가한다. 점수는 해당 실험의 테스트 분할에서만 비교한다. 서로 다른 데이터셋이나 분할의 점수는 직접 비교하지 않는다. 예시는 합성 데이터의 실제 모델 출력이며 정답 라벨과 구분한다.

## 2026-09-26 — 51건 데이터, LoRA 2단계 시험

- 목적: 데이터 준비 → 학습 → 평가 경로와 모델 출력 형식 확인. 성능 개선을 입증하는 실험은 아님.
- 기반 모델: `Qwen/Qwen3-0.6B`
- 데이터: 합성 51건. 학습 분할 41건, 검증 5건, 테스트 5건. 데이터 SHA-256: `06cdef89942060b5979abde56f19ccdead0b278e5b04e380ee97612e0795efa6`.
- 학습: LoRA SFT, rank 8, alpha 16, dropout 0.05, `q_proj`·`v_proj` 대상. BF16, 최대 길이 768, seed 42, 2단계, batch size 1. 학습 분할에 41건이 있지만 이번 실행에서 실제 파라미터 업데이트에 사용된 샘플은 2건이다. 검증 분할은 학습 중 사용하지 않았다.
- 장치: NVIDIA GeForce RTX 3060 Laptop GPU. 최종 학습 loss: 2.041.
- 산출물: `filtering_training/outputs/lora-smoke-51/adapter` (로컬, Git 제외).
- 평가: 테스트 5건에 대해 원본 모델과 학습 어댑터를 같은 조건으로 실행했다.

| 지표 | 원본 모델 | 2단계 LoRA |
| --- | ---: | ---: |
| 유효한 JSON | 5/5 | 5/5 |
| 통과/차단 정답 | 2/5 | 2/5 |
| 긴급도 정확히 일치 | 0/5 | 0/5 |
| 관련도 정확히 일치 | 1/5 | 1/5 |
| 카테고리 macro F1 | 0.0 | 0.0 |
| 차단해야 할 알림을 통과시킨 비율 | 3/3 | 3/3 |

테스트가 5건이라 통계적으로 신뢰할 수 없으며, 2단계 학습 후 개선도 관찰되지 않았다. 아래 JSON은 **학습 어댑터가 실제 생성한 값**이다. 알림 ID와 정답은 비교를 위해 따로 적었다. `ai_summary_reason`을 포함한 JSON 값은 수정하지 않았다.

### 출력 예시 1 — noti_004

정답: 긴급 업무, 긴급도 4, 관련도 5, 통과. 실제 출력은 통과 결정만 같고 카테고리와 점수가 다르다.

```json
{
  "urgency_score": 3,
  "relevance_score": 4,
  "category": "일정/회의",
  "ai_summary_reason": "현재 AI 코드 수정이 필요하며, 알림 내용과 작업 주제가 일정/회의에 해당합니다."
}
```

### 출력 예시 2 — noti_048

정답: 광고/홍보, 긴급도 1, 관련도 1, 차단. 실제 출력은 할인 광고를 업무와 관련 있는 긴급 알림으로 잘못 판단해 통과시킨다.

```json
{
  "urgency_score": 4,
  "relevance_score": 5,
  "category": "일정/회의",
  "ai_summary_reason": "현재 작업은 'Code.exe'와 'WindowsTerminal.exe'가 활성 창을 사용하고 있으며, 'train.py' 파일이 실행 중입니다. 이 알림은 상품 할인 행사가 10분 남는 상황을 나타내며, 작업과 관련성이 높습니다."
}
```

### 출력 예시 3 — noti_003

정답: 시스템/보안, 긴급도 4, 관련도 1, 통과. 실제 출력은 통과 결정만 같고 카테고리·점수가 다르다.

```json
{
  "urgency_score": 3,
  "relevance_score": 4,
  "category": "일정/회의",
  "ai_summary_reason": "새로운 로그인 감지가 일정/회의 관련 작업에 영향을 미치며, 현재 작업과 일치합니다."
}
```

평가 결과 원본은 `filtering_training/outputs/evaluation/base_test_51.json`, 학습 어댑터는 `filtering_training/outputs/evaluation/adapter_test_51.json`, 예시 JSON은 `filtering_training/outputs/evaluation/adapter_examples_51.json`에 있다. 이 경로들은 Git에서 제외된다. 이 문서에 필요한 숫자와 예시를 보존한다.

다음 실험에서는 정답 라벨 검토 후 데이터 규모를 늘리고, 학습 단계 수와 평가 세트를 별도로 기록한다. 3,000건 데이터 제작 및 본학습은 아직 시작하지 않았다.
