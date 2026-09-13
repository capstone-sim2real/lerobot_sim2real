# 검증 기록

## 확인한 항목

- 원본 PPTX SHA-256 유지: 확인
- 수정 PPTX 14장, 미리보기 PDF 14쪽: 확인
- 원본 14장과 수정본 14장의 렌더링 이미지를 view_image로 확인
- 최종 여백 수정 후 캘리브레이션, 적재, 결론과 영상 페이지 재확인
- PPTX ZIP 무결성: 이상 없음
- 모든 본문 텍스트의 PDF 추출 대응: 누락 0개
- 슬라이드 바깥으로 벗어난 개체: 0개
- 발표자 노트: 14장 모두 존재
- 동영상 개체: 수정본 6, 11, 12장에 각각 1개 존재
- 원본 동작/관제 영상: 바이트 동일
- 적재 영상: H.264, 1920×1080, 1,631프레임 유지
- 적재 영상 오디오: 원본 AAC 스트림과 ADTS 추출 해시 동일
- 미리보기 PDF: 영상 주석 제거, 텍스트 일치 확인
- 원본 파일과 기존 워크트리 변경사항 보존

## 파일 식별 정보

- PPTX SHA-256: `a5bc3f540e05c4b69da553eaab72c6d8fdb1c613e6c0b2c068f8fcdb072526aa`
- PDF SHA-256: `bec1db9dd635d8990bf1af7172a7c318fe37e738f7e2fb5c65e0f7357aed0a14`

## 확인 범위의 한계

LibreOffice의 PDF 내보내기와 Poppler 이미지 렌더링으로 배치와 글자를 확인했다. Microsoft PowerPoint에서의 실제 클릭 재생과 발표 장소의 프로젝터 출력은 실행하지 않았다. 영상 길이를 실제 미션 소요시간으로 해석하거나 새 성공률을 산출하지 않았다.

상세 기계 판독 결과: [validation.json](analysis/validation.json), [영상 호환성 확인](analysis/media_compatibility.json)
