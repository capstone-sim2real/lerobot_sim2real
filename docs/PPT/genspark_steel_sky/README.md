# Steel / Sky → Genspark

1. ZIP을 풀고 Genspark AI Slides에서 새 프로젝트를 연다. Professional 모드를 선택한다.
2. 먼저 01_design_reference.png, 02_slide_brief.docx, 03_source_report.pdf를 첨부한다. 이미지 첨부가 어려우면 PNG를 채팅에 붙여넣는다. 로컬 파일 경로만 입력하면 Genspark가 파일을 읽을 수 없다.
3. 04_prompt.txt 내용을 입력한다. 상세 제작 지침을 이미 첨부했으므로 기획을 처음부터 다시 맡길 필요가 없다.
4. 1/7/9번의 디자인을 확인한다. 필요하면 색상, 여백, 제목 크기 등 바꾸려는 항목을 구체적으로 지정한다.
5. 디자인이 맞으면 05_continue_prompt.txt를 입력해 14장을 완성한다.
6. 사진 추출 화질이 부족하면 assets에서 해당 원본 사진만 추가 첨부한다. JSON/CSV 첨부를 지원하지 않는 경우에는 보고서와 제작 지침의 수치를 사용한다.
7. PPTX를 내보낸 뒤 실제 발표 프로그램에서 한글 폰트, 줄바꿈, 사진 크기, 발표자 노트를 확인한다.

02_slide_brief.md는 같은 내용의 읽기/수정용 사본이다. DOCX를 첨부할 수 없으면 이 텍스트를 채팅에 붙여넣는다. ZIP 자체의 자동 해석을 전제로 하지 않는다.

최신 보고서와 선택한 시안을 이 폴더에 복사한 스냅샷이다. 이후 원본 보고서를 수정하면 Genspark에도 수정된 PDF를 다시 첨부한다. 각 복사본의 원본 경로와 SHA-256은 manifest.json에 있다.

서비스 안내 확인일: 2026-09-11. 공식 안내상 AI Slides는 PDF/Word/PPT 입력을 지원하며, Professional 모드에서 직접 편집을 제공한다. PPTX/PDF 내보내기는 유료 플랜이 필요하다.
https://www.genspark.ai/helpcenter/ai-slides
https://www.genspark.ai/docs/ai_slides_faq
