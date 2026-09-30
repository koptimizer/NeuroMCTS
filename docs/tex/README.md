# docs/tex 구성 (2026-09-30 정리)

| 위치 | 내용 | 생성/수정 방법 |
|---|---|---|
| `260923_paper.tex/.pdf` | **현재 논문** (간결판, elsarticle 2단, 10쪽). 수정 시 복사 후 오늘 날짜 이름으로 새 파일을 만든다 | `pdflatex` (elsarticle.cls·.bst 동봉) |
| `260923_paper_kr.pdf` | 현재 논문의 국문판 | `proposed_src/v22_transfer/make_260923_kr.py` |
| `fig/260923/` | 현재 논문 그림 4종 (pdf·png) | `proposed_src/v24_deploy_range/make_260923_figures.py` |
| `detailed/` | 상세판 편집본 (유지 중, 부록/보충 후보): `LPneuroBLS_paper_els` 42쪽 영문 원천, `LPneuroBLS_paper_kr.pdf`, `SWEVO_EN_v22`(파생, 손 편집 금지), `SWEVO_KR_v22.tex`(로컬 컴파일 불가) | `make_paper_kr.py`, `make_swevo_en.py`, `make_swevo_kr.py`; 그림은 `../../../figures/` |
| `legacy/` | 이전 사이클 보고서 v5–v11, v22, 초기 논문 `LPneuroBLS_paper`, `instance_hardness_and_kernel_pump` | 수정하지 않음 |
| `legacy/260917/` | 260917 동결 스냅샷 (tex·pdf·국문 pdf·그림) | `make_260917_kr.py`, `make_260917_figures.py` (경로 갱신됨) |
| `SKILL.md`, `paper_image_skill.md` | 논문·그림 작성 지침 | — |

하위 폴더에서 컴파일할 때는 `TEXINPUTS=..: pdflatex <file>.tex`로 상위의 elsarticle.cls를 찾게 한다 (detailed/ 42쪽 판 컴파일·그림 경로 확인 완료).

빌드 중간 산출물(.aux .log .out .spl .fls .fdb_latexmk .synctex.gz .bbl .blg)은 추적하지 않는다.
