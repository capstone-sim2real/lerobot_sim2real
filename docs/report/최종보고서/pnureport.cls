% !TEX encoding = UTF-8 Unicode
%% pnureport.cls (report를 바탕으로 함)
%% 부산대학교 컴퓨터공학과 Technical Report 공식 서식(워드 양식) 재현판 (pdfLaTeX 전용)

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pnureport}[2026/09/13 PNU CSE Technical Report class for pdfLaTeX]

% English 옵션(기본값: false = 국문)
\newif\ifPNU@english
\PNU@englishfalse

\DeclareOption{en}{\PNU@englishtrue}
\DeclareOption{english}{\PNU@englishtrue}

\DeclareOption{*}{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

\LoadClass[11pt,a4paper]{report} % 기본 클래스 로드; 서식의 본문 크기(22half-pt=11pt)와 동일

% --- 패키지 설정 ---
\ifPNU@english
\else
  \RequirePackage{kotex}                % 한글 지원
  \RequirePackage{indentfirst}          % 한글에 맞는, 첫 행 들여쓰기
\fi

\RequirePackage{amsmath}              % AMS 수학 기호
\RequirePackage{geometry}             % 여백 설정
\RequirePackage{setspace}             % 줄간격
\RequirePackage{graphicx}             % 그림 삽입
\RequirePackage{fancyhdr}             % 머리말/꼬리말
\RequirePackage{titlesec}             % 제목 스타일 조정
\RequirePackage{tocloft}              % 목차 설정
\RequirePackage{url}                  % url 설정 명령어
\RequirePackage{listings}             % 코드 삽입
\RequirePackage{makeidx}              % 색인 삽입
\RequirePackage{tikz}                 % TikZ 그래픽 삽입
\RequirePackage{xcolor}               % 16진법 색상 패키지

% --- 여백 설정 (서식 워드 문서의 pgMar: top 1701, left/right/bottom 1440, header 851, footer 992 twip) ---
\geometry{a4paper,top=3cm,bottom=2.54cm,left=2.54cm,right=2.54cm,headheight=14pt,headsep=0.55cm,footskip=1.1cm}

% --- 글꼴: 서식은 본문·제목 모두 맑은 고딕(고딕 계열) 사용 ---
% pdfLaTeX+kotex 환경에는 맑은 고딕이 없어 가장 가까운 고딕 계열인 나눔고딕(NanumGothic)으로 대체한다.
\ifPNU@english
\else
  \renewcommand{\familydefault}{\sfdefault}
\fi
\singlespacing % 서식의 줄간격(line 259 auto)은 단일 행간에 가깝다

% --- 제목 구성 ---
\newcommand{\reporttitle}[1]{\def\@reporttitle{#1}}
\newcommand{\reportteam}[1]{\def\@reportteam{#1}}
\newcommand{\reportauthor}[1]{\def\@reportauthor{#1}}
\newcommand{\reportdate}[1]{\def\@reportdate{#1}}
\newcommand{\reportadvisor}[1]{\def\@reportadvisor{#1}}
\newcommand{\reporttechid}[1]{\def\@reporttechid{#1}}
\reporttechid{2026-09}

\ifPNU@english
  \newcommand{\PNU@title@advisor}{Advisor}
  \newcommand{\PNU@title@univ}{Pusan National University}
  \newcommand{\PNU@title@dept}{Computer Science and Engineering}
  \newcommand{\PNU@title@report}{Technical Report}
\else
  \newcommand{\PNU@title@advisor}{지도교수}
  \newcommand{\PNU@title@univ}{Pusan National University}
  \newcommand{\PNU@title@dept}{Computer Science and Engineering}
  \newcommand{\PNU@title@report}{Technical Report}
\fi

% --- 표지: 서식 문서 1페이지 레이아웃을 그대로 재현 ---
\def\maketitle{
  \thispagestyle{pnutitle}
  \begin{flushright}
  \bfseries \PNU@title@univ\\ \PNU@title@dept\\ \PNU@title@report\ \@reporttechid
  \end{flushright}
  \vspace{2.6cm}
  \begin{center}
  {\bfseries\fontsize{20}{24}\selectfont \@reporttitle \par}
  \end{center}
  \vspace{1.8cm}
  \begin{center}
  \includegraphics[width=5.7cm]{pnu_emblem}
  \end{center}
  \vfill
  \begin{center}
  \@reportauthor\\[0.4cm]
  \PNU@title@advisor\ \ \@reportadvisor
  \end{center}
  \vspace{1.5cm}
  \clearpage
}

% --- 목차 제목 변경 ---
\ifPNU@english
  \renewcommand{\contentsname}{Contents}
\else
  \renewcommand{\contentsname}{목차}
\fi

% --- 목차: 모든 수준에 점선 리더, 제목은 가운데 정렬, 굵기는 본문과 동일 ---
\renewcommand{\cftchapfont}{\normalfont}
\renewcommand{\cftchappagefont}{\normalfont}
\renewcommand{\cftchapleader}{\cftdotfill{\cftchapdotsep}}
\renewcommand{\cftchapdotsep}{4.5}
\renewcommand{\cfttoctitlefont}{\hfill\Large\bfseries}
\renewcommand{\cftaftertoctitle}{\hfill}
\setlength{\cftaftertoctitleskip}{10pt}

\ifPNU@english
  \renewcommand{\chaptername}{} % Empty String
  \renewcommand{\thechapter}{\arabic{chapter}} % Only Aribic Numerals
\else
  \renewcommand{\chaptername}{} % 빈 문자열로 설정해 불필요한 출력 방지
  \renewcommand{\thechapter}{\arabic{chapter}} % 숫자만 유지
\fi

% --- 장/절/항 제목 형식: "N. 제목" / "N.M. 제목" / "N.M.K. 제목" (서식 heading 1/2/3과 동일) ---
\titleformat{\chapter}[hang]{\bfseries\fontsize{14}{18}\selectfont}{\thechapter.}{0.6em}{}
\titlespacing*{\chapter}{0pt}{0pt}{18pt}
\titleformat{\section}[hang]{\bfseries\fontsize{12}{16}\selectfont}{\thesection.}{0.6em}{}
\titleformat{\subsection}[hang]{\bfseries\fontsize{11}{15}\selectfont}{\thesubsection.}{0.6em}{}

% --- 그림 및 표 이름 변경 ---
\ifPNU@english
\else
  \renewcommand{\figurename}{그림}
  \renewcommand{\tablename}{표}
\fi

% --- 참고문헌: 서식처럼 번호가 매겨진 장(6. 참고 문헌)으로 표시 ---
\ifPNU@english
\else
  \renewcommand{\bibname}{참고 문헌}
\fi
\renewenvironment{thebibliography}[1]
  {\chapter{\bibname}%
   \list{\@biblabel{\@arabic\c@enumiv}}%
        {\settowidth\labelwidth{\@biblabel{#1}}%
         \leftmargin\labelwidth
         \advance\leftmargin\labelsep
         \@openbib@code
         \usecounter{enumiv}%
         \let\p@enumiv\@empty
         \renewcommand\theenumiv{\@arabic\c@enumiv}}%
   \sloppy
   \clubpenalty4000
   \@clubpenalty \clubpenalty
   \widowpenalty4000%
   \sfcode`\.\@m}
  {\def\@noitemerr
    {\@latex@warning{Empty `thebibliography' environment}}%
   \endlist}

% --- 색인 제목 변경 ---
\ifPNU@english
\else
  \renewcommand{\indexname}{찾아보기}
\fi

% --- 코드 리스팅 제목 변경 ---
\ifPNU@english
\else
  \renewcommand{\lstlistingname}{코드}
\fi

% --- 페이지 스타일: 서식처럼 머리말은 가는 구분선만, 꼬리말은 가운데 쪽번호만 ---
\fancypagestyle{pnu}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{1pt}
  \renewcommand{\footrulewidth}{0pt}
  \cfoot{\thepage}
}
\fancypagestyle{pnutitle}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{1pt}
  \renewcommand{\footrulewidth}{0pt}
}
\pagestyle{pnu}
\renewcommand{\chaptermark}[1]{} % chaptermark 잔여 출력 방지
% report 클래스는 각 장의 첫 페이지에 "plain" 스타일을 강제 적용하므로 동일하게 재정의한다.
\fancypagestyle{plain}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{1pt}
  \renewcommand{\footrulewidth}{0pt}
  \cfoot{\thepage}
}

% --- 기본값 설정 ---
\ifPNU@english
  \reporttitle{Title of the Report}
  \reportteam{My Team}
  \reportauthor{My Name}
  \reportadvisor{My Advisor}
  \reportdate{\today}
\else
  \reporttitle{보고서 제목}
  \reportteam{모둠 이름}
  \reportauthor{작성자 이름}
  \reportadvisor{지도교수 이름}
  \reportdate{\today}
\fi

\endinput
