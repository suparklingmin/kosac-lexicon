# KOSAC Lexicon 사용 매뉴얼

`kosac-lexicon`은 KOSAC(한국어 감정 분석 코퍼스)에서 도출한 **형태소 단위 한국어
감정 사전**을 파이썬 패키지로 제공합니다. 배포 이름은 `kosac-lexicon`, import
이름은 `kosac`입니다.

> 빠른 시작만 원하면 [§4 빠른 시작](#4-빠른-시작)으로 가세요. 전체 개념은 [§3](#3-핵심-개념),
> 권장 사용법은 [§6 통합 분석기](#6-통합-분석기-고수준-api)입니다.

## 목차
1. [소개](#1-소개)
2. [설치](#2-설치)
3. [핵심 개념](#3-핵심-개념)
4. [빠른 시작](#4-빠른-시작)
5. [사전 다루기(저수준 API)](#5-사전-다루기-저수준-api)
6. [통합 분석기(고수준 API)](#6-통합-분석기-고수준-api)
7. [토크나이저](#7-토크나이저)
8. [명령줄 인터페이스(CLI)](#8-명령줄-인터페이스-cli)
9. [scikit-learn 연동](#9-scikit-learn-연동)
10. [유틸리티](#10-유틸리티)
11. [자주 묻는 질문·주의사항](#11-자주-묻는-질문주의사항)
12. [인용·라이선스](#12-인용라이선스)

---

## 1. 소개

이 사전은 어휘의 감정 특성이 그것을 포함하는 핵심 주관 표현(Seed)에서 도출된다는
가정 아래, Seed에서 추출한 형태소 N-그램을 표제어로 삼아 구축되었습니다. 데이터는
2016년에 공개되었으며 그 내용은 고정되어 있습니다(데이터 버전 `2016`).

하나의 텍스트를 분석하는 흐름은 다음과 같습니다.

```
한국어 문장 → (형태소 분석) surface/POS 토큰 → (사전 매칭) 감정 표현 추출 → 라벨 확률 분포
```

## 2. 설치

```bash
pip install kosac-lexicon                  # 핵심: 사전 데이터 + 조회 API (pandas/numpy만 필요)
pip install "kosac-lexicon[kiwi]"          # + Kiwi 형태소 분석기 (순수 pip, 자바 불필요)
pip install "kosac-lexicon[transformers]"  # + HuggingFace 서브워드 토크나이저
pip install "kosac-lexicon[sklearn]"       # + scikit-learn 피처 추출기
pip install "kosac-lexicon[all]"           # 전체
```

- **자바(JVM)는 필요 없습니다.** 형태소 분석은 순수 pip로 설치되는 Kiwi(`kiwipiepy`)를
  사용합니다.
- 핵심 설치만으로 사전 데이터 로드·조회·필터링이 모두 됩니다. 문장 토큰화가 필요할
  때만 `[kiwi]`를 추가하세요.

## 3. 핵심 개념

### 3.1 표제어 형식

표제어는 `형태소/품사` 토큰을 공백으로 이은 N-그램입니다(세종 태그셋). 예:

- 유니그램: `좋/VA`, `힘/NNG`
- 바이그램: `행복/NNG 하/XSA`

각 사전은 16,361개 표제어(유니그램 3,476 · 바이그램 6,579 · 트라이그램 6,307)를
가집니다.

### 3.2 여섯 가지 의미 특성

`kosac.FEATURES`로 확인할 수 있습니다.

| 특성(feature) | 라벨 값 |
| --- | --- |
| `polarity` | COMP, NEG, NEUT, None, POS |
| `intensity` | High, Low, Medium, None |
| `expressive-type` | dir-action, dir-explicit, dir-speech, indirect, writing-device |
| `nested-order` | 0, 1, 2, 3 |
| `subjectivity-polarity` | COMP, NEG, NEUT, POS |
| `subjectivity-type` | Agreement, Argument, Emotion, Intention, Judgment, Others, Speculation |

각 라벨 값의 의미 요약은 `kosac.describe_feature("polarity")`로 볼 수 있습니다([§10](#10-유틸리티)).

### 3.3 사전 데이터 구조

사전은 `entry`(공백으로 이은 형태소)를 인덱스로 하는 pandas DataFrame이며 열은
다음과 같습니다.

- `freq`: 해당 N-그램을 포함하는 Seed의 개수
- (라벨별 열): 로드 시 상대빈도 → 절대빈도(=비율×freq)로 변환됨
- `max.value`: 가장 비율이 높은 라벨
- `max.prop`: 그 비율 값

## 4. 빠른 시작

```python
import kosac

# (1) 사전만 조회 — 토크나이저/자바 불필요
lex = kosac.load_lexicon("polarity", ngrams=[1], min_freq=5)
lex.get_entry("힘/NNG")

# (2) 문장 분석 — Kiwi 필요: pip install "kosac-lexicon[kiwi]"
from kosac import SentimentAnalyzer

analyzer = SentimentAnalyzer("polarity")
result = analyzer.analyze("이 영화 정말 좋다")
result["features"]["polarity"]["label"]   # 'POS'
```

## 5. 사전 다루기 (저수준 API)

### 5.1 로드

```python
import kosac
from kosac.lexicon import PolarityLexicon

lex = kosac.load_lexicon("polarity", ngrams=[1, 2, 3], min_freq=0, threshold=0.0)
lex = PolarityLexicon.load(ngrams=[1])                 # 클래스 메서드도 동일
lex = PolarityLexicon(filepath="my-lexicon.csv")       # 직접 만든 CSV 사용
```

- `ngrams`: 사용할 N-그램 길이 목록(기본 `[1]`).
- `load_lexicon`의 `feature`는 `kosac.FEATURES` 중 하나여야 합니다.

### 5.2 필터링·조회

```python
lex.set_lexicon(min_freq=5, threshold=0.6)  # freq>=5 그리고 max.prop>0.6 만 남김
lex.set_lexicon(min_freq=1)                 # 다시 느슨하게 — 원본 기준으로 재필터됨
lex.get_size()              # 표제어 개수
lex.get_labels()            # ['COMP','NEG','NEUT','None','POS']
lex.get_lexicon()           # 현재(필터된) DataFrame
lex.get_original_lexicon()  # 필터 이전 원본
lex.get_entry("좋/VA")      # 한 표제어의 행 (레거시 별칭: lex.get("좋/VA"))
```

`set_lexicon`은 매번 **원본 사전**을 기준으로 필터링하므로 임계값을 더 좁히거나
**다시 넓힐 수 있습니다**.

### 5.3 매칭·점수 (토크나이저 필요)

```python
from kosac.tokenizers import KiwiTokenizer
tok = KiwiTokenizer()

lex.match_patterns("나는 정말 행복하다", tok)   # 매칭된 표제어 문자열 목록 (별칭: lex.match)
lex.get_match_info("나는 정말 행복하다", tok)   # (표제어, max.value, max.prop) 목록
lex.get_sent_probs("나는 정말 행복하다", tok)   # 라벨별 확률(softmax) Series
```

> `sorting=True`(기본)는 긴 N-그램·높은 확률 표제어부터, `sorting=False`는 원래
> 순서대로 매칭합니다.

> 더 풍부한 결과(문자 스팬, 부정 처리 등)는 [§6 통합 분석기](#6-통합-분석기-고수준-api)를
> 사용하세요. `match_patterns`/`get_sent_probs`는 단순 사전 룩업입니다.

### 5.4 커스텀 사전 만들기·확장

빈 사전에서 시작하거나 코퍼스로부터 빈도를 학습할 수 있습니다.

```python
from kosac.lexicon import GenericLexicon
from kosac.corpora import Corpus

lex = GenericLexicon(ngrams=[1, 2])
lex.set_labels(["POS", "NEG"])           # GenericLexicon 전용
lex.add_token("좋/VA", "POS")            # 표제어에 라벨 카운트 1 추가
lex.update([("싫/VA", "NEG"), ("좋/VA", "POS")])

corpus = Corpus("data/example.csv")      # 헤더 없는 text,label CSV
lex.update_from_corpus(corpus, KiwiTokenizer())

lex.export_user_dict("user_dictionary.txt")  # 유니그램을 form\tPOS 로 내보내기
```

## 6. 통합 분석기 (고수준 API)

`SentimentAnalyzer`는 사전과 토크나이저를 묶어 한 번의 호출로 텍스트를 분석합니다.
**권장 진입점**입니다.

```python
from kosac import SentimentAnalyzer

analyzer = SentimentAnalyzer(
    features="polarity",     # 'all' 또는 ['polarity','intensity', ...] 도 가능
    tokenizer=None,          # 기본 KiwiTokenizer()
    ngrams=(1, 2, 3),
    min_freq=0, threshold=0.0,
    smoothing=True,
    align=False,             # Kiwi 사용자사전을 사전으로 시드 (§7.3)
    negation=False,          # 부정 처리 (§6.3)
    intensifier=False,       # 강조 처리 (§6.3)
    window=2, intensifier_factor=2.0,
)
```

### 6.1 `analyze(text)` 반환 구조

```python
r = analyzer.analyze("이 영화 정말 좋다")
```

```python
{
  "text": "이 영화 정말 좋다",
  "tokens": ["이/MM", "영화/NNG", "정말/MAG", "좋/VA", "다/EF"],
  "features": {
    "polarity": {
      "label": "POS",                # 최상위 라벨
      "prob": 0.97,                  # 그 라벨의 확률
      "probs": {"COMP": ..., "NEG": ..., "POS": 0.97, ...},
      "matches": [
        {"entry": "좋/VA", "span": [8, 9], "text": "좋",
         "max_value": "POS", "max_prop": 0.92,
         "negated": False, "weight": 1.0},
        ...
      ]
    }
  }
}
```

- `span`은 **원문 문자 오프셋** `[시작, 끝)`이라 `text[span[0]:span[1]]`로 다시 잘라낼 수 있습니다.
- 결과는 JSON 직렬화 가능한 순수 dict입니다.

### 6.2 여러 특성 동시 분석 / 배치

```python
SentimentAnalyzer("all").analyze("이 영화는 정말 좋았고 너무 행복했다")
# polarity=POS, intensity=Medium, expressive-type=dir-speech,
# nested-order=1, subjectivity-polarity=POS, subjectivity-type=Argument

analyzer.analyze_batch(["좋다", "싫다"])     # 결과 dict 목록
analyzer.analyze_frame(["좋다", "싫다"])     # tidy pandas DataFrame
#    text  polarity.label  polarity.prob
# 0   좋다             POS           ...
```

### 6.3 부정·강조 처리

옵트인 휴리스틱입니다. 매칭된 표현 주변 `window` 토큰 안에 마커가 있으면 적용합니다.

```python
a = SentimentAnalyzer("polarity", negation=True, intensifier=True)
a.analyze("이 영화는 안 좋다")["features"]["polarity"]["label"]   # 'NEG'
```

- **부정**(`안/못/않/없 …`): 매칭 표현의 POS↔NEG 질량을 교환. POS·NEG 라벨을 모두
  가진 특성(`polarity`, `subjectivity-polarity`)에만 적용됩니다.
- **강조**(`정말/너무/매우 …`): 매칭 표현의 가중치를 `intensifier_factor`배.
- 마커 집합은 `kosac.analyzer.DEFAULT_NEGATIONS` / `DEFAULT_INTENSIFIERS`이며,
  생성자 인자 `negations=`, `intensifiers=`로 교체할 수 있습니다.

```python
from kosac.analyzer import DEFAULT_NEGATIONS
SentimentAnalyzer("polarity", negation=True,
                  negations=DEFAULT_NEGATIONS | {"별로/MAG"}, window=3)
```

> ⚠️ 윈도 기반 휴리스틱이라 인접한 다른 형태소까지 부정으로 표시될 수 있습니다.
> 최종 라벨에는 보통 영향이 없지만 정밀한 부정 범위 판정이 필요하면 `window`를
> 조절하거나 직접 후처리하세요.

### 6.4 빈도 계산 방식 (사회과학용)

많은 사회과학 연구는 확률 대신 **감정 단어의 빈도**(긍정/부정 단어 수와 점유율)로
텍스트를 분류합니다. `count()`가 이 방식을 제공합니다 — 매칭된 형태소를 각자의
대표 라벨(`max.value`)로 집계합니다.

```python
a = SentimentAnalyzer("polarity")
c = a.count("빗물이 흐르고 내 눈물도 흐르고 잃어버린 첫사랑도 흐르네")["features"]["polarity"]
c["counts"]       # {'NEG': 6, 'POS': 3, ...} — 라벨별 단어 수
c["proportions"]  # 라벨별 점유율 (합 1)
c["total"]        # 매칭된 단어 수
c["label"]        # 가장 빈도 높은 라벨 → 'NEG'

a.count_batch(texts)   # 결과 dict 목록
a.count_frame(texts)   # 문서별 라벨 카운트 DataFrame (<특성>.POS, <특성>.NEG, ...)
```

`analyze()`(확률 방식)와 `count()`(빈도 방식)는 동일한 매칭 결과를 다르게 집계합니다.
부정·강조 옵션은 빈도 방식에는 적용되지 않습니다(단어 수 집계이므로).

## 7. 토크나이저

`kosac.tokenizers` 모듈.

### 7.1 종류

| 클래스 | 설명 | 필요 익스트라 |
| --- | --- | --- |
| `Tokenizer` | 공백 분리(기본). 미리 태깅된 입력 테스트용 | 없음 |
| `KiwiTokenizer` | Kiwi 형태소 분석(권장). `surface/POS` 반환 | `[kiwi]` |
| `HuggingFaceTokenizer` | HuggingFace 서브워드 | `[transformers]` |

### 7.2 공통 메서드

```python
tok.tokenize("나는 좋다")               # ['나/NP', '는/JX', '좋/VA', '다/EF']
tok.get_tokens_str("나는 좋다")         # '나/NP 는/JX 좋/VA 다/EF'
tok.tokenize_with_offsets("나는 좋다")  # [('나/NP', 0, 1), ('는/JX', 1, 2), ...]
tok.get_ngrams("나는 좋다", ns=[1, 2]) # 유니그램+바이그램 문자열 목록
```

### 7.3 Kiwi 사용자사전 정렬

사전 표제어를 Kiwi 사용자사전에 등록하면 분절이 사전과 더 잘 맞습니다(태그셋
불일치 완화).

```python
from kosac.tokenizers import KiwiTokenizer

tok = KiwiTokenizer(user_words=[("행복하", "VA")])   # 직접 등록
tok.add_user_words(["좋/VA", ("멋지", "VA")])         # 'form/tag' 또는 (form, tag)

lex = kosac.load_lexicon("polarity", ngrams=[1])
tok = KiwiTokenizer.from_lexicon(lex, tags={"NNG", "VV", "VA", "XR"})  # 사전 유니그램으로 시드
```

`SentimentAnalyzer(..., align=True)`는 위 `from_lexicon`을 자동으로 적용합니다.

## 8. 명령줄 인터페이스 (CLI)

설치하면 `kosac` 명령이 생깁니다(또는 `python -m kosac`).

```bash
kosac analyze "이 영화 정말 좋다"                 # 보기 좋은 JSON (확률 방식)
kosac analyze "이 영화는 안 좋다" --negation       # 부정 처리
kosac analyze "좋다" --features all                # 6개 특성 모두
kosac analyze "빗물이 흐르고 눈물도 흐르고" --count  # 빈도 계산 방식
kosac features                                     # 특성 목록
kosac citation                                     # BibTeX

printf '좋다\n싫다\n' | kosac analyze --compact     # 한 줄에 JSON 하나(JSONL)
```

`analyze` 옵션: `--features`(쉼표 구분 또는 `all`), `--ngrams`, `--min-freq`,
`--negation`, `--intensifier`, `--align`, `--count`(빈도 방식), `--compact`.
텍스트 인자를 생략하면 표준입력에서 한 줄당 한 문장을 읽습니다.

## 9. scikit-learn 연동

`KosacVectorizer`는 텍스트를 `<특성>=<라벨>` 확률 피처로 변환합니다(`[sklearn]` 필요).

```python
from kosac.sklearn import KosacVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

clf = make_pipeline(KosacVectorizer("all", negation=True), LogisticRegression())
clf.fit(train_texts, labels)
clf.predict(test_texts)

vec = KosacVectorizer("polarity").fit(train_texts)
vec.get_feature_names_out()   # ['polarity=COMP', 'polarity=NEG', ...]
```

생성자 인자는 `SentimentAnalyzer`와 동일(`features`, `ngrams`, `min_freq`,
`negation`, `intensifier`, `align`, `tokenizer`)합니다.

## 10. 유틸리티

```python
import kosac

kosac.FEATURES            # 6개 특성 이름 튜플
kosac.__version__         # 패키지 버전 (예: '1.0.0')
kosac.__data_version__    # 데이터 빈티지 ('2016')

kosac.citation()                    # BibTeX 문자열
kosac.describe_feature("polarity")  # {'feature','values','reference'}
kosac.FEATURE_VALUES["intensity"]   # 라벨 값 → 설명
```

## 11. 자주 묻는 질문·주의사항

- **자바가 필요한가요?** 아니요. Kiwi는 순수 pip 설치입니다.
- **태그셋 차이.** 표제어는 세종 태그셋 기반입니다. Kiwi 태그셋도 세종 기반이라 주요
  형태소 태그(NNG/VV/VA/JKS/EC …)는 일치하지만, 일부 기호·웹 태그나 분절은 원본 KOSAC
  구축에 쓰인 분석기와 다를 수 있습니다. `align=True`로 완화할 수 있습니다.
- **부정 처리는 완벽하지 않습니다.** [§6.3](#63-부정강조-처리)의 휴리스틱 한계를 참고하세요.
- **와일드카드 표제어.** 일부 표제어에는 정규식 특수문자 `*`가 포함됩니다(예: `가*/JKS`).
  내부적으로 이스케이프 처리되어 안전하게 리터럴 매칭됩니다.
- **빈 매칭.** 매칭이 하나도 없으면 `analyze`의 해당 특성 `label`은 `None`,
  `probs`는 빈 dict입니다.

## 12. 인용·라이선스

```python
print(kosac.citation())
```

- **코드**: MIT (`LICENSE`)
- **사전 데이터** (`kosac/data/*.csv`): CC BY-SA 4.0, KOSAC(서울대학교)에서 도출
  (`src/kosac/data/LICENSE`). 재배포 시 출처 표시와 동일조건 변경허락을 지켜 주세요.
