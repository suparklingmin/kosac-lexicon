# kosac-lexicon — 한국어 감정 어휘 목록

KOSAC(한국어 감정 분석 코퍼스)에서 도출한 **형태소 단위 한국어 감정 사전**을
파이썬 패키지로 배포한 것입니다. 2016년에 CSV로 공개한 어휘 목록을 그대로
패키지에 담았습니다.

> ⚠️ **사전 공개(베타, 0.4.1).** 동료·베타 유저 검토 중이며 1.0 정식 공개 전까지
> API가 바뀔 수 있습니다.

> 📖 전체 사용 매뉴얼: [`docs/manual.ko.md`](docs/manual.ko.md)

## 설치

```bash
pip install kosac-lexicon                  # 핵심: 사전 데이터 + 조회 API (pandas/numpy만 필요)
pip install "kosac-lexicon[kiwi]"          # + Kiwi 형태소 분석기 (순수 pip 설치, 자바 불필요)
pip install "kosac-lexicon[transformers]"  # + HuggingFace 서브워드 토크나이저
pip install "kosac-lexicon[all]"           # 전체
```

> **베타:** 아직 정식 PyPI에는 없고 [TestPyPI](https://test.pypi.org/project/kosac-lexicon/)에
> 올라갑니다. 1.0 정식 공개 전까지는 TestPyPI에서 설치하세요
> (`--extra-index-url`이 `pandas` / `numpy` / `kiwipiepy` 등 의존성을 정식 PyPI에서 가져옵니다):

```bash
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ "kosac-lexicon[kiwi]"
```

## 사용 예시

```python
import kosac

kosac.FEATURES                                  # 여섯 가지 의미 특성
lex = kosac.load_lexicon('polarity', ngrams=[1], min_freq=5)
lex.get_entry('힘/NNG')                         # 형태소 하나의 감정 분포
```

문장을 직접 태깅하려면 토크나이저(선택 의존성)가 필요합니다. 자바가 필요 없는
Kiwi를 권장합니다.

```python
from kosac.tokenizers import KiwiTokenizer   # pip install "kosac-lexicon[kiwi]"

tok = KiwiTokenizer()
lex = kosac.load_lexicon('polarity', ngrams=[1, 2, 3])
lex.match_patterns('나는 정말 행복하다', tok)
lex.get_sent_probs('나는 정말 행복하다', tok)
```

사전 표제어는 세종 태그셋 기반의 `surface/POS` 형식입니다. Kiwi의 태그셋도 세종
기반이라 주요 형태소 태그가 일치하지만, 일부 기호/웹 태그는 원본 KOSAC 구축에
쓰인 분석기와 다르게 분절될 수 있습니다.

## 예제

실행 가능한 스크립트는 [`examples/`](examples/)에 있습니다.

- [`quickstart.py`](examples/quickstart.py) — 현재 API로 조회·분석 한 번에. 여기서 시작하세요.
- [`nsmc_lexicon.py`](examples/nsmc_lexicon.py) — NSMC 영화 리뷰 코퍼스로 **새** POS/NEG 사전 만들기.
- [`nikl_lexicon.py`](examples/nikl_lexicon.py) — 모두의 말뭉치(NIKL) 감성 점수 표현으로
  **3-클래스**(POS/NEUT/NEG) 극성 사전 만들기.

사례 연구 튜토리얼: [NSMC](docs/tutorials/nsmc-case-study.md) ·
[NIKL](docs/tutorials/nikl-lexicon.md).

## 어휘 목록 구축 방식

이 어휘 목록은 KOSAC 데이터를 한국어 감정 분석 연구에 폭넓게 활용할 수 있도록
형태소 단위의 감정 특성을 제공하는 것을 목적으로 한다. 여기서는 어휘의 감정
특성이 그 어휘를 포함하는 핵심 주관 표현(이하 Seed)의 감정 특성에서 도출될 수
있다고 가정하고, Seed에서 추출한 형태소 N-그램을 어휘 표제어로 삼았다. 단, 한
Seed가 다른 Seed를 포함하는 경우 상위 Seed가 하위 Seed를 인용하거나 부정하거나
강조하는 등의 방식으로 감정 특성값을 전환할 수 있으므로, 일관된 감정 특성값을
얻기 위해 다른 Seed와 중첩되지 않는 최하위 Seed에 포함된 형태소만을 사용하였다.
형태소 N-그램은 가능한 모든 것을 뽑되 한글 이외의 문자나 문장 부호가 포함된
것은 제외하였다. 여섯 가지 의미 특성의 종류 및 값의 설명은 KOSAC V 1.0 README
파일에서 볼 수 있다.

여섯 개의 CSV 파일은 각각 의미 특성 하나에 해당하며, 위 방식으로 얻은 형태소
N-그램 표제어 16,362개(유니그램 3,476개, 바이그램 6,579개, 트라이그램 6,307개)가
가지는 의미 특성값들의 분포로 구성되어 있다. 파일 내에서 각 행은 하나의 N-그램의
감정 특성을 가리킨다. 열의 의미는 순서대로 다음과 같다.

- `ngram`: 표제어 N-그램을 이루는 형태소
- (의미 특성값): 각 라벨 값을 가지는 Seed의 **개수**(절대빈도)

`freq`(N-그램을 포함하는 Seed 수 = 카운트의 합), `max.value`(최빈 라벨), `max.prop`(그 비율)은
파일에 저장하지 않고 로더가 카운트에서 계산합니다. (2016년 최초 공개본은 상대빈도와
`freq`·`max.value`·`max.prop`을 저장했으며, 절대빈도 = round(비율 × freq).)

## 라이선스

- **코드** — MIT ([`LICENSE`](LICENSE))
- **사전 데이터** (`kosac/data/*.csv`) — CC BY-SA 4.0
  ([`src/kosac/data/LICENSE`](src/kosac/data/LICENSE)). KOSAC(서울대학교)에서 도출.
