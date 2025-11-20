# MLnDL

머신러닝 및 딥러닝 예제 모음입니다.

## 개요

scikit-learn, TensorFlow 등을 사용한 다양한 머신러닝 및 딥러닝 예제 코드입니다.

## 주요 기능

### 분류 (Classification)
- **03tree.py**: 의사결정 트리 (Iris 데이터셋)
- **03cancer_classify.py**: 암 분류 예제
- **svm-sklearn.py**: 서포트 벡터 머신 (SVM)
- **tree_sgip.py**: 의사결정 트리 (SGIP 데이터셋)

### 클러스터링 (Clustering)
- **kmeans-sklearn.py**: K-Means 클러스터링 (Iris 데이터셋)
- **04kmeans-cluster.py**: K-Means 클러스터링 기본 예제
- **04kmeans-run.py**: K-Means 실행 스크립트

### 차원 축소
- **PCA_PCoA_3d.py**: PCA/PCoA 3차원 시각화
- **PCA_PCoA(MDS).py**: PCA/PCoA (MDS) 분석

### 회귀
- **dt_ameshousing.py**: 의사결정 트리 회귀 (Ames Housing 데이터셋)
- **nsl.py**: 회귀 분석 예제

### 딥러닝
- **tensorflow_test.py**: TensorFlow 기본 테스트

## 사용 방법

각 스크립트를 개별적으로 실행할 수 있습니다:

```bash
# 의사결정 트리
python 03tree.py

# K-Means 클러스터링
python kmeans-sklearn.py

# SVM
python svm-sklearn.py
```

## 요구사항

- Python 3.12
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- graphviz
- dtreeviz
- tensorflow (tensorflow_test.py)

## 설치

### uv 설치

#### Windows
```powershell
# PowerShell에서 실행
irm https://astral.sh/uv/install.ps1 | iex
```

#### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 후 터미널을 재시작하거나 다음 명령어로 PATH에 추가:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

### 가상환경 설정

```bash
# Python 3.12 가상환경 생성
uv venv --python 3.12

# 가상환경 활성화
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 패키지 설치

```bash
# uv를 사용한 패키지 설치
uv pip install -r requirements.txt

# graphviz 설치 (시각화용)
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz
```

## 파일 구조

### 분류 예제
- `03tree.py`: Iris 데이터셋 의사결정 트리
- `03cancer_classify.py`: 암 분류
- `svm-sklearn.py`: SVM 분류
- `tree_sgip.py`: SGIP 데이터셋 의사결정 트리

### 클러스터링 예제
- `kmeans-sklearn.py`: K-Means (Iris)
- `04kmeans-cluster.py`: K-Means 기본
- `04kmeans-run.py`: K-Means 실행

### 차원 축소
- `PCA_PCoA_3d.py`: 3D PCA/PCoA
- `PCA_PCoA(MDS).py`: PCA/PCoA MDS

### 회귀
- `dt_ameshousing.py`: Ames Housing 회귀
- `nsl.py`: 회귀 분석

### 딥러닝
- `tensorflow_test.py`: TensorFlow 테스트

## 데이터셋

일부 스크립트는 다음 데이터셋을 사용합니다:
- Iris 데이터셋 (scikit-learn 내장)
- Ames Housing 데이터셋
- Wine 데이터셋
- SGIP 데이터셋

## 시각화

의사결정 트리 시각화를 위해 graphviz와 dtreeviz를 사용합니다. PNG 및 SVG 형식으로 저장됩니다.

---

해당 프로젝트는 Examples-Python의 Private Repository에서 공개 가능한 수준의 소스를 Public Repository로 변환한 것입니다.

