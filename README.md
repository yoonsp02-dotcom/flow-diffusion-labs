# Flow Matching & Diffusion Labs

수식에서 실제 이미지 생성까지 연결하는 MNIST 학습 프로젝트입니다. 일반적인 probability path에서 Gaussian path를 유도하고, target·loss·sampling 코드로 구현합니다.

## 실습

| Notebook | 내용 | 상태 |
| --- | --- | --- |
| 01 | Flow Matching과 Diffusion: 수식에서 이미지 생성까지 | 완료 |
| 02 | Conditional Generation과 Classifier / Classifier-Free Guidance | 완료 |
| 03 | U-Net·DiT를 이용한 Flow / Noise 및 ODE / SDE 비교 | 완료 |

- [01 — Colab에서 실행](https://colab.research.google.com/github/yoonsp02-dotcom/flow-diffusion-labs/blob/main/notebooks/01_flow_matching_and_diffusion.ipynb)
- [02 — Colab에서 실행](https://colab.research.google.com/github/yoonsp02-dotcom/flow-diffusion-labs/blob/main/notebooks/02_conditional_generation_and_guidance.ipynb)
- [03 — Colab에서 실행](https://colab.research.google.com/github/yoonsp02-dotcom/flow-diffusion-labs/blob/main/notebooks/03_unet_dit_flow_noise_ode_sde.ipynb)

## 실행

Google Colab에서 GPU runtime을 선택하고 각 notebook의 셀을 순서대로 실행합니다.
PyTorch, torchvision, NumPy, Matplotlib 등을 사용하며 MNIST는 자동 다운로드됩니다.
일부 평가에서는 별도의 사전학습 classifier를 사용합니다.

- **01:** MLP 기반 Flow / Noise 모델을 학습하고 ODE / SDE sampling을 비교합니다.
- **02:** 01의 checkpoint를 baseline과 classifier guidance에 사용하고,
  conditional 생성 모델과 time-dependent classifier를 별도로 학습합니다.
- **03:** 작은 U-Net과 DiT를 각각 Flow / Noise target으로 학습합니다.
  기본 학습 설정은 모델당 30 epoch입니다.

Google Drive 연결 시 checkpoint와 실험 결과를 저장할 수 있습니다.
실행 전 각 notebook의 저장 경로와 checkpoint 설정을 확인하세요.

## Baseline
- MNIST train/validation/test: 55,000 / 5,000 / 10,000
- Unconditional MLP, linear Gaussian path, 직접 velocity/noise 출력
- 같은 initial weights, batch, time, noise, optimizer 설정
- 동일 Heun ODE 비교와 noise 모델의 ODE/SDE 비교

두 모델 모두 loss가 감소했지만 일부 생성 이미지의 형태와 획이 불완전했습니다. 작은 time 구간에서는 Noise 모델을 velocity로 변환한 오차가 크게 나타났습니다. 자체 loss는 target이 다르므로 직접 비교하지 않습니다.
PCA projection과 시각적 비교는 전체 distribution의 일치나 일반적인 방법론의 우열을 입증하지 않습니다. 이후 실습은 이 결과를 baseline으로 사용합니다.

## Conditional Generation과 Guidance

02에서는 MNIST label을 조건으로 사용해 vanilla conditional generation,
classifier guidance(CG), classifier-free guidance(CFG)를 비교합니다.

이번 실험에서 Flow CFG는 시각적으로 Flow CG보다 더 깨끗한 결과를 보였고,
평가한 설정에서 높은 target-label accuracy를 얻었습니다.
다만 CG와 CFG는 서로 다른 모델 및 학습 구성을 사용하며,
label accuracy만으로 이미지의 자연스러움이나 다양성을 판단할 수는 없습니다.

## U-Net과 DiT 실험

03에서는 MNIST pixel 공간에서 작은 U-Net과 DiT를 직접 구현하고,
각 architecture에서 Flow와 Noise prediction을 학습했습니다.
동일한 모델에 ODE와 SDE sampling을 적용해 결과를 비교합니다.

이번 설정에서는 약한 guidance에서 Flow가 Noise보다 높은
target-label accuracy를 보였습니다. Noise 출력의 초반 velocity 오차와
모델 교체 실험도 살펴봤지만, 성능 차이의 원인을 초반 오차 하나로
확정할 수는 없습니다.

이 결과는 특정 데이터, 모델 크기와 학습 설정에서 얻은 관찰이며,
일반적인 Flow와 Noise prediction의 우열을 뜻하지 않습니다.

## 재현 범위
Notebook에 저장된 결과는 기존 Colab 실행에서 얻었습니다. 정확한 당시 package 버전과 GPU 모델은 기록되지 않아 bitwise 재현을 보장하지 않습니다. 저장된 그래프와 이미지가 포함되어 있습니다.
Checkpoint와 dataset은 repository에 포함하지 않습니다.

## Reference
Peter Holderrieth and Ezra Erives, *An Introduction to Flow Matching and Diffusion Models*, MIT 6.S184.
- https://diffusion.csail.mit.edu/
- 01: Ch.3–4 기반 학습·구현 실습
- 02: Ch.5 기반 conditional generation과 guidance 실습
- 03: Ch.6의 architecture 관련 내용을 바탕으로 한 U-Net·DiT 실험
이 프로젝트는 교재 기반의 개인 학습·실험 프로젝트이며 새로운 알고리즘을 제안하지 않습니다.
