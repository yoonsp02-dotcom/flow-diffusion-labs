# Flow Matching & Diffusion Labs

수식에서 실제 이미지 생성까지 연결하는 MNIST 학습 프로젝트입니다. 일반적인 probability path에서 Gaussian path를 유도하고, target·loss·sampling 코드로 구현합니다.

## 실습
| Notebook | 내용 | 상태 |
| --- | --- | --- |
| 01 | Flow Matching과 Diffusion: 수식에서 이미지 생성까지 | 완료 |
| 02 | Conditional Generation과 Guidance | 예정 |
| 03 | U-Net·DiT architecture 비교 | 예정 |

[Colab에서 01 실행](https://colab.research.google.com/github/yoonsp02-dotcom/flow-diffusion-labs/blob/main/notebooks/01_flow_matching_and_diffusion.ipynb)

## 실행
Google Colab에서 GPU runtime을 선택하고 셀을 순서대로 실행합니다. PyTorch, torchvision, NumPy, Matplotlib을 사용하며 MNIST는 자동 다운로드됩니다.
기본 설정은 두 MLP를 300 epoch 학습합니다. Google Drive 연결 시 checkpoint를 자신의 Drive에 저장합니다.
새 실험은 RUN_DIR을 변경하고, 중단된 실험은 같은 경로에서 RESUME=True로 재개합니다.
MAX_EPOCHS는 도달할 epoch 번호이며 SCHEDULE_EPOCHS는 resume 시 유지합니다.

## Baseline
- MNIST train/validation/test: 55,000 / 5,000 / 10,000
- Unconditional MLP, linear Gaussian path, 직접 velocity/noise 출력
- 같은 initial weights, batch, time, noise, optimizer 설정
- 동일 Heun ODE 비교와 noise 모델의 ODE/SDE 비교

두 모델 모두 loss가 감소했지만 일부 생성 이미지의 형태와 획이 불완전했습니다. 작은 time 구간에서는 Noise 모델을 velocity로 변환한 오차가 크게 나타났습니다. 자체 loss는 target이 다르므로 직접 비교하지 않습니다.
PCA projection과 시각적 비교는 전체 distribution의 일치나 일반적인 방법론의 우열을 입증하지 않습니다. 이후 실습은 이 결과를 baseline으로 사용합니다.

## 재현 범위
Notebook에 저장된 결과는 기존 Colab 실행에서 얻었습니다. 정확한 당시 package 버전과 GPU 모델은 기록되지 않아 bitwise 재현을 보장하지 않습니다. 저장된 그래프와 이미지가 포함되어 있습니다.
Checkpoint와 dataset은 repository에 포함하지 않습니다.

## 참고
Peter Holderrieth and Ezra Erives, *An Introduction to Flow Matching and Diffusion Models*, MIT 6.S184.
- https://diffusion.csail.mit.edu/
- 현재 notebook: Ch.3–4 기반 학습·구현 실습
- 예정: Ch.5 guidance, Ch.6 architecture

이 프로젝트는 교재 기반의 개인 학습·실험 프로젝트이며 새로운 알고리즘을 제안하지 않습니다.
