1. extract_imbalanced_index.py을 통해서 dict 디렉토리 생성 및 특정 클래스 index를 담고 있는 pkl 파일을 생성 및 저장합니다..

2. AC_GAN 디렉토리에서 main.py를 실행시키면, dict 디렉토리에서 index를 담고 있는 pkl 파일을 불러와, 모델을 학습시킨 뒤

models 디렉토리를 생성하고, 최적 Generator, Discriminator pickle를 저장합니다.

3. AC_GAN 디렉토리에서 oversampling.py을 실행시키면 models 디렉토리에서 저장한 모델을 불러와서 지정한 개수만큼의 이미지를

임의로 생성합니다. 최종적으로 cnn_main에서 쓰일 두번째 dataset이 되는 image folder를 구성합니다.

4. cnn_main을 통해서 분류 및 결과를 저장합니다.