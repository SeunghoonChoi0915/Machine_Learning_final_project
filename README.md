anchor와 ResNet18을 이용한 할리갈리 카드 분류기

<img width="400" height="305" alt="스크린캐스트 06-15-2026 01_13_57 AM" src="https://github.com/user-attachments/assets/14612540-79fc-4028-b418-f84ed9fd8dd8" />

📌 프로젝트 개요

ResNet18 기반 CNN 모델과 anchor를 활용하여 할리갈리 카드를 실시간으로 분류하는 시스템입니다.

웹캠 영상에서 카드를 인식하고 과일 종류와 개수 위치를 자동으로 예측합니다.


🍓 클래스 구성 (20개)

4종류의 과일 × 5단계 개수 = 총 20클래스

과일1개2개3개4개5개딸기s1s2s3s4s5바나나b1b2b3b4b5라임l1l2l3l4l5자두p1p2p3p4p5


🏗️ 모델 구조


Backbone: ResNet18
Detection: Anchor 기반 객체 탐지
Classification: 20-class softmax 분류기


