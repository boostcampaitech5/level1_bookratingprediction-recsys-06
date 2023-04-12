# Book Recommendation

사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측하는 프로젝트이다. 책 한권을 모두 읽기 위해서는 보다 긴 물리적인 시간을 필요로 한다. 또한 소비자 입장에서는 제목, 저자, 표지, 카테고리 등 한정된 정보로 각자가 콘텐츠를 유추하고 구매 유무를 결정해야 하기 때문에 상대적으로 선택에 더욱 신중을 가하게 된다.

이러한 소비자들의 책 구매 결정에 대한 도움을 주고자 한다. 사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측하는 모델을 구축 한다.

책에 대한 메타 데이터인 books, 고객에 대한 메타 데이터인 users, 고객이 책에 남긴 평점 ratings 의 데이터 셋을 활용해, 최종적으로 1과 10 사이 평점을 예측하는 것을 목적으로 한다.

(데이터셋 출처: https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)

# Project architecture

```
├─src
	├─data
	├─ensembles
	├─models
	├─utils.py
├─main.py
├─ensemble.py
├─requirements.txt
```

# Environment Requirements

```

```

# Reference

- [Factorization Machine](https://ieeexplore.ieee.org/document/5694074)
- [Field-aware Factorization Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [Neural Collaborotaive Filtering](https://arxiv.org/abs/1708.05031)
- [DeepCoNN](https://arxiv.org/abs/1701.04783)

# Contributors

