# Book Recommendation
### 책에 대한 유저의 선호도(평점) 예측

| 항목 | 설명 |
| --- | --- |
| 활용 장비 및 재료 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| • 서버: Tesla V100, 88GB RAM Server<br>• 개발 IDE: Jupyter Notebook, VS Code<br>• 협업 Tool: Notion, Slack, Zoom |
| Metric | RMSE Score |
| Dataset | • books.csv : 149,570개의 책(item)에 대한 정보를 담고 있는 메타데이터<br>• users.csv : 68,092명의 고객(user)에 대한 정보를 담고 있는 메타데이터<br>• train_ratings.csv : 59,803명의 사용자(user)가 129,777개의 책(item)에 대해 남긴 306,795건의 평점(rating) 데이터 <br>https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset|


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
```
pip install -r requirements.txt
```

- train & Inference : `main.py`

```
python main.py --MODEL FM --DATA_PATH data
```


# Reference

- [Factorization Machine](https://ieeexplore.ieee.org/document/5694074)
- [Field-aware Factorization Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [Neural Collaborotaive Filtering](https://arxiv.org/abs/1708.05031)
- [DeepCoNN](https://arxiv.org/abs/1701.04783)

# Contributors
- 김지연
- 음이레
- 오승민
- 조재오
- 윤한나
