# Book Recommendation -- 책에 대한 유저의 선호도(평점) 예측
### Contributors
| [<img src="https://github.com/ji-yunkim.png" width="100px">](https://github.com/ji-yunkim) | [<img src="https://github.com/YirehEum.png" width="100px">](https://github.com/YirehEum) | [<img src="https://github.com/osmin625.png" width="100px">](https://github.com/osmin625) | [<img src="https://github.com/Grievle.png" width="100px">](https://github.com/Grievle) | [<img src="https://github.com/HannahYun.png" width="100px">](https://github.com/HannahYun) |
| :--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------:
|                          [김지연](https://github.com/ji-yunkim)                           |                            [음이레](https://github.com/YirehEum)                             |                        [오승민](https://github.com/osmin625)                           |                          [조재오](https://github.com/Grievle)                           |                            [윤한나](https://github.com/HannahYun)  

## 활용 장비 및 재료(개발 환경, 협업 tool 등)
| 항목 | 설명 |
| --- | --- |
| 환경 | • 로컬 환경: `Windows`, `Mac`<br> • 서버: `Linux (Tesla V100)`, `88GB RAM Server`<br>• 협업 Tool: `Slack`, `Notion`, `Github`<br>• 사용 버전: `Python == 3.8.5`, `Pandas == 2.0.0`, `Torch == 1.7.1`|
| Metric | RMSE Score |
| Dataset | • books.csv : 149,570개의 책(item)에 대한 정보를 담고 있는 메타데이터<br>• users.csv : 68,092명의 고객(user)에 대한 정보를 담고 있는 메타데이터<br>• train_ratings.csv : 59,803명의 사용자(user)가 129,777개의 책(item)에 대해 남긴 306,795건의 평점(rating) 데이터 <br>https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset|

## Project architecture

```
code
├── main.py
├── model_omin
│   ├── EDA.py
│   ├── /* baseline_study.py -- removed*/
│   ├── context_data_modified.py
│   ├── distribution.ipynb
│   ├── lgbm.ipynb
│   ├── minmax_scaler.ipynb
│   └── rf.ipynb
├── src
│   / * baseline code -- removed */
└── ensemble.py
```

## Environment Requirements
```
pip install -r requirement.txt
```

- train & Inference : `main.py`

```
python main.py --MODEL FM
```
## 프로젝트 수행 절차 및 방법
![image](https://user-images.githubusercontent.com/46878927/234211322-57c3b810-0f95-46e9-9dbf-d20932d43d6a.png)

## 프로젝트 수행 결과
![image](https://user-images.githubusercontent.com/46878927/234227754-7f10d5eb-4da8-404b-af22-479cf0069932.png)
![image](https://user-images.githubusercontent.com/46878927/234215909-4f16ea55-1ff7-4244-8ccf-b10a967c59ee.png)

### 최종 순위
- Public 7위 (RMSE: 2.1207) / Private 7위 (RMSE: 2.1159)
![image](https://user-images.githubusercontent.com/46878927/234215232-3b8a7c1c-84d0-4ed5-8a06-b8657a4e56c2.png)
![image](https://user-images.githubusercontent.com/46878927/234215518-3e24e018-9c21-49e0-b825-9c40453ae81c.png)
