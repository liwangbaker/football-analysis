
# 足球比赛预测模型

## 项目简介
本项目旨在通过机器学习模型预测英超足球比赛的结果，包括：
- 比赛输赢（主队赢、平局、客队赢）
- 具体比分
- 全场和上半场大小球

## 数据集
数据集包含英超比赛的历史数据，文件名为 `premier_league.csv`。数据字段包括：
- `Date`：比赛日期
- `HomeTeam`：主队
- `AwayTeam`：客队
- `FTHG`：主队全场进球数
- `FTAG`：客队全场进球数
- `FTR`：比赛结果（H=主队赢，D=平局，A=客队赢）
- `HTHG`：主队上半场进球数
- `HTAG`：客队上半场进球数
- 其他统计字段（如赔率、球队排名等）

