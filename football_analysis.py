# football_analysis.py

# 1. 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import os

# 2. 加载数据
def load_data(file_path):
    """
    加载数据集
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 文件 {file_path} 不存在!")

    # 加载数据
    df = pd.read_csv(file_path)
    print("数据加载成功!")

    # 打印数据的前5行
    print("\n数据的前5行:")
    print(df.head())

    # 打印数据的基本信息
    print("\n数据的基本信息:")
    print(df.info())

    # 打印数据的统计摘要
    print("\n数据的统计摘要:")
    print(df.describe(include='all'))  # 包括非数值列

    return df

# 3. 数据清洗
def clean_data(df):
    """
    数据清洗
    """
    # 检查缺失值
    print("\n缺失值统计:")
    print(df.isnull().sum())

    # 填充缺失值（例如用平均值填充数值列）
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # 删除重复值
    df.drop_duplicates(inplace=True)

    print("\n数据清洗完成!")
    return df

# 4. 数据探索与分析
def explore_data(df):
    """
    数据探索与分析
    """
    # 比赛结果分布
    print("\n比赛结果分布:")
    print(df['FTR'].value_counts())

    # 主客场进球数分布
    print("\n主队平均进球数: {:.2f}".format(df['FTHG'].mean()))
    print("客队平均进球数: {:.2f}".format(df['FTAG'].mean()))

# 5. 特征工程
def feature_engineering(df):
    """
    特征工程
    """
    # 删除非数值列（如果需要）
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    print("非数值列:", non_numeric_columns)

    # 选择特征和目标变量
    features = ['FTHG', 'FTAG', 'HTHG', 'HTAG']  # 使用全场和半场进球数作为特征
    target = 'FTR'  # 假设 'FTR' 是目标变量列

    # 将目标变量编码为数值
    label_encoder = LabelEncoder()
    df[target] = label_encoder.fit_transform(df[target])

    X = df[features]
    y = df[target]

    print("\n特征矩阵 X:")
    print(X.head())

    print("\n目标变量 y:")
    print(y.head())

    print("\n特征工程完成!")
    return X, y, label_encoder

# 6. 训练模型
def train_model(X, y):
    """
    训练模型
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n模型准确率: {accuracy:.2f}")

    return model, X_test, y_test, y_pred

# 7. 预测新比赛结果
def predict_new_match(model, label_encoder, home_team, away_team, team_avg_goals, df):
    """
    预测新比赛结果
    """
    # 获取主队和客队的平均进球数
    home_avg_fthg = team_avg_goals.loc[team_avg_goals['Team'] == home_team, 'AvgFTHG'].values[0]
    away_avg_ftag = team_avg_goals.loc[team_avg_goals['Team'] == away_team, 'AvgFTAG'].values[0]
    home_avg_hthg = team_avg_goals.loc[team_avg_goals['Team'] == home_team, 'AvgHTHG'].values[0]
    away_avg_htag = team_avg_goals.loc[team_avg_goals['Team'] == away_team, 'AvgHTAG'].values[0]

    # 创建新比赛的特征矩阵
    new_match = {
        'FTHG': home_avg_fthg,  # 主队历史平均全场进球数
        'FTAG': away_avg_ftag,  # 客队历史平均全场进球数
        'HTHG': home_avg_hthg,  # 主队历史平均半场进球数
        'HTAG': away_avg_htag   # 客队历史平均半场进球数
    }
    new_match_df = pd.DataFrame([new_match])

    # 预测比赛结果
    prediction = model.predict(new_match_df)

    # 将预测结果转换为比赛结果
    result_map = {0: '客队赢', 1: '平局', 2: '主队赢'}
    predicted_result = result_map[prediction[0]]

    print(f"\n预测比赛结果: {home_team} vs {away_team}")
    print(f"预测结果: {predicted_result}")

    # 预测具体比分
    predict_score(df, home_team, away_team, home_avg_fthg, away_avg_ftag)

    # 预测大小球
    predict_over_under(home_avg_fthg, away_avg_ftag, home_avg_hthg, away_avg_htag)

# 8. 预测具体比分
def predict_score(df, home_team, away_team, home_avg_fthg, away_avg_ftag):
    """
    预测具体比分
    """
    # 准备训练数据
    X = df[['FTHG', 'FTAG']]  # 特征: 主队和客队的全场进球数
    y_home = df['FTHG']  # 目标: 主队进球数
    y_away = df['FTAG']  # 目标: 客队进球数

    # 训练随机森林回归模型
    model_home = RandomForestRegressor(random_state=42)
    model_away = RandomForestRegressor(random_state=42)
    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    # 创建新比赛的特征矩阵（确保列名与训练数据一致）
    new_match = pd.DataFrame([[home_avg_fthg, away_avg_ftag]], columns=['FTHG', 'FTAG'])

    # 预测新比赛的比分
    y_home_pred = int(np.round(model_home.predict(new_match)[0]))
    y_away_pred = int(np.round(model_away.predict(new_match)[0]))

    print(f"\n预测比分: {home_team} {y_home_pred} - {y_away_pred} {away_team}")

# 9. 预测大小球
def predict_over_under(home_avg_fthg, away_avg_ftag, home_avg_hthg, away_avg_htag):
    """
    预测大小球
    """
    # 全场大小球
    total_goals = home_avg_fthg + away_avg_ftag
    over_under_total = "大球" if total_goals > 2.5 else "小球"

    # 上半场大小球
    first_half_goals = home_avg_hthg + away_avg_htag
    over_under_first_half = "大球" if first_half_goals > 1.5 else "小球"

    print(f"\n全场大小球预测: {over_under_total} (总进球数: {total_goals:.2f})")
    print(f"上半场大小球预测: {over_under_first_half} (上半场进球数: {first_half_goals:.2f})")

# 10. 自动化更新流程
def automated_pipeline(file_path, home_team, away_team):
    """
    自动化更新流程
    """
    try:
        # 1. 加载数据
        df = load_data(file_path)

        # 2. 数据清洗
        df = clean_data(df)

        # 3. 数据探索与分析
        explore_data(df)

        # 4. 特征工程
        X, y, label_encoder = feature_engineering(df)

        # 5. 训练模型
        model, X_test, y_test, y_pred = train_model(X, y)

        # 6. 计算球队平均进球数
        home_avg_goals = df.groupby('HomeTeam')['FTHG'].mean().reset_index()
        home_avg_goals.columns = ['Team', 'AvgFTHG']
        away_avg_goals = df.groupby('AwayTeam')['FTAG'].mean().reset_index()
        away_avg_goals.columns = ['Team', 'AvgFTAG']
        home_avg_hthg = df.groupby('HomeTeam')['HTHG'].mean().reset_index()
        home_avg_hthg.columns = ['Team', 'AvgHTHG']
        away_avg_htag = df.groupby('AwayTeam')['HTAG'].mean().reset_index()
        away_avg_htag.columns = ['Team', 'AvgHTAG']
        team_avg_goals = pd.merge(home_avg_goals, away_avg_goals, on='Team', how='outer')
        team_avg_goals = pd.merge(team_avg_goals, home_avg_hthg, on='Team', how='outer')
        team_avg_goals = pd.merge(team_avg_goals, away_avg_htag, on='Team', how='outer')

        # 7. 预测新比赛结果
        predict_new_match(model, label_encoder, home_team=home_team, away_team=away_team, team_avg_goals=team_avg_goals, df=df)

    except Exception as e:
        print(f"程序运行出错: {e}")

# 主函数
def main():
    # 文件路径
    file_path = '/Users/liwang/Desktop/data/premier_league.csv'

    # 预测的比赛
    home_team = 'Newcastle'
    away_team = 'Bournemouth'

    # 运行自动化更新流程
    automated_pipeline(file_path, home_team, away_team)

# 运行主函数
if __name__ == "__main__":
    main()