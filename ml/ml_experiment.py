#実験用のコード

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,LeaveOneOut,StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.svm import SVC,SVR
import string
import time
from collections import Counter

start_time = time.time()

#データを読み込む
file_paths = []
file_paths.append("./datas/dummy/featureData_2024218193953J.csv")
file_paths.append("./datas/dummy/featureData_2024218193953J.csv")
file_paths.append("./datas/dummy/featureData_2024218193953J.csv")
file_paths.append("./datas/dummy/featureData_2024218193953J.csv")
file_paths.append("./datas/dummy/featureData_2024218193953J.csv")
file_paths.append("./datas/dummy/featureData_2024218193953J.csv")

columns = ['qid','TotalGazeData', 'XStdDev', 'YStdDev', 'GazeOnOpt1', 'GazeOnOpt2', 'GazeOnOpt3', 'GazeOnOpt4', 'TotalGazeOnOpts', 'GazeOnCorrectOpt', 'GazeOnIncorrectOpts', 'GazeOnQuestion', 'CorrectOptIndex', 'VASConfidence', 'VASHesitation', 'IsCorrect','sid']

#被験者idをつけ、データをまとめる
datas = [pd.read_csv(file_path,header=None) for file_path in file_paths]

for index,data in enumerate(datas):
    data['sid'] = string.ascii_uppercase[index]
combined_data = pd.concat(datas,ignore_index=True)

combined_data.columns = columns


#練習問題を取り除く
combined_data = combined_data[combined_data["qid"] != "practice"] 
#解答の正解不正解(True,False)を1,0で示すようにする
combined_data["IsCorrect"] = combined_data["IsCorrect"].apply(lambda x:1 if x else 0) 
#VASConfidenceの値を二値にする(分類用)。0~99を"False"、100~199を"True"とする
combined_data["VASConfidence_cl"] = combined_data["VASConfidence"].apply(lambda x:False if x  <= 99 else True) 
combined_data["VASConfidence_re"] = combined_data["VASConfidence"]

#正解の選択肢があった箇所はカテゴリカルデータデータでは？と思ったので変換できるらしくやって見る。
onehot_correct_opt_index = pd.get_dummies(combined_data['CorrectOptIndex'],drop_first=True,prefix='CorrectOptIndex')
combined_data = pd.concat([combined_data,onehot_correct_opt_index],axis=1)

"""
その他特徴量の追加、調整
GazeOnOptsStdDev: 選択肢上の視線数の分散
"""

combined_data["GazeOnOptsStdDev"] = combined_data[['GazeOnOpt1', 'GazeOnOpt2', 'GazeOnOpt3', 'GazeOnOpt4']].var(axis=1)


#一人の被験者のデータをテストデータとし、残りの被験者のデータを学習データとする(A,B,Cがいたら、A,Bを学習データCをテストデータとして使う)
unique_sids = combined_data['sid'].unique()

"""
実験1
一人の被験者のデータをテストデータとし、残りの被験者のデータを学習データとする(A,B,Cがいたら、A,Bを学習データCをテストデータとして使う)
"""
print("✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿実験1✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿")
#説明変数として利用する特徴量
label_columns_ex1 = ['TotalGazeData', 'XStdDev', 'YStdDev', 'GazeOnOpt1', 'GazeOnOpt2', 'GazeOnOpt3', 'GazeOnOpt4', 'TotalGazeOnOpts', 'GazeOnCorrectOpt', 'GazeOnIncorrectOpts', 'GazeOnQuestion',"GazeOnOptsStdDev",'IsCorrect']

#目的変数として利用する特徴量
target_column_ex1_cl = ['VASConfidence_cl'] #分類で使用する目的変数
target_column_ex1_re = ['VASConfidence_re'] #回帰で使用する目的変数

data_ex1 = combined_data.copy()

accuracy_dict_ex1 = {} #精度を保存する辞書を作っておこう。あとでグラフとか表示する用。
question_num_in_area = {} #データを正解の有無、確信度の有無で分けた際の数をカウント
question_predicted_in_area = {} #データを正解の有無、確信度の有無で分けた際にどの程度その範囲内の要素を推定することができたか。
question_num_in_area_c = {} #データを正解の有無、確信度の有無で分けた際の数をカウント
question_predicted_in_area_c = {} #データを正解の有無、確信度の有無で分けた際にどの程度その範囲内の要素を推定推定することができたか。

print(f"説明変数:{label_columns_ex1}")
print(f"目的変数:{target_column_ex1_cl,target_column_ex1_re}")
for target_sid in unique_sids:
    accuracy_dict_ex1[target_sid] = {}
    question_num_in_area[target_sid] = {}
    question_predicted_in_area[target_sid] = {}
    print("~~~~~~  ~~~~~~  ~~~~~~  ~~~~~~やるよ！！~~~~~~  ~~~~~~  ~~~~~~  ~~~~~~")
    print(f"sid:{target_sid}")
    #一人を除く他の人を学習データとする
    X_train = data_ex1[data_ex1['sid'] != target_sid][label_columns_ex1]
    y_train_cl = data_ex1[data_ex1['sid'] != target_sid][target_column_ex1_cl]
    y_train_re = data_ex1[data_ex1['sid'] != target_sid][target_column_ex1_re]

    X_test = data_ex1[data_ex1['sid'] == target_sid][label_columns_ex1]
    y_test_cl = data_ex1[data_ex1['sid'] == target_sid][target_column_ex1_cl]
    y_test_re = data_ex1[data_ex1['sid'] == target_sid][target_column_ex1_re]



    #標準化
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_trasformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    #scv
    print("-----svc-----")
    #ハイパーパラメーターの候補を指定
    param_grid = {'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto',0.1, 1, 10]}
    svc = SVC()
    #グリッドサーチを行う
    grid_search_svc = GridSearchCV(svc, param_grid, cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=0))
    grid_search_svc.fit(X_train_trasformed, y_train_cl.values.ravel())
    #最適なハイパーパラメーターを表示する
    print("Best parameters: ", grid_search_svc.best_params_)
    #最適なハイパーパラメーターにおける交差検証の正答率を表示する
    print("Best cross-validation accuracy: {:.2f}".format(grid_search_svc.best_score_))
    #テストデータで評価する
    svc_accuracy = grid_search_svc.score(X_test_transformed, y_test_cl)
    print("Accuracy on test set: {:.2f}".format(svc_accuracy))
    #精度を記録
    accuracy_dict_ex1[target_sid]["svc"] = svc_accuracy

    #正解している&自信がある、正解している&自信がない、正解していない&自信がある、正解していない&自信がないの数を出す
    correct_mask = data_ex1[data_ex1['sid'] == target_sid]["IsCorrect"] == 1
    high_confidence_mask = data_ex1[data_ex1['sid'] == target_sid]["VASConfidence_cl"]
    correct_and_high_confidence = correct_mask & high_confidence_mask
    correct_and_low_confidence = correct_mask &  (high_confidence_mask != True)
    incorrect_and_high_confidence = (correct_mask != True) &  high_confidence_mask 
    incorrect_and_low_confidence = (correct_mask != True) &  (high_confidence_mask != True)

    
    print(f"correct_and_high_confidence:{correct_and_high_confidence.sum()}")
    print(f"correct_and_low_confidence:{correct_and_low_confidence.sum()}")
    print(f"incorrect_and_high_confidence:{incorrect_and_high_confidence.sum()}")
    print(f"incorrect_and_low_confidence:{incorrect_and_low_confidence.sum()}")

    question_num_in_area[target_sid]["correct_and_high_confidence"] = correct_and_high_confidence.sum()
    question_num_in_area[target_sid]["correct_and_low_confidence"] = correct_and_low_confidence.sum()
    question_num_in_area[target_sid]["incorrect_and_high_confidence"] = incorrect_and_high_confidence.sum()
    question_num_in_area[target_sid]["incorrect_and_low_confidence"] = incorrect_and_low_confidence.sum()

    print(f"全体のうち正解かつ自信がある、不正解かつ自信がないの割合:{(correct_and_high_confidence.sum() + incorrect_and_low_confidence.sum())/incorrect_and_high_confidence.shape[0]}")
    print(f"全体のうち不正解かつ自信がある、正解かつ自信がないの割合:{(incorrect_and_high_confidence.sum() + correct_and_low_confidence.sum())/incorrect_and_high_confidence.shape[0]}")

    y_pred = grid_search_svc.predict(X_test_transformed)
    true_label = y_pred  == y_test_cl.values.ravel() #予測が正解した物がtrueになるはず
    #各領域の正答率を出そう、だせる？
    print("各領域の正答率")
    if correct_and_high_confidence.sum() > 0:
        mask = correct_and_high_confidence & true_label #うまく予測できた物がtrueになるはず。
        correct_ratio = mask.sum() / correct_and_high_confidence.sum() #これが各項目の分類できた割合になるはず？
        question_predicted_in_area[target_sid]["correct_and_high_confidence"] = mask.sum()
        print(f"正解かつ自信がある：{correct_ratio}")
        mask_false = correct_and_high_confidence & (true_label != True)#うまく予測できなかった物がtrueになるはず。
        # if mask_false.sum() > 0:
            # print(f"うまく予測できなかったものをプリント")
            # print(data_ex1[data_ex1['sid'] == target_sid][mask_false])
    else: 
        question_predicted_in_area[target_sid]["correct_and_high_confidence"] = 0

    if correct_and_low_confidence.sum() > 0:
        mask = correct_and_low_confidence & true_label #うまく予測できた物がtrueになるはず。
        correct_ratio = mask.sum() / correct_and_low_confidence.sum() #これが各項目の分類できた割合になるはず？
        question_predicted_in_area[target_sid]["correct_and_low_confidence"] = mask.sum()
        print(f"正解かつ自信がない：{correct_ratio}")
        mask_false = correct_and_low_confidence & (true_label != True)#うまく予測できなかった物がtrueになるはず。
        # if mask_false.sum() > 0:
            # print(f"うまく予測できなかったものをプリント")
            # print(data_ex1[data_ex1['sid'] == target_sid][mask_false])
    else:
        question_predicted_in_area[target_sid]["correct_and_low_confidence"] = 0

    if incorrect_and_high_confidence.sum() > 0:
        mask = incorrect_and_high_confidence & true_label #うまく予測できた物がtrueになるはず。
        correct_ratio = mask.sum() / incorrect_and_high_confidence.sum() #これが各項目の分類できた割合になるはず？
        question_predicted_in_area[target_sid]["incorrect_and_high_confidence"] = mask.sum()
        print(f"不正解かつ自信がある：{correct_ratio}")
        mask_false = incorrect_and_high_confidence & (true_label != True)#うまく予測できなかった物がtrueになるはず。
        # if mask_false.sum() > 0:
            # print(f"うまく予測できなかったものをプリント")
            # print(data_ex1[data_ex1['sid'] == target_sid][mask_false])
    else:
        question_predicted_in_area[target_sid]["incorrect_and_high_confidence"] = 0

    if incorrect_and_low_confidence.sum() > 0:
        mask = incorrect_and_low_confidence & true_label #うまく予測できた物がtrueになるはず。
        correct_ratio = mask.sum() / incorrect_and_low_confidence.sum() #これが各項目の分類できた割合になるはず？
        question_predicted_in_area[target_sid]["incorrect_and_low_confidence"] = mask.sum()
        print(f"不正解かつ自信がない：{correct_ratio}")
        mask_false = incorrect_and_low_confidence & (true_label != True)#うまく予測できなかった物がtrueになるはず。
        # if mask_false.sum() > 0:
            # print(f"うまく予測できなかったものをプリント")
            # print(data_ex1[data_ex1['sid'] == target_sid][mask_false])
    else :
        question_predicted_in_area[target_sid]["incorrect_and_low_confidence"] = 0

    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true=y_test_cl,y_pred=grid_search_svc.predict(X_test_transformed))
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot(cmap="Reds",ax=ax)
    ax.set_title(f"sid:{target_sid}")
    fig.savefig(f"./figs/ex1/{target_sid}.png")

    #svr
    print("-----svr-----")
    svr = SVR()
    grid_search_svr = GridSearchCV(svr,param_grid,cv=KFold(n_splits=5,shuffle=True,random_state=0))
    grid_search_svr.fit(X_train_trasformed, y_train_re.values.ravel())
    #最適なハイパーパラメーターを表示する
    print("Best parameters: ", grid_search_svr.best_params_)
    #最適なハイパーパラメーターにおける交差検証の正答率を表示する
    print("Best cross-validation accuracy: {:.2f}".format(grid_search_svr.best_score_))
    #テストデータで評価する
    svr_accuracy = grid_search_svr.score(X_test_transformed, y_test_re)
    print("Accuracy on test set: {:.2f}".format(svr_accuracy))
    #精度を記録
    rmse = mean_squared_error(y_test_re,grid_search_svr.predict(X_test_transformed),squared=False)
    mae = mean_absolute_error(y_test_re,grid_search_svr.predict(X_test_transformed))
    accuracy_dict_ex1[target_sid]["svr_rmse"] = rmse
    accuracy_dict_ex1[target_sid]["svr_mae"] = mae



print("✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿実験1(比較用)✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿")
#説明変数として利用する特徴量
label_columns_ex1 = ['IsCorrect']
#目的変数として利用する特徴量
target_column_ex1_cl = ['VASConfidence_cl'] #分類で使用する目的変数
target_column_ex1_re = ['VASConfidence_re'] #回帰で使用する目的変数

data_ex1 = combined_data.copy()

#一人の被験者のデータをテストデータとし、残りの被験者のデータを学習データとする(A,B,Cがいたら、A,Bを学習データCをテストデータとして使う)
unique_sids = combined_data['sid'].unique()
print(f"説明変数:{label_columns_ex1}")
print(f"目的変数:{target_column_ex1_cl,target_column_ex1_re}")
for target_sid in unique_sids:
    print("~~~~~~  ~~~~~~  ~~~~~~  ~~~~~~やるよ！！~~~~~~  ~~~~~~  ~~~~~~  ~~~~~~")
    print(f"sid:{target_sid}")
    question_num_in_area_c[target_sid] = {}
    question_predicted_in_area_c[target_sid] = {}
    #一人を除く他の人を学習データとする
    X_train = data_ex1[data_ex1['sid'] != target_sid][label_columns_ex1]
    y_train_cl = data_ex1[data_ex1['sid'] != target_sid][target_column_ex1_cl]
    y_train_re = data_ex1[data_ex1['sid'] != target_sid][target_column_ex1_re]

    X_test = data_ex1[data_ex1['sid'] == target_sid][label_columns_ex1]
    y_test_cl = data_ex1[data_ex1['sid'] == target_sid][target_column_ex1_cl]
    y_test_re = data_ex1[data_ex1['sid'] == target_sid][target_column_ex1_re]

    #標準化
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_trasformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    #scv
    print("-----svc-----")
    #ハイパーパラメーターの候補を指定
    param_grid = {'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto',0.1, 1, 10]}
    svc = SVC()
    #グリッドサーチを行う
    grid_search_svc = GridSearchCV(svc, param_grid, cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=0))
    grid_search_svc.fit(X_train_trasformed, y_train_cl.values.ravel())
    #最適なハイパーパラメーターを表示する
    print("Best parameters: ", grid_search_svc.best_params_)
    #最適なハイパーパラメーターにおける交差検証の正答率を表示する
    print("Best cross-validation accuracy: {:.2f}".format(grid_search_svc.best_score_))
    #テストデータで評価する
    svc_accuracy = grid_search_svc.score(X_test_transformed, y_test_cl)
    print("Accuracy on test set: {:.2f}".format(svc_accuracy))
    #精度を記録
    accuracy_dict_ex1[target_sid]["svc_c"] = svc_accuracy

    #正解している&自信がある、正解している&自信がない、正解していない&自信がある、正解していない&自信がないの数を出す
    correct_mask = data_ex1[data_ex1['sid'] == target_sid]["IsCorrect"] == 1
    high_confidence_mask = data_ex1[data_ex1['sid'] == target_sid]["VASConfidence_cl"]
    correct_and_high_confidence = correct_mask & high_confidence_mask
    correct_and_low_confidence = correct_mask &  (high_confidence_mask != True)
    incorrect_and_high_confidence = (correct_mask != True) &  high_confidence_mask 
    incorrect_and_low_confidence = (correct_mask != True) &  (high_confidence_mask != True)
    print(f"correct_and_high_confidence:{correct_and_high_confidence.sum()}")
    print(f"correct_and_low_confidence:{correct_and_low_confidence.sum()}")
    print(f"incorrect_and_high_confidence:{incorrect_and_high_confidence.sum()}")
    print(f"incorrect_and_low_confidence:{incorrect_and_low_confidence.sum()}")

    question_num_in_area_c[target_sid]["correct_and_high_confidence"] = correct_and_high_confidence.sum()
    question_num_in_area_c[target_sid]["correct_and_low_confidence"] = correct_and_low_confidence.sum()
    question_num_in_area_c[target_sid]["incorrect_and_high_confidence"] = incorrect_and_high_confidence.sum()
    question_num_in_area_c[target_sid]["incorrect_and_low_confidence"] = incorrect_and_low_confidence.sum()

    print(f"全体のうち正解かつ自信がある、不正解かつ自信がないの割合:{(correct_and_high_confidence.sum() + incorrect_and_low_confidence.sum())/incorrect_and_high_confidence.shape[0]}")
    print(f"全体のうち不正解かつ自信がある、正解かつ自信がないの割合:{(incorrect_and_high_confidence.sum() + correct_and_low_confidence.sum())/incorrect_and_high_confidence.shape[0]}")

    y_pred = grid_search_svc.predict(X_test_transformed)
    true_label = y_pred  == y_test_cl.values.ravel() #予測が正解した物がtrueになるはず
    #各領域の正答率を出そう、だせる？
    print("各領域の正答率")
    if correct_and_high_confidence.sum() > 0:
        mask = correct_and_high_confidence & true_label #うまく予測できた物がtrueになるはず。
        correct_ratio = mask.sum() / correct_and_high_confidence.sum() #これが各項目の分類できた割合になるはず？
        question_predicted_in_area_c[target_sid]["correct_and_high_confidence"] = mask.sum()
        print(f"正解かつ自信がある：{correct_ratio}")
    else:
        question_predicted_in_area_c[target_sid]["correct_and_high_confidence"] = 0

    if correct_and_low_confidence.sum() > 0:
        mask = correct_and_low_confidence & true_label #うまく予測できた物がtrueになるはず。
        correct_ratio = mask.sum() / correct_and_low_confidence.sum() #これが各項目の分類できた割合になるはず？
        question_predicted_in_area_c[target_sid]["correct_and_low_confidence"] = mask.sum()
        print(f"正解かつ自信がない：{correct_ratio}")
    else:
        question_predicted_in_area_c[target_sid]["correct_and_low_confidence"] = 0

    if incorrect_and_high_confidence.sum() > 0:
        mask = incorrect_and_high_confidence & true_label #うまく予測できた物がtrueになるはず。
        correct_ratio = mask.sum() / incorrect_and_high_confidence.sum() #これが各項目の分類できた割合になるはず？
        question_predicted_in_area_c[target_sid]["incorrect_and_high_confidence"] = mask.sum()
        print(f"不正解かつ自信がある：{correct_ratio}")
    else:
        question_predicted_in_area_c[target_sid]["incorrect_and_high_confidence"] = 0

    if incorrect_and_low_confidence.sum() > 0:
        mask = incorrect_and_low_confidence & true_label #うまく予測できた物がtrueになるはず。
        correct_ratio = mask.sum() / incorrect_and_low_confidence.sum() #これが各項目の分類できた割合になるはず？
        question_predicted_in_area_c[target_sid]["incorrect_and_low_confidence"] = mask.sum()
        print(f"不正解かつ自信がない：{correct_ratio}")
    else:
        question_predicted_in_area_c[target_sid]["incorrect_and_low_confidence"] = 0


    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true=y_test_cl,y_pred=grid_search_svc.predict(X_test_transformed))
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot(cmap="Reds",ax=ax)
    ax.set_title(f"sid:{target_sid}")
    fig.savefig(f"./figs/ex1/{target_sid}_c.png")

    #svr
    print("-----svr-----")
    svr = SVR()
    grid_search_svr = GridSearchCV(svr,param_grid,cv=KFold(n_splits=5,shuffle=True,random_state=0))
    grid_search_svr.fit(X_train_trasformed, y_train_re.values.ravel())
    #最適なハイパーパラメーターを表示する
    print("Best parameters: ", grid_search_svr.best_params_)
    #最適なハイパーパラメーターにおける交差検証の正答率を表示する
    print("Best cross-validation accuracy: {:.2f}".format(grid_search_svr.best_score_))
    #テストデータで評価する
    svr_accuracy = grid_search_svr.score(X_test_transformed, y_test_re)
    print("Accuracy on test set: {:.2f}".format(svr_accuracy))
    #精度を記録
    rmse = mean_squared_error(y_test_re,grid_search_svr.predict(X_test_transformed),squared=False)
    mae = mean_absolute_error(y_test_re,grid_search_svr.predict(X_test_transformed))
    accuracy_dict_ex1[target_sid]["svr_c_rmse"] = rmse
    accuracy_dict_ex1[target_sid]["svr_c_mae"] = mae

#グラフの作成
# 被験者ごとにsvcとsvc_cの値を抽出
subjects = list(accuracy_dict_ex1.keys())
svc_values = []
svc_c_values = []
svr_rmse_values = []
svr_mae_values = []
svr_c_rmse_values = []
svr_c_mae_values = []
for subject in subjects:
    svc_values.append(accuracy_dict_ex1[subject]['svc'])
    svc_c_values.append(accuracy_dict_ex1[subject]['svc_c'])
    svr_rmse_values.append(accuracy_dict_ex1[subject]['svr_rmse'])
    svr_mae_values.append(accuracy_dict_ex1[subject]['svr_mae'])
    svr_c_rmse_values.append(accuracy_dict_ex1[subject]['svr_c_rmse'])
    svr_c_mae_values.append(accuracy_dict_ex1[subject]['svr_c_mae'])

print(f"svc_values:{sum(svc_values)/len(svc_values)}")
print(f"svc_c_values:{sum(svc_c_values)/len(svc_c_values)}")
print(f"svr_rmse_values:{sum(svr_rmse_values)/len(svr_rmse_values)}")
print(f"svr_c_rmse_values:{sum(svr_c_rmse_values)/len(svr_c_rmse_values)}")
print(f"svr_mae_values:{sum(svr_mae_values)/len(svr_mae_values)}")
print(f"svr_c_mae_values:{sum(svr_c_mae_values)/len(svr_c_mae_values)}")


# グラフの作成 SVC用
bar_width = 0.35
index = range(len(subjects))

fig, ax = plt.subplots()
bar1 = ax.bar(index, svc_values, bar_width, label='SVC')
bar2 = ax.bar([i + bar_width for i in index], svc_c_values, bar_width, label='SVC_onlyIsCorrect')

# グラフの装飾
ax.set_xlabel('subjects')
ax.set_ylabel('accuracy')
ax.set_title('compare accuracy')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(subjects)
ax.legend()

fig.savefig(f"./figs/ex1/svc.png")

# グラフの作成 SVR_rmse用
bar_width = 0.35
index = range(len(subjects))

fig, ax = plt.subplots()
bar1 = ax.bar(index, svr_rmse_values, bar_width, label='SVR')
bar2 = ax.bar([i + bar_width for i in index], svr_c_rmse_values, bar_width, label='SVR_onlyIsCorrect')

# グラフの装飾
ax.set_xlabel('subjects')
ax.set_ylabel('RMSE')
ax.set_title('compare RMSE')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(subjects)
ax.legend()

fig.savefig(f"./figs/ex1/svr_rmse.png")

# グラフの作成 SVR_mae用
bar_width = 0.35
index = range(len(subjects))

fig, ax = plt.subplots()
bar1 = ax.bar(index, svr_mae_values, bar_width, label='SVR')
bar2 = ax.bar([i + bar_width for i in index], svr_c_mae_values, bar_width, label='SVR_onlyIsCorrect')

# グラフの装飾
ax.set_xlabel('subjects')
ax.set_ylabel('MAE')
ax.set_title('compare MAE')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(subjects)
ax.legend()

fig.savefig(f"./figs/ex1/svr_mae.png")


#グラフを作りたい、各領域？の数を示すもの（自信ありとか確信ありとか）
fig, ax = plt.subplots()

for index_subject,subject in enumerate(question_num_in_area):
    areas = ["correct_and_high_confidence","correct_and_low_confidence","incorrect_and_high_confidence","incorrect_and_low_confidence"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    correct_high = question_num_in_area[subject]["correct_and_high_confidence"]
    correct_low = question_num_in_area[subject]["correct_and_low_confidence"]
    incorrect_high = question_num_in_area[subject]["incorrect_and_high_confidence"]
    incorrect_low = question_num_in_area[subject]["incorrect_and_low_confidence"]

    margin = 0.2 #被験者ごとの距離
    width = (1 - margin) / len(areas) #棒グラフの太さ
    for index_area,area in enumerate(areas):
        data = question_num_in_area[subject][area]
        pos = index_subject + margin + width * index_area
        if index_subject == 0:
            ax.bar(pos,data,width=width,color=colors[index_area],align="edge",label=area) #一回だけラベルをつける
        else:
            ax.bar(pos,data,width=width,color=colors[index_area],align="edge")
    steps = [ i +  margin + (1 - margin)/2 for i in range(len(question_num_in_area.keys()))]
    ax.set_xticks(steps)
    ax.set_xticklabels(question_num_in_area.keys())
    ax.set_xlabel('subjects')
    ax.set_ylabel('question count')
ax.set_ylim(0, 40)
ax.legend()
fig.savefig(f"./figs/ex1/correct_confidence.png")


#グラフを作りたい、各領域？の数を示すもの（自信ありとか確信ありとか）をどの程度うまく推定できたか
fig, ax = plt.subplots()

for index_subject,subject in enumerate(question_predicted_in_area):
    areas = ["correct_and_high_confidence","correct_and_low_confidence","incorrect_and_high_confidence","incorrect_and_low_confidence"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    predicted_per = {}
    #ゼロ徐さんが発生しているかな？多分0になっている？でも普通に処理がつづている様に見える？NANになって処理が続いているっぽい。でNANの部分は多分0として扱われているっぽいグラフにする時は
    # print(question_predicted_in_area[subject])
    # print(question_num_in_area[subject])
    predicted_per["correct_and_high_confidence"] = question_predicted_in_area[subject]["correct_and_high_confidence"] / question_num_in_area[subject]["correct_and_high_confidence"]
    predicted_per["correct_and_low_confidence"] = question_predicted_in_area[subject]["correct_and_low_confidence"] / question_num_in_area[subject]["correct_and_low_confidence"]
    predicted_per["incorrect_and_high_confidence"] = question_predicted_in_area[subject]["incorrect_and_high_confidence"] / question_num_in_area[subject]["incorrect_and_high_confidence"]
    predicted_per["incorrect_and_low_confidence"] = question_predicted_in_area[subject]["incorrect_and_low_confidence"] / question_num_in_area[subject]["incorrect_and_low_confidence"]
    # print(predicted_per)
    margin = 0.2 #被験者ごとの距離
    width = (1 - margin) / len(areas) #棒グラフの太さ
    for index_area,area in enumerate(areas):
        data = predicted_per[area]
        pos = index_subject + margin + width * index_area
        ax.bar(pos,data,width=width,color=colors[index_area],align="edge")
    steps = [ i +  margin + (1 - margin)/2 for i in range(len(question_predicted_in_area.keys()))]
    ax.set_xticks(steps)
    ax.set_xticklabels(question_predicted_in_area.keys())
    ax.set_xlabel('subjects')
    ax.set_ylabel('predicted_per')

fig.savefig(f"./figs/ex1/question_predicted_in_area.png")

#ヒートマップ作成。各領域の正答率の 人を区別せず

fig, ax = plt.subplots()
count_confidence_and_correct = {"correct_and_high_confidence":0,"correct_and_low_confidence":0,"incorrect_and_high_confidence":0,"incorrect_and_low_confidence":0}
count_confidence_and_correct_predicted = {"correct_and_high_confidence":0,"correct_and_low_confidence":0,"incorrect_and_high_confidence":0,"incorrect_and_low_confidence":0}
for index_subject,subject in enumerate(question_predicted_in_area):
    areas = ["correct_and_high_confidence","correct_and_low_confidence","incorrect_and_high_confidence","incorrect_and_low_confidence"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for area in areas:
        count_confidence_and_correct[area] += question_num_in_area[subject][area]
        count_confidence_and_correct_predicted[area] += question_predicted_in_area[subject][area]

predicted_per_in_areas = {}
for key in count_confidence_and_correct.keys():
    predicted_per_in_areas[key] = count_confidence_and_correct_predicted[key] /count_confidence_and_correct[key]
#[正解、自信あり,正解、自信なし]
#[不正解、自信あり，不正解，自信なし]
predicted_per = pd.DataFrame([[predicted_per_in_areas["correct_and_high_confidence"],predicted_per_in_areas["correct_and_low_confidence"]],[predicted_per_in_areas["incorrect_and_high_confidence"],predicted_per_in_areas["incorrect_and_low_confidence"]]],index=["correct","incorrect"],columns=["high cofidence","low confidence"])
# ax.pcolor(predicted_per,cmap="Reds")
sns.heatmap(predicted_per,annot=True)

fig.savefig(f"./figs/ex1/heatmap_predicted.png")

#ヒートマップ作成。各領域の正答率ので、説明変数に生後情報のみを使った場合。人を区別せず

fig, ax = plt.subplots()
count_confidence_and_correct = {"correct_and_high_confidence":0,"correct_and_low_confidence":0,"incorrect_and_high_confidence":0,"incorrect_and_low_confidence":0}
count_confidence_and_correct_predicted = {"correct_and_high_confidence":0,"correct_and_low_confidence":0,"incorrect_and_high_confidence":0,"incorrect_and_low_confidence":0}
for index_subject,subject in enumerate(question_predicted_in_area_c):
    areas = ["correct_and_high_confidence","correct_and_low_confidence","incorrect_and_high_confidence","incorrect_and_low_confidence"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for area in areas:
        count_confidence_and_correct[area] += question_num_in_area_c[subject][area]
        count_confidence_and_correct_predicted[area] += question_predicted_in_area_c[subject][area]

predicted_per_in_areas = {}
for key in count_confidence_and_correct.keys():
    predicted_per_in_areas[key] = count_confidence_and_correct_predicted[key] /count_confidence_and_correct[key]
#[正解、自信あり,正解、自信なし]
#[不正解、自信あり，不正解，自信なし]
predicted_per = pd.DataFrame([[predicted_per_in_areas["correct_and_high_confidence"],predicted_per_in_areas["correct_and_low_confidence"]],[predicted_per_in_areas["incorrect_and_high_confidence"],predicted_per_in_areas["incorrect_and_low_confidence"]]],index=["correct","incorrect"],columns=["high cofidence","low confidence"])
# ax.pcolor(predicted_per,cmap="Reds")
sns.heatmap(predicted_per,annot=True)

fig.savefig(f"./figs/ex1/heatmap_predicted_com.png")

#ヒートマップ作成。各領域の正答率の。個人個人で作ってみる。

for index_subject,subject in enumerate(question_predicted_in_area):
    fig, ax = plt.subplots()
    areas = ["correct_and_high_confidence","correct_and_low_confidence","incorrect_and_high_confidence","incorrect_and_low_confidence"]
    correct_percent = {}
    for area in areas:
        if question_num_in_area[subject][area] != 0:
            correct_percent[area] = question_predicted_in_area[subject][area] / question_num_in_area[subject][area]
        else:
            #該当するものがなかったら0%とする。
            correct_percent[area] = 0
    predicted_per = pd.DataFrame([[correct_percent["correct_and_high_confidence"],correct_percent["correct_and_low_confidence"]],[correct_percent["incorrect_and_high_confidence"],correct_percent["incorrect_and_low_confidence"]]],index=["correct","incorrect"],columns=["high cofidence","low confidence"])
    sns.heatmap(predicted_per,annot=True)
    fig.savefig(f"./figs/ex1/heatmap_predicted_{subject}.png")


"""
実験2
特定の被験者のデータのみを用いた場合LeaveOneOutで評価
"""
print("✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿実験2✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿")

#説明変数として利用する特徴量
label_columns_ex2 = ['TotalGazeData', 'XStdDev', 'YStdDev', 'GazeOnOpt1', 'GazeOnOpt2', 'GazeOnOpt3', 'GazeOnOpt4', 'TotalGazeOnOpts', 'GazeOnCorrectOpt', 'GazeOnIncorrectOpts', 'GazeOnQuestion',"GazeOnOptsStdDev",'IsCorrect']
#目的変数として利用する特徴量
target_column_ex2_cl = ['VASConfidence_cl'] #分類で使用する目的変数
target_column_ex2_re = ['VASConfidence_re'] #回帰で使用する目的変数

data_ex2 = combined_data.copy()
accuracy_dict_ex2 = {}

print(f"説明変数:{label_columns_ex2}")
print(f"目的変数:{target_column_ex2_cl,target_column_ex2_re}")
for target_sid in unique_sids:
    accuracy_dict_ex2[target_sid] = {}
    print("~~~~~~  ~~~~~~  ~~~~~~  ~~~~~~やるよ！！~~~~~~  ~~~~~~  ~~~~~~  ~~~~~~")
    print(f"sid:{target_sid}")
    
    X = data_ex2[data_ex2['sid'] == target_sid][label_columns_ex2]
    y_cl = data_ex2[data_ex2['sid'] == target_sid][target_column_ex2_cl]
    y_re = data_ex2[data_ex2['sid'] == target_sid][target_column_ex2_re]
    
    #numpyに変換
    X = X.values
    y_cl = y_cl.values
    y_re = y_re.values

    #scv
    hyperparameters = []
    print("-----svc-----")
    loo_svc = LeaveOneOut()
    actual_labels = []
    predicted_labels = []
    for train_index,test_index in loo_svc.split(X):
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y_cl[train_index],y_cl[test_index]
        #標準化
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_transformed = scaler.transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        #ハイパーパラメーターの候補を指定
        param_grid = {'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto',0.1, 1, 10]}
        svc = SVC()
        #グリッドサーチを行う(内側のクロスバリデーション)
        grid_search_svc = GridSearchCV(svc, param_grid, cv=5)
        grid_search_svc.fit(X_train_transformed,y_train.ravel())
        #最適なハイパーパラメーターを追加する
        # print(grid_search_svc.best_params_)
        hyperparameters.append(grid_search_svc.best_params_)
        #予測を行う
        best_model = grid_search_svc.best_estimator_
        y_pred = best_model.predict(X_test_transformed)
        predicted_labels.extend(y_pred)
        actual_labels.extend(y_test)
    accuracy = accuracy_score(actual_labels, predicted_labels)
    print("Accuracy:", accuracy)
    param_counter = Counter(str(param) for param in hyperparameters)
    print("ハイパーパラメータ:", param_counter.most_common(5))
    accuracy_dict_ex2[target_sid]["svc"] = accuracy

    
    #svr
    print("-----svr-----")
    hyperparameters = []
    loo_svc = LeaveOneOut()
    actual_values = []
    predicted_values = []
    for train_index,test_index in loo_svc.split(X):
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y_re[train_index],y_re[test_index]
        #標準化
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_transformed = scaler.transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        #ハイパーパラメーターの候補を指定
        param_grid = {'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto',0.1, 1, 10]}
        svr = SVR()
        #グリッドサーチを行う(内側のクロスバリデーション)
        grid_search_svr = GridSearchCV(svr,param_grid,cv=5)
        grid_search_svr.fit(X_train_transformed,y_train.ravel())
        #予測を行う
        best_model = grid_search_svr.best_estimator_
        #最適なハイパーパラメーターを追加する
        hyperparameters.append(grid_search_svr.best_params_)
        y_pred = best_model.predict(X_test_transformed)
        actual_values.extend(y_test)
        predicted_values.extend(y_pred)
    rmse = mean_squared_error(actual_values,predicted_values,squared=False)
    mae = mean_absolute_error(actual_values,predicted_values)
    print("RMse:",rmse)
    param_counter = Counter(str(param) for param in hyperparameters)
    print("ハイパーパラメータ:", param_counter.most_common(5))
    accuracy_dict_ex2[target_sid]["svr_rmse"] = rmse
    accuracy_dict_ex2[target_sid]["svr_mae"] = mae

print("✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿実験2(比較用)✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿")
label_columns_ex2 = ['IsCorrect']
#目的変数として利用する特徴量
target_column_ex2_cl = ['VASConfidence_cl'] #分類で使用する目的変数
target_column_ex2_re = ['VASConfidence_re'] #回帰で使用する目的変数

data_ex2 = combined_data.copy()

print(f"説明変数:{label_columns_ex2}")
print(f"目的変数:{target_column_ex2_cl,target_column_ex2_re}")
for target_sid in unique_sids:
    print("~~~~~~  ~~~~~~  ~~~~~~  ~~~~~~やるよ！！~~~~~~  ~~~~~~  ~~~~~~  ~~~~~~")
    print(f"sid:{target_sid}")
    
    X = data_ex2[data_ex2['sid'] == target_sid][label_columns_ex2]
    y_cl = data_ex2[data_ex2['sid'] == target_sid][target_column_ex2_cl]
    y_re = data_ex2[data_ex2['sid'] == target_sid][target_column_ex2_re]
    
    #numpyに変換
    X = X.values
    y_cl = y_cl.values
    y_re = y_re.values

    #scv
    print("-----svc-----")
    hyperparameters = []
    loo_svc = LeaveOneOut()
    actual_labels = []
    predicted_labels = []
    for train_index,test_index in loo_svc.split(X):
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y_cl[train_index],y_cl[test_index]
        #標準化
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_transformed = scaler.transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        #ハイパーパラメーターの候補を指定
        param_grid = {'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto',0.1, 1, 10]}
        svc = SVC()
        #グリッドサーチを行う(内側のクロスバリデーション)
        grid_search_svc = GridSearchCV(svc, param_grid, cv=5)
        grid_search_svc.fit(X_train_transformed,y_train.ravel())
        hyperparameters.append(grid_search_svc.best_params_)
        #予測を行う
        best_model = grid_search_svc.best_estimator_
        y_pred = best_model.predict(X_test_transformed)
        predicted_labels.extend(y_pred)
        actual_labels.extend(y_test)
    accuracy = accuracy_score(actual_labels, predicted_labels)
    print("Accuracy:", accuracy)
    param_counter = Counter(str(param) for param in hyperparameters)
    print("ハイパーパラメータ:", param_counter.most_common(5))
    accuracy_dict_ex2[target_sid]["svc_c"] = accuracy
    
    #svr
    print("-----svr-----")
    hyperparameters = []
    loo_svc = LeaveOneOut()
    actual_values = []
    predicted_values = []
    for train_index,test_index in loo_svc.split(X):
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y_re[train_index],y_re[test_index]
        #標準化
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_transformed = scaler.transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        #ハイパーパラメーターの候補を指定
        param_grid = {'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto',0.1, 1, 10]}
        svr = SVR()
        #グリッドサーチを行う(内側のクロスバリデーション)
        grid_search_svr = GridSearchCV(svr,param_grid,cv=5)
        grid_search_svr.fit(X_train_transformed,y_train.ravel())
        hyperparameters.append(grid_search_svr.best_params_)
        #予測を行う
        best_model = grid_search_svr.best_estimator_
        y_pred = best_model.predict(X_test_transformed)
        actual_values.extend(y_test)
        predicted_values.extend(y_pred)
    rmse = mean_squared_error(actual_values,predicted_values,squared=False)
    mae = mean_absolute_error(actual_values,predicted_values)
    print("RMse:",rmse)
    param_counter = Counter(str(param) for param in hyperparameters)
    print("ハイパーパラメータ:", param_counter.most_common(5))
    accuracy_dict_ex2[target_sid]["svr_c_rmse"] = rmse
    accuracy_dict_ex2[target_sid]["svr_c_mae"] = mae


#グラフの作成
# 被験者ごとにsvcとsvc_cの値を抽出
subjects = list(accuracy_dict_ex2.keys())
svc_values = []
svc_c_values = []
svr_rmse_values = []
svr_mae_values = []
svr_c_rmse_values = []
svr_c_mae_values = []
for subject in subjects:
    svc_values.append(accuracy_dict_ex2[subject]['svc'])
    svc_c_values.append(accuracy_dict_ex2[subject]['svc_c'])
    svr_rmse_values.append(accuracy_dict_ex2[subject]['svr_rmse'])
    svr_mae_values.append(accuracy_dict_ex2[subject]['svr_mae'])
    svr_c_rmse_values.append(accuracy_dict_ex2[subject]['svr_c_rmse'])
    svr_c_mae_values.append(accuracy_dict_ex2[subject]['svr_c_mae'])

print(f"svc_values:{sum(svc_values)/len(svc_values)}")
print(f"svc_c_values:{sum(svc_c_values)/len(svc_c_values)}")
print(f"svr_rmse_values:{sum(svr_rmse_values)/len(svr_rmse_values)}")
print(f"svr_c_rmse_values:{sum(svr_c_rmse_values)/len(svr_c_rmse_values)}")
print(f"svr_mae_values:{sum(svr_mae_values)/len(svr_mae_values)}")
print(f"svr_c_mae_values:{sum(svr_c_mae_values)/len(svr_c_mae_values)}")


# グラフの作成 SVC用
bar_width = 0.35
index = range(len(subjects))

fig, ax = plt.subplots()
bar1 = ax.bar(index, svc_values, bar_width, label='SVC')
bar2 = ax.bar([i + bar_width for i in index], svc_c_values, bar_width, label='SVC_onlyIsCorrect')

# グラフの装飾
ax.set_xlabel('subjects')
ax.set_ylabel('accuracy')
ax.set_title('compare accuracy')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(subjects)
ax.legend()

fig.savefig(f"./figs/ex2/svc.png")

# グラフの作成 SVR_rmse用
bar_width = 0.35
index = range(len(subjects))

fig, ax = plt.subplots()
bar1 = ax.bar(index, svr_rmse_values, bar_width, label='SVR')
bar2 = ax.bar([i + bar_width for i in index], svr_c_rmse_values, bar_width, label='SVR_onlyIsCorrect')

# グラフの装飾
ax.set_xlabel('subjects')
ax.set_ylabel('RMSE')
ax.set_title('compare RMSE')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(subjects)
ax.legend()

fig.savefig(f"./figs/ex2/svr_rmse.png")

# グラフの作成 SVR_mae用
bar_width = 0.35
index = range(len(subjects))

fig, ax = plt.subplots()
bar1 = ax.bar(index, svr_mae_values, bar_width, label='SVR')
bar2 = ax.bar([i + bar_width for i in index], svr_c_mae_values, bar_width, label='SVR_onlyIsCorrect')

# グラフの装飾
ax.set_xlabel('subjects')
ax.set_ylabel('MAE')
ax.set_title('compare MAE')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(subjects)
ax.legend()

fig.savefig(f"./figs/ex2/svr_mae.png")


end_time = time.time()
execution_time = end_time - start_time
print(f"実行時間: {execution_time}秒")
