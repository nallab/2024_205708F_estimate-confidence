#csvで管理していた英語四択問題を実験ページにて出題できるようなjson形式に変換するためのコード

import csv
import json

# CSVファイルのパス
csv_file_path = 'english_exam.csv'

# JSONファイルのパス
json_file_path = 'english_exam.json'

# CSVファイルを読み込む
csv_data = []
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        #データを整形する
        answer_candidates = []
        for num in range(1,5):
            answer_candidates.append(row["選択肢"+str(num)])
        qid = str(row["何年度"]) + "_" + str(row["第何回"]) + "_" + str(row["何級"]) + "_" + str(row["問題番号"]).zfill(2)

        fixed_row = {"qid": qid,"question": row["問題文"],"answer_entity":row["正解"],"answer_candidates":answer_candidates,"answer_index":row["正解の選択肢"],"grade":row["何級"]}
        csv_data.append(fixed_row)

print(set(data["grade"] for data in csv_data))

#各級から何問づつ取り出すか指定する(1,pre1,2,pre2,3,4,5)
#(7,7,7,7,7,7,7:合計49問)から(7,10,10,10,5,4,4:合計50問)に変更する
#question_length_per_grade = 8
class_problem_counts = {
    '5': 4,
    '4': 4,
    '3': 5,
    'Pre-2': 10,
    '2': 10,
    'Pre-1': 10,
    '1': 7
}

selected_questions = []
question_count_dict = {}

for question_data in csv_data:
    grade = question_data["grade"]
    if class_problem_counts[grade] > 0:
        selected_questions.append(question_data)
        class_problem_counts[grade] -= 1
    

# CSVデータをJSONに変換して書き込む
with open(json_file_path, 'w') as json_file:
    json.dump(selected_questions, json_file, indent=4)

print(f'{csv_file_path} を {json_file_path} に変換しました。')
