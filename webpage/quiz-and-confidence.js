import { setCurrentQestionNumber,showFinishMessage,answerData,convertGazeDataToFeatureData,concatAnswerDataToFeatureData,downloadFeatureData,showMessageBeforeExperiment,downloadRawGazeData, addQuestionsArea, downloadQuestionsAreas,answerDataDownload} from "./main.js";

let questions;
let vasValue = [-1,-1]; //選択されたvasの値
const vasElement = document.getElementById("vasElement");
const vasContainer = document.querySelectorAll("#vasElement .container .vas-container");
//const vas = document.querySelector("#vasElement .container .vas");
//const value = document.getElementById("value");
let countQuestion = 0; //今何問目か
let maxQuestionNum = 100; //最大問題を何問出題するか
let practiceMode; //練習問題を出題しているかどうか
//const answerData = []; // 回答、自信度情報を格納する配列

// 四択問題のデータ
export function loadQuestion(filename,receivedPracticeMode) {
    practiceMode = receivedPracticeMode;
    countQuestion = 0; 
    fetch(filename)
        .then(response => response.json())
        .then(data => {
            // データの取得が成功した場合の処理
            questions = data
            console.log(questions);
            console.log(questions.length);
            // ここから取得したデータを使った処理を行います
            displayQuestion()

        })
        .catch(error => {
            // データの取得が失敗した場合の処理
            console.error('ファイルが見つかりません。代わりのファイルを読み込みます:', error);
            fetch("dummy.json")
                .then(response => response.json())
                .then(data => {
                    // データの取得が成功した場合の処理
                    questions = data
                    console.log(questions);
                    console.log(questions.length);
                    // ここから取得したデータを使った処理を行います
                    displayQuestion()
                })
                .catch(error => {
                    console.error("データの読み込みに失敗しました。",error);
                })
        });
}


//let currentQuestionNumber; //現在の問題id

// 問題をランダムに選択するための関数
function getRandomQuestion() {
    let selectedIndex = Math.floor(Math.random() * questions.length)
    let showQuestion = questions[selectedIndex]
    questions.splice(selectedIndex, 1) //同じ問題が選択されないようにする
    return showQuestion;
}

// 問題を表示する関数
export function displayQuestion() {
    countQuestion += 1; //出題した問題数カウントを進める
    const questionElement = document.getElementById("question");
    const choicesElement = document.getElementById("choices");
    const questionArea = document.getElementById("questionArea");

    // ランダムな問題を取得
    const question = getRandomQuestion();

    // 問題文を表示
    questionArea.style.display = "flex";
    questionElement.innerText = question.question;
    //questionElement.style.display = "block";

    // 選択肢を表示する(dev1_question.json形式)
    // let answer_entity = question.answer_entity
    // let answer_candidates = question.answer_candidates
    // let answer_index = answer_candidates.indexOf(answer_entity)
    // let selected_candidates_index = []
    // //選択肢から正解を除く三つの回答候補を選ぶ
    // while (selected_candidates_index.length < 3) {
    //     let randomIndex = Math.floor(Math.random() * answer_candidates.length)
    //     if (!selected_candidates_index.includes(randomIndex) && answer_index != randomIndex) {
    //         selected_candidates_index.push(randomIndex)
    //     }
    // }
    // let selected_candidates = selected_candidates_index.map(index => answer_candidates[index])

    // //選択肢の中に正解を入れる
    // let correct_answer_index_in_cadidates = Math.floor(Math.random() * (selected_candidates.length + 1))
    // selected_candidates.splice(correct_answer_index_in_cadidates, 0, answer_entity)

    //選択肢を表示する(english_exam.json形式)
    let selected_candidates = question.answer_candidates;
    let correct_answer_index_in_cadidates = parseInt(question.answer_index) - 1;
    choicesElement.innerHTML = "";
    selected_candidates.forEach((choice, index) => {
        const button = document.createElement("button");
        button.textContent = choice;
        if (index == correct_answer_index_in_cadidates) {
            button.id = "correctOption"; // ボタンに独自のIDを与える
        }
        setCurrentQestionNumber(question.qid); //test.html内の関数を呼び出す
        button.addEventListener("click", () => checkAnswer(index, correct_answer_index_in_cadidates, question.qid));
        const choiceContainer = document.createElement("div");
        choiceContainer.classList.add("choice");
        choiceContainer.appendChild(button);
        choicesElement.appendChild(choiceContainer);
    });
}

// 回答をチェックする関数
async function checkAnswer(userAnswer, correctAnswer, qid) {

    setCurrentQestionNumber(null); //確信度チェック中の視線データを取得しないようにするため。
    
    await convertGazeDataToFeatureData(qid,correctAnswer); //視線特徴量抽出を行う
    await addQuestionsArea(qid); //問題文が表示されている領域を示す値を配列に追加する
    //　確信度をチェックする
    checkConfidence(userAnswer === correctAnswer, correctAnswer, qid)
}

// 確信度をチェックする
function checkConfidence(isCorrect, correctAnswerIndex, qid) {
    //　選択肢を表示
    const choicesElement = document.getElementById("choices");
    const questionElement = document.getElementById("question");
    const questionArea = document.getElementById("questionArea");
    questionArea.style.display = "none";
    //questionElement.textContent = "正解している自信はありますか？"
    choicesElement.innerHTML = "";

    // VASを表示する
    let vasElement = document.getElementById("vasElement");
    vasElement.style.display = "flex"
    // vas選択した位置を非表示にする
    let vas = document.querySelectorAll("#vasElement .container .vas")
    vas.forEach((element) => {
        element.style.display = "none";
    })
    //VASInitialize();
    //値の初期化
    //vasValue = -1
    //questionElement.style.display = "none"; //id=questionを非表示にする

    //仮、次の問題を出題する用のボタン
    //let nextQuestionButton = document.getElementById("nextQuestion");
    let nextQuestionButton = document.createElement("button");
    nextQuestionButton.textContent = "次に進む";
    nextQuestionButton.style.display = "none";
    nextQuestionButton.id = "nextQuestionButton"
    nextQuestionButton.addEventListener("click", () => {

        vasElement.style.display = "none"; //vasを非表示にする

        //回答データを記録する
        answerData.push([qid, correctAnswerIndex, vasValue[0], vasValue[1], isCorrect])
        vasValue[0],vasValue[1] = -1; //vasの値を初期化する
        concatAnswerDataToFeatureData(); //answerdataをfeaturedataにくっつける

        if(questions.length == 0 || countQuestion >= maxQuestionNum){
            //全ての問題を出題したor指定数問題を出題した場合
            if(practiceMode){
                //練習モードを終了後の関数を呼び出す
                showMessageBeforeExperiment();    
            }else{
                //終了のメッセージを表示し、特徴データファイルをダウンロードする
                showFinishMessage();
                downloadFeatureData(); //特徴量が書き込まれたファイルをダウンロードする
                downloadRawGazeData(); //特徴量に変換されていない視線データをダウンロードする
                downloadQuestionsAreas(); //問題文選択肢の範囲情報データをダウンロードする
                answerDataDownload(); //回答データをダウンロードする
            }
        } else {

            displayQuestion(); //次の問題を出題する
            //id=questionを表示する
            //const questionElement = document.getElementById("question");
            //questionElement.style.display = "block";
        }

        nextQuestionButton.parentNode.removeChild(nextQuestionButton); //自分自身を取り除く
    })
    vasElement.appendChild(nextQuestionButton);
}

function getAnswerData() {
    return answerData;
}


//VAS用
vasContainer.forEach(function(vasContainerElement,index){

    vasContainerElement.addEventListener("click", (e) => {
        const vas = vasContainerElement.querySelector(".vas")
        const rect = vasContainerElement.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const width = vasContainerElement.offsetWidth;
        /*
        元のコード
        const percentage = (clickX / width) * 100;
        */
        const dividedNum = 200; //何通りの値が欲しいか
        const lengthPerVasValue = width / dividedNum; //等分された一区間の長さ
        vasValue[index] = Math.floor(clickX / lengthPerVasValue); //VASの値
        console.log(clickX, width / lengthPerVasValue, vasValue[index]);

        if (vasValue[index] < 0) {
            vas.style.left = "0%";
            vasValue[index] = 0;
        } else if (vasValue[index] >= (dividedNum - 1)) {
            vas.style.left = width + "px";
            vasValue[index] = dividedNum - 1; //最大値
        } else {
            //位置を指定する縦線の位置とテキストを調整する
            vas.style.left = clickX + "px";
        }



        // vasの選択線画非表示ならば表示する
        let isShowLine = vas.style.display;
        console.log(isShowLine);
        if (isShowLine == "none") {
            vas.style.display = "block";
        }

        // 次の問題へ進むためのボタンを表示する
        if (vasValue.every(value => value >=0)){
        let nextQuestionButton = document.getElementById("nextQuestionButton");
        nextQuestionButton.style.display = "block";
        }
    });
})




