
/*
1. 視線キャリブレーション
2. 練習問題出題
3. 問題出題
4. 確信度解答
(3,4をn回繰り返す)
*/

import {startCalibrate} from "./calibration.js";
import { displayQuestion,loadQuestion} from "./quiz-and-confidence.js";
import { convertRawDataToFeature } from "./feature-extractor.js";
import { convertToCSV,downloadCSV } from "./downloads.js";

export let gazeData = []; // その問題の視線情報を格納する配列
//下二つは後から処理をする際に役に立ちそうなもの
let rawGazeDatum = []; //複数の問題の視線情報を格納する配列
let questionsAreas = [];//複数の問題(問題文、選択肢が表示されている範囲を左上から左、右、上、下の距離で示す)
export let answerData = []; // 回答、自信度情報を格納する配列
export const featureData = []; //視線情報などから計算した特徴量を格納する配列
export let currentQuestionNumber = null; //現在出題している問題番号
let messageDiv = document.getElementById("message"); //メッセージ
let nextButton = document.getElementById("nextButton"); //次へ進むボタン
let instructionMessageDiv = document.getElementById("instructionMessage"); //メッセージと次へ進むボタンの親

export function setCurrentQestionNumber(num) {
    currentQuestionNumber = num
}

// function getCurrentQuestionNumber() {
//     return currentQuestionNumber
// }

// function getRawGazeData() {
//     return gazeData
// }

window.onload = function () {

    /* webgazerを読み込む */
    
    webgazer.saveDataAcrossSessions(false);
    webgazer.applyKalmanFilter(true);
    webgazer.showVideo(false);
    webgazer.showPredictionPoints(true);
    let webgazerLoadPromise = webgazer.setGazeListener(function (data, elapsedTime) {
            //let currentQuestionNumber = getCurrentQuestionNumber()
            if (data != null && currentQuestionNumber != null) {
              //var xprediction = data.x;
              //var yprediction = data.y;
              //console.log(data.x,data.y)
              gazeData.push([data.x, data.y, currentQuestionNumber]);
            }
            
    }).begin();
    webgazerLoadPromise.then( () => {
         //ローディングを非表示にする
        let loaderDiv = document.getElementsByClassName("loader");
        loaderDiv[0].style.visibility = "hidden";
        console.log("hello");
        setFirstMessage();
    }
    );

}

/*キャリブレーションを行う*/
function setFirstMessage(){
    messageDiv.style.display = "block";
    nextButton.style.display = "block";
    messageDiv.innerHTML = `WebGazerのキャリブレーションを行います<br>
    画面に表示される赤いボタンを見つめながらボタンの色が赤から黄色に変わるまで複数回クリックしてください<br>
    下に表示されているボタンをクリックするとキャリブレーションを開始します`; //メッセージを表示する
}


let calibrationStart = function () {
    instructionMessageDiv.style.display = "none";
    // messageDiv.textContent = "";
    // nextButton.style.display = "none";
    startCalibrate();
    //webgazer.showVideo(true);
};

nextButton.addEventListener("click", calibrationStart);

/*練習問題出題*/

export function startPractice(){
    //calibration終了後に呼び出される
    //視線予想の点を非表示にする
    webgazer.showPredictionPoints(false)
    instructionMessageDiv.style.display = "flex";
    //nextButton.style.display = "block";
    nextButton.removeEventListener("click",calibrationStart); //イベントリスナーを取り外す
    messageDiv.textContent = "これから練習問題を1問出題します";
    nextButton.addEventListener("click",showPracticeQuestion);
}

function showPracticeQuestion(){

    //問題を出題する(引数でどの問題出すか決められるといいかも？練習問題本番など。何問出題するかとか決められうといいかも)


    //メッセージ、次へ進むボタンを非表示にする
    instructionMessageDiv.style.display = "none";
    
    // messageDiv.textContent = "";
    // nextButton.style.display = "none";
    
    //練習問題を読み出題する(未実装)
    loadQuestion("practice_questions.json",true);
}

export function showMessageBeforeExperiment(){
    //本番問題を出題する前にメッセージを表示する
    nextButton.removeEventListener("click",showPracticeQuestion);
    instructionMessageDiv.style.display = "flex";
    messageDiv.textContent = "これから本番問題を50問出題します";
    nextButton.addEventListener("click",showQuestion);
}

function showQuestion(){
    //本番問題を出題する
    nextButton.removeEventListener("click",showQuestion);
    instructionMessageDiv.style.display = "none";
    loadQuestion("english_exam.json",false);
}

export function showFinishMessage(){
    instructionMessageDiv.style.display = "flex";
    messageDiv.textContent = "終了です";
    nextButton.style.display = "none";
    console.log(answerData)
}

export async function convertGazeDataToFeatureData(qId,correctAnswerIndex){
    //視線データを特徴データに変換する
    convertRawDataToFeature(qId,gazeData,correctAnswerIndex);
    rawGazeDatum = rawGazeDatum.concat(gazeData); //複数問題の視線情報を溜めておく
    gazeData = []; //変換したgazeDataを削除する
    
}

export async function addQuestionsArea(qid){
    //問題文、選択肢の表示領域を示す値を配列に追加する
    //問題文の表示領域に関する値を取り出す
    let questionRect = document.getElementById("question").getBoundingClientRect();
    let questionArea = [questionRect.left,questionRect.right,questionRect.top,questionRect.bottom]

    //選択肢の表示領域に関する値を取り出す
    let buttonContainers = document.querySelectorAll(".choice");
    let buttonRects = [];
    let choicesArea = [] ;
    buttonContainers.forEach(container => {
        let button = container.querySelector("button");
        buttonRects.push(button.getBoundingClientRect());
    });
    buttonRects.forEach(buttonRect => {
        //choicesArea.push([buttonRect.left,buttonRect.right,buttonRect.top,buttonRect.bottom]);
        questionArea = questionArea.concat([buttonRect.left,buttonRect.right,buttonRect.top,buttonRect.bottom]);
    });

    //問題idを追加
    questionArea.unshift(qid)

    // console.log("questionAreaだよ〜");
    // console.log(questionArea);
    questionsAreas.push(questionArea);
}

export function concatAnswerDataToFeatureData(){
    //回答データを特徴データにくっつける
    let lastFeatureData = featureData.pop(); //最後に追加されたfeatureDataを取り出す
    lastFeatureData = lastFeatureData.concat(answerData[answerData.length-1].slice(1)); //answerDataをくっつける(qidを取り除くためのslice)
    featureData.push(lastFeatureData); //featureDataに戻す
    //answerData = []; //くっつけたanswerDataを削除する
    console.log(featureData)
}

export function downloadFeatureData(){
    //特徴データをダウンロードする
    let featureDataCSV = convertToCSV(featureData);
    downloadCSV(featureDataCSV,"featureData");
}

export function downloadRawGazeData(){
    //特徴量に変換されていない視線情報を取得する
    let rawGazeDataCSV = convertToCSV(rawGazeDatum);
    downloadCSV(rawGazeDataCSV,"rawGazeData");
}

export function downloadQuestionsAreas(){
    //問題文、選択肢などの範囲情報をダウンロードする
    let questionsAreasCSV = convertToCSV(questionsAreas);
    downloadCSV(questionsAreasCSV,"questionsAreas");
}

export function answerDataDownload(){
    /*
    answerDataをダウンロードする。
    一応特徴量データの後ろにくっつけているけど
    別でもダウンロードしていた方が生データを
    利用することになった際に便利だと思うので
    */
   let answerDataCSV = convertToCSV(answerData);
   downloadCSV(answerDataCSV,"answerData");
}