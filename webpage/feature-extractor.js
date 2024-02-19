import { featureData } from "./main.js";

let rawGazeData 
let qidAndIndexOfCorrectChoice
// 生の視線データと正解選択肢の位置をセットする。あまり良くない方法だと思うが。正攻法がわからん。
export function setRawGazeDataAndAnswerData(rgd,ansd){
    rawGazeData = rgd;
    qidAndIndexOfCorrectChoice = new Map(ansd.map(column => column.slice(0,2)));
    console.log(qidAndIndexOfCorrectChoice)
}

// 視線サンプルの数
function getNumberOfGazeSamples(gazeDatum){
    return gazeDatum.length
}

// 視線データのばらつき（分散）を求める。x,yそれぞれで求める。平均までの距離の平方根の平均値。標準偏差かな？
//gazedatumは[x0,y0],[x1,y1]って言う感じだと思うので多分gazedatum[0]としたらダメかも！！
function getDispersionOfGazePoints(gazeDatum){
    let sumX = 0
    let sumY = 0
    for(let gazePointPosition of gazeDatum){
        sumX += gazePointPosition[0];
        sumY += gazePointPosition[1];
    }
    const meanX = sumX/gazeDatum.length
    const meanY = sumY/gazeDatum.length
    let squaredDifferenceX = 0
    let squaredDifferenceY = 0
    for(let gazePointPosition of gazeDatum){
        squaredDifferenceX += Math.pow(gazePointPosition[0] - meanX, 2);
        squaredDifferenceY += Math.pow(gazePointPosition[1] - meanY, 2);
    }
    const RMSX = Math.sqrt(squaredDifferenceX/gazeDatum.length)
    const RMSY = Math.sqrt(squaredDifferenceY/gazeDatum.length)

    return [RMSX,RMSY]
}

// 選択肢の範囲内にあった視線データの数
function NumberOfGazePointsInOption(gazeDatum,correctChoiceIndex){
    // 戻り値:選択肢1,選択肢2,選択肢3,選択肢4,選択肢上の合計,正解上、不正解上
    
    let buttonContainers = document.querySelectorAll(".choice");
    let buttonRects = []
    buttonContainers.forEach(container => {
        let button = container.querySelector("button");
        buttonRects.push(button.getBoundingClientRect());
    });
    //const buttons = Array.from(buttonContainer.querySelectorAll("button"))
    //let buttonRects = new Array(buttons.length);
    /*
    for (let i = 0;i < buttons.length;i++){
        buttonRects[i] = buttons[i].getBoundingClientRect();
    }
    */
    let countOfPointsInOptions = new Array(buttonRects.length).fill(0);
    for (let gazePointPosition of gazeDatum){
        for (let i = 0;i < buttonRects.length;i++){
            let pointX = gazePointPosition[0]
            let pointY = gazePointPosition[1]

            var isInsideButton = (
            pointX >= buttonRects[i].left &&
            pointX <= buttonRects[i].right &&
            pointY >= buttonRects[i].top &&
            pointY <= buttonRects[i].bottom
            );
            
            if (isInsideButton){
                countOfPointsInOptions[i] += 1
            }
        }
    }
    
    //選択肢上の合計
    let sum = 0;
    for (let i = 0;i < buttonContainers.length;i++){
        sum += countOfPointsInOptions[i];
    }
    countOfPointsInOptions.push(sum);

    //正解のボタン上の視線の数を探す
    /*
    let correctButtonIndex = null
    buttonContainers.forEach((container,index) => {
        let button = container.querySelector("button");
        if (button.id == "correctOption"){
            correctButtonIndex = index
        }
    });
    */
    countOfPointsInOptions.push(countOfPointsInOptions[correctChoiceIndex])

    //不正解上の視線の数
    let missOption = countOfPointsInOptions[4] - countOfPointsInOptions[5];
    countOfPointsInOptions.push(missOption)

    return countOfPointsInOptions
}

//問題文上の視線情報を取る
function numberOfGazePointsInQuestion(gazeDatum){
    let countOfPointsInQuestion = 0
    for (let gazePointPosition of gazeDatum){
        let questionRect = document.getElementById("question").getBoundingClientRect();
        let pointX = gazePointPosition[0]
        let pointY = gazePointPosition[1]

        var isInsideQuestion = (
        pointX >= questionRect.left &&
        pointX <= questionRect.right &&
        pointY >= questionRect.top &&
        pointY <= questionRect.bottom
        );

        if (isInsideQuestion){
            countOfPointsInQuestion += 1
        }
    }
    return countOfPointsInQuestion
}

// 生の視線データを特徴量に変換する
export function convertRawDataToFeature(qId,gazeData,answerIndex){

    // // ユニークな問題idを取り出す
    // let rawGazeDataDict = new Map()
    // let questionIdIndex = 2
    // for (let gazePoint of rawGazeData){
    //     let qId = gazePoint[questionIdIndex];
    //     if(!rawGazeDataDict.has(qId)){
    //         rawGazeDataDict.set(qId,[gazePoint])
    //         //uniqueQuestionId.push(qId)
    //     }
    //     else{
    //         rawGazeDataDict.get(qId).push(gazePoint)
    //     }
    // }

    // // qId,視線データの数,視線の標準偏差X座標,視線の標準偏差Y座標,選択肢1,選択肢2,選択肢3,選択肢4,選択肢上の合計,正解上,不正解上,問題文上の視線
    // let featuredDatas = []
    // for (let [qId,gazeDatum] of rawGazeDataDict){
    //     if (qidAndIndexOfCorrectChoice.has(qId)){
    //     //gazeDatumはある問題の時の視線のデータの配列[x0,y0],[x1,y1]...みたいな感じだよね
    //     //console.log(gazeDatum)
    //     let correctChoiceIndex = qidAndIndexOfCorrectChoice.get(qId) //正解選択肢が何番目に表示されていたか
    //     let tmpDatas = []
    //     tmpDatas.push(qId)
    //     tmpDatas.push(getNumberOfGazeSamples(gazeDatum)) // 視線サンプルの数を求める
    //     tmpDatas = tmpDatas.concat(getDispersionOfGazePoints(gazeDatum))//標準偏差を求める
    //     tmpDatas = tmpDatas.concat(NumberOfGazePointsInOption(gazeDatum,correctChoiceIndex))// 選択肢上の視線の数
    //     //console.log(NumberOfGazePointsInOption(gazeDatum))
    //     tmpDatas.push(numberOfGazePointsInQuestion(gazeDatum))
    //     featuredDatas.push(tmpDatas)
    //     }else{
    //         continue
    //     }
    // }

    //変更中
    let tmpDatas = []
    tmpDatas.push(qId)
    tmpDatas.push(getNumberOfGazeSamples(gazeData)) // 視線サンプルの数を求める
    tmpDatas = tmpDatas.concat(getDispersionOfGazePoints(gazeData))//標準偏差を求める
    tmpDatas = tmpDatas.concat(NumberOfGazePointsInOption(gazeData,answerIndex))// 選択肢上の視線の数
    //console.log(NumberOfGazePointsInOption(gazeDatum))
    tmpDatas.push(numberOfGazePointsInQuestion(gazeData))//問題文上の視線の数
    featureData.push(tmpDatas)

    //return featuredDatas
}
