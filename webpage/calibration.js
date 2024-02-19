
import { startPractice } from "./main.js";

const overlay = document.getElementById("overlay");

var PointCalibrate = 0;
var CalibrationPoints = {};

export function startCalibrate(){
    webgazer.showVideo(true); //webカメラの映像を表示する
    ClearCalibration()
    document.querySelectorAll(".Calibration").forEach((i) => {
       i.onclick = function() {
        countPointClick(i);
       };

    })
}

function countPointClick(pt) {
    const id = pt.id;

    if (!CalibrationPoints[id]){
        CalibrationPoints[id] = 0;
    }
    CalibrationPoints[id]++;

    if (CalibrationPoints[id] == 5){ 
        pt.style.setProperty("background-color", "yellow");
        pt.setAttribute("disabled", "disabled");
        PointCalibrate++;
    }else if (CalibrationPoints[id] < 5){
        var opacity = 0.2*CalibrationPoints[id] + 0.2;
        pt.style.setProperty("opacity", opacity);
    }

    if (PointCalibrate == 8){
        document.getElementById("Pt5").style.display = "inline";
    }

    if (PointCalibrate >= 9){ 
        overlay.style.setProperty("display","none");
        finishCalibration(); //キャリブレーションを終了する
    }
}

function ClearCalibration(){

    overlay.style.setProperty("display","flex");
    document.querySelectorAll(".Calibration").forEach((i) => {
      i.style.setProperty("background-color", "red");
      i.style.setProperty("opacity", "0.2");
      i.removeAttribute("disabled");
    });

    document.getElementById("Pt5").style.display = "none";
  
    CalibrationPoints = {};
    PointCalibrate = 0;
}

function finishCalibration(){
    //キャリブレーション終了時の処理

    webgazer.showVideo(false);
    //練習問題出題のコードを呼び出す
    startPractice();
}