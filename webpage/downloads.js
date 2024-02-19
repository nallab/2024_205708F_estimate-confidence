export function convertToCSV(dataArray) {
  const csvRows = [];

  // 配列の各要素をCSV行に変換
  dataArray.forEach((row) => {
    const csvRow = row.join(","); // カンマで要素を結合
    csvRows.push(csvRow);
  });

  // CSV行を改行で結合してCSVデータを作成
  const csvData = csvRows.join("\n");

  return csvData;
}

export function downloadCSV(csvData, filename) {
  const blob = new Blob([csvData], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.setAttribute("href", url);
  const currentDate = new Date();
  const options = { timeZone: 'Asia/Tokyo', timeZoneName: 'short' };
  const formattedDate = currentDate.toLocaleString('ja-JP', options).slice(0, 20).replace(/[\/:\ ,]/g, "");
  //const formattedDate = currentDate.toISOString().slice(0, 19).replace(/[-T:]/g, "");
  filename = filename + "_" + formattedDate + ".csv";
  link.setAttribute("download", filename);
  link.style.visibility = "hidden";

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function getGazeData() {
  // CSVデータの作成
  const csvData = convertToCSV(gazeData);

  // CSVファイルのダウンロード
  const filename = "gaze_data";
  downloadCSV(csvData, filename);
}

//回答、確信度のデータのダウンロード
function downloadAnswerData(answerData) {
  const csvData = convertToCSV(answerData);
  const filename = "answer_data";
  downloadCSV(csvData, filename);
}