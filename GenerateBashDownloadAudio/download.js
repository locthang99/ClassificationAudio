const fs = require("fs");
const lineReader = require("line-reader");
const request = require("request");
// ---------------------------------------------------------------------------------
function writeData(path, data) {
  fs.appendFileSync(path, data, function (err) {
    if (err) writeErrorWrite(data);
  });
}

function writeErrorWrite(err) {
  fs.appendFile("logWrite.txt", "\n" + err, function (err) {
    if (err) throw err;
  });
}
function writeError(err) {
  fs.appendFile("log.txt", "\n" + err, function (err) {
    if (err) throw err;
  });
}
// ---------------------------------------------------------------------------------

var listTypes = [
  "-",
  "viet-nam-cai-luong-",
  "viet-nam-nhac-tre-v-pop-",
  "viet-nam-nhac-trinh-",
  "viet-nam-nhac-tru-tinh-",
  "viet-nam-rap-viet-",
  "viet-nam-nhac-thieu-nhi-",
  "viet-nam-nhac-cach-mang-",
  "viet-nam-nhac-dan-ca-que-huong-",
  "viet-nam-nhac-ton-giao-",
  "viet-nam-nhac-khong-loi-",
  "au-my-classical-",
  "au-my-folk-",
  "au-my-country-",
  "au-my-pop-",
  "au-my-rock-",
  "au-my-latin-",
  "au-my-rap-hip-hop-",
  "au-my-alternative-",
  "au-my-blues-jazz-",
  "au-my-reggae-",
  "au-my-r-b-soul-",
  "trung-quoc-",
  "han-quoc-",
  "nhat-ban-",
  "thai-lan-",
];

var listVN = [
  "viet-nam-cai-luong-",
  "viet-nam-nhac-tre-v-pop-",
  "viet-nam-nhac-trinh-",
  "viet-nam-nhac-tru-tinh-",
  "viet-nam-rap-viet-",
  "viet-nam-nhac-thieu-nhi-",
  "viet-nam-nhac-cach-mang-",
  "viet-nam-nhac-dan-ca-que-huong-",
  "viet-nam-nhac-ton-giao-",
  "viet-nam-nhac-khong-loi-",
];

var listAnother = [
  "trung-quoc-",
  "han-quoc-",
  "nhat-ban-",
  "thai-lan-",
  "phap-",
];

var listAU = [
  "au-my-classical-",
  "au-my-folk-",
  "au-my-country-",
  "au-my-pop-",
  "au-my-rock-",
  "au-my-latin-",
  "au-my-rap-hip-hop-",
  "au-my-alternative-",
  "au-my-blues-jazz-",
  "au-my-reggae-",
  "au-my-r-b-soul-",
];

// ---------------------------------------------------------------------------------


var count = 0;
var target = 0;
var step = 500
var idType = 10;
var output = "/content/gdrive/MyDrive/KLTN/DatasetSong/"

AddListId(10);

setTimeout(()=>{

  for(let i = 0; i<ListID.length;i++)
  {
    let lineBash = "curl -s -L api.mp3.zing.vn/api/streaming/audio/"+ListID[i]+"/128 -o "+ListID[i]+".mp3 & "
    if(i%500 == 0)
      lineBash = "! "+ lineBash
    if(i%500 ==499)
      lineBash = lineBash +"\n"
    writeData("BashLinux/10.sh",lineBash)
    //writeData("BashLinux/2.sh",i)
  }

},3000)

