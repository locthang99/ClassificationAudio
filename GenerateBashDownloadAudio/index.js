const fs = require("fs");
const { throws } = require("assert");
const { exception } = require("console");
function writeData(path, data) {
  fs.appendFile(path, "\n" + data, function (err) {
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

const lineReader = require("line-reader");

async function rmDup(path, outPath,sum) {
  console.log(path);
  var checkList = new Map();
  var count = 0;
  await lineReader.eachLine(path, function (line) {
    var r = /Z[O|W][0-9|A-F|U|I|O|Z|W]{6}/g
    lineData = line.slice(14, 22)
    if (!checkList[lineData] && r.test(lineData) && count <= sum) {
      console.log(lineData)
      //console.log(obj)
      checkList[lineData] = true;
      writeData(outPath, line);
      count++;
    }
  });
}

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
]

var listAnother = [
  "trung-quoc-",
  "han-quoc-",
  "nhat-ban-",
  "thai-lan-",
  "phap-"
]

var listAU =[
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
]

var listTest = ["viet-nam-nhac-khong-loi-"]

const RemoveDupJSON = () => {
  const pathInput = "C:\\Users\\ldthang\\Desktop\\Area 51\\CloneByPython"
  for (let i = 1; i <= 7; i++) {
    fs.readdir(pathInput + "\\clonezingbypython" + i + "\\Data", (err, files) => {
      files.forEach(file => {
        //console.log(file);
        if (listAnother.includes(file))
          rmDup(pathInput + "\\clonezingbypython" + i + "\\Data\\" + file, "Output/"+file,2000)
      });
    });
  }
}

var listT = ["thai-lan-"]
const RemoveDupAfter = () =>{
  fs.readdir("Output", (err, files) => {
    files.forEach(file => {
      if (listT.includes(file))
        rmDup("Output/"+ file, "OutputFinal/"+file+".txt",10000)
    });
  });
}

//RemoveDupJSON()
RemoveDupAfter()

// rmDup("C:/Users/KhunGLonG/Desktop/clone/Cover/NotVN/thai-lan-","thai-lan-")

// fs.readdir("DataJSON", (err, files) => {
//   files.forEach((file) => {
//     var path = ("Download/"+file).replace(".txt","")
//     if (!fs.existsSync(path)) {
//       fs.mkdirSync(path);
//     }

//     // if (listTest.includes(file))
//     //   rmDup("Input" + "/clonezingbypython" + i + "/Data/" + file, file);
//   });
// });

// const request = require("request");
// const download = async (id,output) => {
//   try {
//     await request("http://api.mp3.zing.vn/api/streaming/audio/"+id+"/128").pipe(fs.createWriteStream(output));
//   } catch (err) {
//     console.log("toang");
//   }
// };

// function DownManyFromFile(i) {
//   var r = /Z[O|W][0-9|A-F|U|I|O|Z|W]{6}/g;
//   lineReader.eachLine("DataJSON/"+listTypes[i]+".txt", function (line) {
//     let rs1 = r.test(line.slice(14, 22));
//     let rs2 = r.test(line.slice(13, 21))
//         if (rs1==true) {
//           console.log(line.slice(14, 22) + " true14");
//           let id =line.slice(14, 22)
//           download(id,"Download/"+listTypes[i]+"/"+id+".mp3").then(()=>{console.log("done")})
//           return;
//         }
//         if (rs2==true) {
//           console.log(line.slice(13, 21) + " true13");
//           let id =line.slice(13, 21)
//           download(id,"Download/"+listTypes[i]+"/"+id+".mp3").then(()=>{console.log("done")})
//           return;
//         }

//   });
// }


// const fetch = require("node-fetch")
// fetch("https://604f32afc20143001744c8aa.mockapi.io/api/v1/config/cf").then(res=>res.json())
// .then(res=>{
//     if(!fs.existsSync("Download"))
//     {
//       fs.mkdirSync("Download");
//     }
//       if (!fs.existsSync("Download/"+listTypes[res.id])) {
//       fs.mkdirSync("Download/"+listTypes[res.id]);
//     }
//     DownManyFromFile(res.id);
// })
// //var r =/Z[O|W][0-9|A-F|U|I|O|Z|W]{6}/g

// // for(let i = 0;i<100;i++)
// // {
// //   download("ZWF09OZ7",i+".mp3")
// // }
