import {parse} from "@babel/parser";

async function help_boss(text){
    let ast = "None"
    try{
        ast = parse(text,{
            sourceType:"module",
            plugins:["jsx"]
        });
    }catch(e){
        ast = {error:`Error of: ${e}`}
    }
    ast = JSON.stringify(ast,null,2)
    return ast
}

function readinputandprocess(){
    let data ="";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data",(chunk)=> data+=chunk)
    process.stdin.on("end",()=>{
        help_boss(data).then(result=>console.log(result))
    })
}

readinputandprocess();
