const { parse } = require("@babel/parser");

async function help_boss(text) {
    let ast = "None";
    try {
        // CRITICAL FIX: Normalize line endings to \n before parsing
        // This ensures AST indices match the source code regardless of platform
        const normalizedText = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        
        ast = parse(normalizedText, {
            sourceType: "module",
            plugins: ["jsx"]
        });
    } catch (e) {
        ast = { error: `Error of: ${e}` };
    }
    ast = JSON.stringify(ast, null, 2);
    return ast;
}

function readinputandprocess() {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => data += chunk);
    process.stdin.on("end", () => {
        help_boss(data).then(result => console.log(result));
    });
}

readinputandprocess();