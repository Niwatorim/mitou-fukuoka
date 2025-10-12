import {parse} from "@babel/parser";
import _traverse from "@babel/traverse";
const traverse = _traverse.default;
import {readFile,writeFile} from "fs/promises";
import path from "path";


//Unused but original code
async function parsefile() {
    try{
        const text = await readFile(filepath,"utf-8");
        const ast = parse(text,{
            sourceType:"module",
            plugins:["jsx"]
        });

        const attributes=[]

        // find all attribute functions
        traverse(ast,{
            JSXOpeningElement(path){
                //component
                const component_name = path.node.name.name
                //find parent
                const parentFunction = path.findParent((p)=>p.isFunctionDeclaration())
                if (!parentFunction){
                    return
                }
                
                let function_name="None";
                function_name=parentFunction.node.id.name;

                const attrInfo={
                    name:component_name,
                    parentFunction:function_name,
                    attributes:{}
                }

                //find attributes
                path.traverse({
                    JSXAttribute(attrpath){

                        const attrName = attrpath.node.name.name
                        let attributeValueType =" None "
                        
                        attrpath.traverse({
                            ArrowFunctionExpression(funcPath) {
                                attributeValueType = "ArrowFunctionExpression";
                                funcPath.stop(); // Stop this inner search, we found it
                            
                            }
                        });

                        attrInfo.attributes[attrName]=attributeValueType
                    }
                });

                attributes.push(attrInfo);
            }
        });

        console.log("-----Analysis:")
        console.log(JSON.stringify(attributes,null,2))
    }catch(e){
        console.warn("error with ",e);
    }
}

/*

Find functions: and all their attributes inside
All arrow functions or function calls inside attributes

Function:
find attributes(attribute):
    {Functions: parents, children
    Inside attribute functions: arrow functions and stuff
    }

*/

async function parseFileNew(filepath, options={}){
    try{
        const text = await readFile(filepath,"utf-8") // reads file

        const extension = path.extname(filepath); // to get the file extension
        const plugins = ["jsx"]

        //in case file is typescript
        if (extension== ".tsx"||extension==".ts"){
            plugins.push("typescript")
        }

        //make ast
        const ast = parse(text,
            {
                sourceType:"module",
                plugins:plugins
            }
        );

        //go down the tree
        const components=[]; //captures each component
        traverse(ast,{
            JSXOpeningElement(path){ //find every react component
                const component = extractComponentInfo(path,text) //get info on the attributes
                if (component){
                    components.push(component)
                }
            }}

        )

        const result={
            file:filepath,
            components:components,
            summary:`Attributes from the code of ${filepath}`
        }

        console.log(JSON.stringify(result,null,2));
    } catch(e){
        console.error("error in reading file", e.message);
    }
}

function getComponentName(node){
    if (node.type === 'JSXIdentifier') {
        return node.name;
    }
    if (node.type === 'JSXMemberExpression') {
        return `${node.object.name}.${node.property.name}`;
    }
    if (node.type === 'JSXNamespacedName') {
        return `${node.namespace.name}:${node.name.name}`;
    }
    return 'Unknown';
}

function getFunctionName(parentFunctionPath) {
    const parentNode = parentFunctionPath.node;
    // For function App() {}
    if (parentNode.id?.name) {
        return parentNode.id.name;
    }
    // For const App = () => {}
    if (parentFunctionPath.parent?.type === 'VariableDeclarator') {
        return parentFunctionPath.parent.id.name;
    }
    // For myMethod() {} in a class
    if (parentNode.key?.name) {
        return parentNode.key.name;
    }
    return 'Anonymous';
}

function findText(path){
    let text = "";
    path.traverse({
        JSXText(new_path){
            const textValue = new_path.node.value.trim();
            if (textValue){
                text += textValue + " ";
            }
        }
    });
    return text.trim();
}

function divChildren(path, text) {
    const children = [];
    const jsxElement = path.parentPath;
    
    if (jsxElement.node.children) {
        jsxElement.node.children.forEach((child) => {
            if (child.type === 'JSXElement') {
                const childName = getComponentName(child.openingElement.name);
                const testableAttrs = [];
                
                child.openingElement.attributes.forEach(attr => {
                    if (attr.type === "JSXAttribute") {
                        const attrName = attr.name.name;
                        const attrValue = extractAttributeValue(attr, text);
                        
                        if (isTestable(attrName, attrValue)) {
                            testableAttrs.push({
                                name: attrName,
                                value: attrValue
                            });
                        }
                    }
                });
                
                if (testableAttrs.length > 0) {
                    children.push({
                        name: childName,
                        testableAttributes: testableAttrs
                    });
                }
            }
        });
    }
    
    return children;
}

function extractComponentInfo(path,text){
    const node = path.node //current node
    const name = getComponentName(node.name); //current nodes name
    
    //get any parents
    const parent = path.findParent((p)=> 
        p.isFunctionDeclaration() ||
        p.isArrowFunctionExpression() ||
        p.isClassMethod()
    );
    if (!parent){
        return null; //has no components or functions
    }

    //parent functions
    const functionName = getFunctionName(parent)

    const info={
        name: name,
        parentFunction: functionName,
        testableAttributes:[],
        text: "",
        attributes:{},
        selectors:[]
    }

    if (name == "p" || name == "h1" || name == "h2" || name == "h3"){
        //find text
        info.text = findText(path.parentPath)
    }

    if (name == "div"){
        info.attributes["contains"] = divChildren(path,text)
    }

    //for every attribute
    node.attributes.forEach( attr => {
        if (attr.type === "JSXAttribute"){
            const attrName = attr.name.name;

            const attrValue = extractAttributeValue(attr,text); //if its a JSX attribute then we have the content from there
            info.attributes[attrName] = attrValue; 
            if (isTestable(attrName,attrValue)){
                info.testableAttributes.push(
                    {
                        name:attrName,
                        value:attrValue,
                        strategy:testStrategy(attrName,attrValue)
                    }    
                )
            }
            if (attrName === "id" && attrValue.type === "StringLiteral"){
                info.selectors.push(`#${attrValue.value}`);
            }
            if (attrName === "className" && attrValue.type === "StringLiteral"){
                const classes = attrValue.value.split(" ").filter(Boolean);
                info.selectors.push(`.${classes.join(".")}`);
            }
        }

    });
    return info;

}

function extractAttributeValue(attr,text){
    if (!attr.value){
        return {
            type: "BooleanLiteral",
            value: true
        }
    }
    if(attr.value.type == "StringLiteral"){
        return {
            type: "StringLiteral",
            value: attr.value.value
        }
    }
    if (attr.value.type == "JSXExpressionContainer"){
        const expr = attr.value.expression
        if (expr.type == "ArrowFunctionExpression"){
            return {
                type: "Function",
                functionType: expr.type,
                param: expr.params
            }
        }
        if (expr.type === "Identifier"){
            return{
                type:"Identifier",
                name: expr.name
            }
        }
        return {
            type: expr.type,
            code: text.slice(expr.start,expr.end)
        }
    }
    return {type: "Unknown"};
}

function isTestable(name,value){
    // Event Handlerrs
    if (name.startsWith("on")) return true;

    // Selector attributes
    if (name.startsWith("data-") || name === "id" || name === "className") return true;

    const interactive = ["disabled","checked","value","href"];
    if (interactive.includes(name)) return true;
    return false;
}

function testStrategy(name,value){
        // --- Event Handlers ---
        if (name.startsWith("onClick") || name.startsWith("onSubmit") || name.startsWith("onMouseDown") || name.startsWith("onMouseUp")) {
            return {
                action: "click",
                expectation: "handler_triggered",
                description: `Click element to trigger ${name}`
            };
        }
    
        if (name.startsWith("onChange") || name.startsWith("onInput")) {
            return {
                action: "input",
                expectation: "value_changed",
                description: `Type into the element to trigger ${name}`
            };
        }
    
        if (name.startsWith("onFocus") || name.startsWith("onBlur")) {
            return {
                action: "focus/blur",
                expectation: "state_changed",
                description: `Focus or blur the element to trigger ${name}`
            };
        }
    
        if (name.startsWith("onMouseOver") || name.startsWith("onMouseEnter") || name.startsWith("onMouseLeave")) {
            return {
                action: "hover",
                expectation: "handler_triggered",
                description: `Hover over the element to trigger ${name}`
            };
        }
        
        // Generic 'on' handler for other events
        if (name.startsWith('on')) {
            const event = name.slice(2).toLowerCase();
            return {
                action: event,
                expectation: 'handler_triggered',
                description: `Trigger ${event} event on element`
            };
        }
    
        // --- State & Accessibility ---
        if (name === "disabled" || name === "checked" || name === "selected" || name === "readOnly") {
            return {
                action: "verify",
                expectation: "state_matches",
                description: `Verify ${name} state is correct`
            };
        }
    
        if (name === "aria-label" || name === "aria-labelledby" || name === "aria-describedby") {
            return {
                action: "verify",
                expectation: "accessibility_property_present",
                description: `Verify accessibility property "${name}" exists`
            };
        }
    
        // --- Content & Links ---
        if (name === "href") {
            return {
                action: "click",
                expectation: "navigation",
                description: "Click link and verify navigation to its destination"
            };
        }
    
        if (name === "src" || name === "alt") {
            return {
                action: "verify",
                expectation: "content_matches",
                description: "Verify image source or alt text is correct"
            };
        }
    
        if (name === "value" || name === "placeholder" || name === "title") {
            return {
                action: "verify",
                expectation: "text_present",
                description: `Verify "${name}" content is correct`
            };
        }
        
        // --- Selectors ---
        if (name === "id" || name === "className" || name.startsWith("data-")) {
            return {
                action: "inspect",
                expectation: "selector_available",
                description: "Use this attribute to locate the element"
            };
        }
        
        // A default fallback
        return {
            action: 'inspect',
            expectation: 'attribute_present',
            description: `Verify ${name} attribute exists`
        };
}

const filepath = process.argv[2]; 

if (!filepath) {
    console.error("Error: Please provide a filepath to analyze.");
    process.exit(1); // Exit with an error
}

// Run the main analysis function with the filepath
parseFileNew(path.resolve(filepath));
