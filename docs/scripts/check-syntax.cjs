const fs=require('node:fs'),path=require('node:path');
// Optional override is for an offline verification environment, not a project dependency.
const ts=require(process.env.TYPESCRIPT_MODULE||'typescript');
function walk(p){return fs.readdirSync(p,{withFileTypes:true}).flatMap(e=>e.isDirectory()&&!['node_modules','.next','.source'].includes(e.name)?walk(path.join(p,e.name)):e.isFile()?[path.join(p,e.name)]:[])}
const files=walk(process.cwd()).filter(p=>/\.tsx?$/.test(p)&&!p.endsWith('.d.ts'));const diagnostics=[];
for(const file of files){const result=ts.transpileModule(fs.readFileSync(file,'utf8'),{fileName:file,reportDiagnostics:true,compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext,jsx:ts.JsxEmit.ReactJSX,isolatedModules:true}});for(const d of result.diagnostics||[])if(d.category===ts.DiagnosticCategory.Error)diagnostics.push({file,message:ts.flattenDiagnosticMessageText(d.messageText,'\n')});}
console.log(JSON.stringify({scope:'Syntax and transpilation only, not dependency resolution or TypeScript semantic checking',typescript:ts.version,files:files.length,diagnostics},null,2));process.exitCode=diagnostics.length?1:0;
