import { validate } from './content.mjs';
const result=validate();
console.log(JSON.stringify(result,null,2));
if(result.errors.length) process.exitCode=1;
