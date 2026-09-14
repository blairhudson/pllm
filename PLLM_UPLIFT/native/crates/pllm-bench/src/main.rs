use std::process::ExitCode;
fn run()->Result<String,String>{
 let args:Vec<String>=std::env::args().skip(1).collect();
 match args.first().map(String::as_str){
  Some("assure")=>{if args.len()!=1{return Err("assure takes no arguments".into())}pllm_assurance::json()},
  Some("matrix")=>{
   if args.len()!=6{return Err("usage: pllm-native-lab matrix DIM LANES REPETITIONS BITS TILE".into())}
   let num=|i:usize|args[i].parse::<usize>().map_err(|_|format!("invalid unsigned integer at argument {i}"));
   let bits=u8::try_from(num(4)?).map_err(|_|"bits exceeds u8".to_string())?;
   pllm_bench::benchmark(num(1)?,num(2)?,num(3)?,bits,num(5)?)
  },
  _=>Err("usage: pllm-native-lab assure | matrix DIM LANES REPETITIONS BITS TILE".into())
 }
}
fn main()->ExitCode {match run(){Ok(json)=>{println!("{json}");ExitCode::SUCCESS},Err(e)=>{eprintln!("{e}");ExitCode::FAILURE}}}
