fn main() {
    let input = std::env::args().nth(1).expect("no file name given");
    let path = std::path::Path::new(&input);
    let file = std::fs::read_to_string(path).expect("could not read file");
    let lexer = dexterws_compiler::tokenizing::Lexer::new(&file);
    let tokens = lexer.collect::<Vec<_>>();
    let parser = dexterws_compiler::parsing::Parser::new(tokens.into_iter());
    let ast = match parser.parse() {
        Ok(ast) => ast,
        Err(err) => {
            println!("{}", err.to_string(&file));
            return;
        }
    };
    let mut semantical = dexterws_compiler::semantical::SemanticAnalyzer::new();
    let file_name = path.file_name().unwrap().to_str().unwrap();
    let semantical_result = semantical.analyze_ast(ast, file_name);
    let new_ast = match semantical_result {
        Ok(ast) => ast,
        Err(err) => {
            println!("{}", err.to_string(&file));
            return;
        }
    };
    let llvm_context = inkwell::context::Context::create();
    let llvm_module = llvm_context.create_module("main");
    let mut codegen = dexterws_compiler::codegen::CodeGen::new(&llvm_context, llvm_module);
    codegen.generate(new_ast);
    //codegen.spit_out();
    let verification = codegen.verify();
    if let Err(err) = verification {
        println!("Verification failed: {}", err.to_string());
        return;
    }
    codegen.spit_out_object(file_name);
    println!("Compilation successful!");
}
