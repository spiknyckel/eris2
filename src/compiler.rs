use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::{
    parsing::{self, Ast, ItemKind}, semantical, tokenizing
};

pub struct Compiler {
    files: HashMap<PathBuf, String>,
    file_ids: Vec<PathBuf>,
}

impl Compiler {
    pub fn new() -> Compiler {
        Compiler {
            files: HashMap::new(),
            file_ids: Vec::new(),
        }
    }

    pub fn parse_file(&mut self, path: &Path) -> Result<Option<Ast>, String> {
        let file = std::fs::read_to_string(path);
        match file {
            Ok(file) => {
                let inserted = self.files.insert(path.to_path_buf(), file);
                if inserted.is_some() {
                    return Ok(None);
                }
                self.file_ids.push(path.to_path_buf());
            }
            Err(err) => return Err(err.to_string()),
        }
        let file_id = self.file_ids.len() - 1;
        let lexer = tokenizing::Lexer::new(self.files.get(path).unwrap(), file_id);
        let tokens = lexer.collect::<Vec<_>>();
        let parser = parsing::Parser::new(tokens.into_iter(), file_id);
        let mut ast = match parser.parse() {
            Ok(ast) => ast,
            Err(err) => {
                let file_id = err.file_id();
                let file = self.file_ids.get(file_id).unwrap();
                let file = self.files.get(file).unwrap();
                return Err(err.to_string(&file));
            }
        };
        let mut merge_ast = Ast { nodes: Vec::new() };
        for item in &ast.nodes {
            if let ItemKind::Import(p) = &item.kind {
                let current_dir = path.parent().unwrap();
                let new_path = current_dir.join(p);
                let new_ast = self.parse_file(&new_path)?;
                if let Some(new_ast) = new_ast {
                    merge_ast.nodes.extend(new_ast.nodes);
                }
            }
        }
        ast.nodes.extend(merge_ast.nodes);
        Ok(Some(ast))
    }

    pub fn build(&mut self, path: &Path) -> Option<PathBuf> {
        let ast = self.parse_file(path);
        let ast = match ast {
            Ok(ast) => ast,
            Err(err) => {
                eprintln!("{}", err);
                return None;
            }
        };
        if let Some(ast) = ast {
            let mut semantical = semantical::SemanticAnalyzer::new();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            let semantical_result = semantical.analyze_ast(ast, file_name);
            let new_ast = match semantical_result {
                Ok(ast) => ast,
                Err(err) => {
                    let file_id = err.file_id();
                    let file = self.file_ids.get(file_id).unwrap();
                    let file = self.files.get(file).unwrap();
                    eprintln!("{}", err.to_string(file));
                    return None;
                }
            };
            let llvm_context = inkwell::context::Context::create();
            let llvm_module = llvm_context.create_module("main");
            let mut codegen = crate::codegen::CodeGen::new(&llvm_context, llvm_module);
            codegen.generate(new_ast);
            codegen.create_binary(file_name)
        } else {
            None
        }
    }
}
