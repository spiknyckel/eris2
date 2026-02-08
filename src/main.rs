use std::{io::{self, Write}, process::Command};

use dexterws_compiler::compiler;

fn main() {
    let input = std::env::args().nth(1).expect("no file name given");
    let path = std::path::Path::new(&input);
    let mut compiler = compiler::Compiler::new();
    compiler.build(path);
}
