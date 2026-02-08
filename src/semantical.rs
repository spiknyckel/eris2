use std::collections::HashMap;

use crate::{
    chainmap::ChainMap,
    parsing::{
        self, Arg, Args, Assignment, Ast, BinOpKind, Block, CallArgs, Expr, ExprKind, Function, FunctionSignature, Identifier, ItemKind, LiteralKind, NodeLocation, Statement, StatementKind, Struct, Type, TypeKind, UnaryOpKind
    },
    tokenizing::TokenLocation,
};

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: Identifier,
    pub ty: TypeKind,
}

#[derive(Debug, Clone)]
pub struct ExtendedFunction {
    pub inner: Function,
    pub variables: Vec<Variable>,
}

#[derive(Debug, Clone)]
pub struct ExtendedStruct {
    pub inner: Struct,
    pub fields: HashMap<String, (TypeKind, usize)>,
}

#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub functions: Vec<ExtendedFunction>,
    pub structs: HashMap<String, ExtendedStruct>,
}

pub struct SemanticAnalyzer {
    variables: ChainMap<String, TypeKind>,
    function_variables: Vec<Variable>,
    functions: HashMap<String, FunctionSignature>,
    structs: HashMap<String, Struct>,
    loop_depth: usize,
    current_function: Option<String>,
}

//const BUILTINS: &[(&str, Type, Type)] = &[
//    ("iprint", Type::Int, Type::Int),
//    ("fprint", Type::Float, Type::Int),
//    ("bprint", Type::Bool, Type::Int),
//    ("printf", Type::String, Type::Int),
//];

#[derive(Debug, Clone)]
pub enum SemanticError {
    ExpressionTypeMismatch {
        expected: TypeKind,
        found: TypeKind,
        expr: Expr,
    },
    FunctionNotFound {
        ident: Identifier,
    },
    ArgumentCountMismatch {
        expected: usize,
        found: usize,
        args: CallArgs,
    },
    ArgumentTypeMismatch {
        expected: TypeKind,
        found: TypeKind,
        arg: Expr,
    },
    BreakOutsideOfLoop {
        statement: Statement,
    },
    DoubleVariableDeclaration {
        ident: Identifier,
    },
    DoubleFunctionDeclaration {
        ident: Identifier,
    },
    DoubleStructDeclaration {
        ident: Identifier,
    },
    VariableNotFound {
        ident: Identifier,
    },
    ReturnOutsideOfFunction {
        stmt: Statement,
    },
    ReturnTypeMismatch {
        expected: TypeKind,
        found: TypeKind,
        statement: Statement,
    },
    UnsupportedBinOp {
        ty: TypeKind,
        expr: Expr,
    },
    UnreachableCode {
        statement: Statement,
    },
}

impl SemanticError {
    pub fn to_string(self, source: &str) -> String {
        let span = match &self {
            SemanticError::ExpressionTypeMismatch { expr, .. } => expr.span,
            SemanticError::FunctionNotFound { ident, .. } => ident.span,
            SemanticError::ArgumentCountMismatch { args, .. } => args.span,
            SemanticError::ArgumentTypeMismatch { arg, .. } => arg.span,
            SemanticError::BreakOutsideOfLoop { statement } => statement.span,
            SemanticError::DoubleVariableDeclaration { ident, .. } => ident.span,
            SemanticError::DoubleFunctionDeclaration { ident } => ident.span,
            SemanticError::DoubleStructDeclaration { ident } => ident.span,
            SemanticError::VariableNotFound { ident } => ident.span,
            SemanticError::ReturnOutsideOfFunction { stmt } => stmt.span,
            SemanticError::ReturnTypeMismatch { statement, .. } => statement.span,
            SemanticError::UnsupportedBinOp { expr, .. } => expr.span,
            SemanticError::UnreachableCode { statement } => statement.span,
        };
        dbg!(&span);

        let error_text = match &self {
            SemanticError::ExpressionTypeMismatch {
                expected, found, ..
            } => {
                format!(
                    "invalid expression type, expected {}, found {}",
                    expected, found
                )
            }
            SemanticError::FunctionNotFound { ident, .. } => {
                format!("function '{}' is not defined", ident.value)
            }
            SemanticError::ArgumentCountMismatch {
                expected, found, ..
            } => {
                format!(
                    "invalid argument count, expected {} arguments, found {}",
                    expected, found
                )
            }
            SemanticError::ArgumentTypeMismatch {
                expected, found, ..
            } => {
                format!(
                    "invalid argument type, expected {}, found {}",
                    expected, found
                )
            }
            SemanticError::BreakOutsideOfLoop { .. } => "cannot break outside of loop".to_string(),
            SemanticError::DoubleVariableDeclaration { ident } => {
                format!("variable '{}' already declared", ident.value)
            }
            SemanticError::DoubleFunctionDeclaration { ident } => {
                format!("function '{}' already declared", ident.value)
            }
            SemanticError::DoubleStructDeclaration { ident } => {
                format!("struct '{}' already declared", ident.value)
            }
            SemanticError::VariableNotFound { ident } => {
                format!("variable '{}' is not defined", ident.value)
            }
            SemanticError::ReturnOutsideOfFunction { .. } => {
                "cannot return outside of function".to_string()
            }
            SemanticError::ReturnTypeMismatch {
                expected, found, ..
            } => {
                format!(
                    "invalid return type, expected {}, found {}",
                    expected, found
                )
            }
            SemanticError::UnsupportedBinOp { ty, .. } => {
                format!("unsupported binary operation for type {}", ty)
            }
            SemanticError::UnreachableCode { .. } => "this code is unreachable".to_string(),
        };
        let error = crate::error::error(span.line_span(), span.col_span(), source);
        format!("{}\n{}", error_text, error)
    }

    pub fn file_id(&self) -> usize {
        match self {
            SemanticError::ExpressionTypeMismatch { expr, .. } => expr.span,
            SemanticError::FunctionNotFound { ident } => ident.span,
            SemanticError::ArgumentCountMismatch { args, .. } => args.span,
            SemanticError::ArgumentTypeMismatch { arg, .. } => arg.span,
            SemanticError::BreakOutsideOfLoop { statement } => statement.span,
            SemanticError::DoubleVariableDeclaration { ident } => ident.span,
            SemanticError::DoubleFunctionDeclaration { ident } => ident.span,
            SemanticError::DoubleStructDeclaration { ident } => ident.span,
            SemanticError::VariableNotFound { ident } => ident.span,
            SemanticError::ReturnOutsideOfFunction { stmt } => stmt.span,
            SemanticError::ReturnTypeMismatch { statement, .. } => statement.span,
            SemanticError::UnsupportedBinOp { expr, .. } => expr.span,
            SemanticError::UnreachableCode { statement } => statement.span,
        }.file_id()
    }
}

type SemanticResult<T> = Result<T, SemanticError>;

impl SemanticAnalyzer {
    pub fn new() -> SemanticAnalyzer {
        SemanticAnalyzer {
            variables: ChainMap::new(),
            function_variables: vec![],
            functions: HashMap::new(),
            structs: HashMap::new(),
            loop_depth: 0,
            current_function: None,
        }
    }

    pub fn analyze_ast(&mut self, ast: Ast, name: &str) -> SemanticResult<Module> {
        let mut module = Module {
            name: name.to_string(),
            functions: vec![],
            structs: HashMap::new(),
        };

        //self.declare_builtins();
        self.declare_structs(&ast)?;
        self.declare_functions(&ast)?;

        for item in ast.nodes {
            match item.kind {
                ItemKind::Function(func) => {
                    let extended_func = self.analyze_function(func)?;
                    module.functions.push(extended_func);
                }
                ItemKind::Import(_) => {}
                ItemKind::Struct(struct_) => {
                    let name = struct_.name.value.clone();
                    let extended_struct = self.analyze_struct(struct_)?;
                    module.structs.insert(name, extended_struct);
                }
            }
        }

        Ok(module)
    }

    //fn declare_builtins(&mut self) {
    //    for (name, arg_ty, ret_ty) in BUILTINS {
    //        let sig = FunctionSignature {
    //            name: Identifier {
    //                value: name.to_string(),
    //                span: NodeLocation::NULL,               },
    //            args: vec![("arg".to_string(), arg_ty.clone())],
    //            ret_ty: ret_ty.clone(),
    //        };
    //        self.functions.insert(sig.name.value.clone(), sig);
    //    }
    //}

    fn declare_functions(&mut self, ast: &Ast) -> SemanticResult<()> {
        for item in &ast.nodes {
            match &item.kind {
                ItemKind::Function(func) => {
                    if self.functions.contains_key(&func.sig.name.value) {
                        return Err(SemanticError::DoubleFunctionDeclaration {
                            ident: func.sig.name.clone(),
                        });
                    }
                    self.functions
                        .insert(func.sig.name.value.clone(), func.sig.clone());
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn declare_structs(&mut self, ast: &Ast) -> SemanticResult<()> {
        for item in &ast.nodes {
            match &item.kind {
                ItemKind::Struct(struct_) => {
                    if self.structs.contains_key(&struct_.name.value) {
                        return Err(SemanticError::DoubleStructDeclaration {
                            ident: struct_.name.clone(),
                        });
                    }
                    let name = struct_.name.value.clone();
                    self.structs.insert(name, struct_.clone());
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn analyze_function(&mut self, func: Function) -> SemanticResult<ExtendedFunction> {
        self.function_variables.clear();
        self.variables.push();
        for arg in &*func.sig.args {
            let var = Variable {
                name: arg.name.clone(),
                ty: arg.ty.kind.clone(),
            };
            self.variables
                .insert(var.name.value.clone(), var.ty)
                .unwrap();
        }

        self.current_function = Some(func.sig.name.value.clone());
        if let Some(body) = &func.body {
            self.analyze_body(body)?;
        }
        self.current_function = None;
        self.variables.pop();

        let mut extended_func = ExtendedFunction {
            inner: func,
            variables: vec![],
        };

        extended_func.variables.append(&mut self.function_variables);

        Ok(extended_func)
    }

    fn analyze_struct(&mut self, struct_: Struct) -> SemanticResult<ExtendedStruct> {
        let mut fields = HashMap::new();
        for (i, field) in struct_.fields.iter().enumerate() {
            fields.insert(field.name.value.clone(), (field.ty.kind.clone(), i));
        }
        Ok(ExtendedStruct {
            inner: struct_,
            fields,
        })
    }

    fn analyze_statement(&mut self, stmt: &Statement) -> SemanticResult<bool> {
        match &stmt.kind {
            StatementKind::Expr(expr) => {
                self.analyze_expr(expr)?;
            }
            StatementKind::Decl(decl) => self.analyze_decl(decl)?,
            StatementKind::Assign(assign) => self.analyze_assignment(assign)?,
            StatementKind::If { cond, then, or } => self.analyze_if(cond, then, or)?,
            StatementKind::Loop { body } => self.analyze_loop(body)?,
            StatementKind::Break => {
                if self.loop_depth == 0 {
                    return Err(SemanticError::BreakOutsideOfLoop {
                        statement: stmt.clone(),
                    });
                }
                return Ok(true);
            }
            StatementKind::Return(_) => {
                self.analyze_return(stmt)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn analyze_body(&mut self, body: &Block) -> SemanticResult<()> {
        self.variables.push();

        for stmt in &body.stmts {
            self.analyze_statement(stmt)?;
        }

        self.variables.pop();
        Ok(())
    }

    fn analyze_assignment(&mut self, assign: &Assignment) -> SemanticResult<()> {
        let expr_result = self.analyze_expr(&assign.value)?;
        let expected = self.analyze_expr(&assign.assignee)?;
        if expr_result != expected {
            return Err(SemanticError::ExpressionTypeMismatch {
                expected: expected.clone(),
                found: expr_result,
                expr: assign.value.clone(),
            });
        }
        Ok(())
    }

    fn analyze_decl(&mut self, decl: &Assignment) -> SemanticResult<()> {
        let ident = match &decl.assignee.kind {
            ExprKind::Var(ident) => ident,
            _ => unreachable!("This function should only be called with a variable assignment"),
        };
        if self.variables.contains_key(&ident.value) {
            return Err(SemanticError::DoubleVariableDeclaration {
                ident: ident.clone(),
            });
        }
        let var = Variable {
            name: ident.clone(),
            ty: self.analyze_expr(&decl.value)?,
        };

        self.variables
            .insert(var.name.value.clone(), var.ty.clone());
        self.function_variables.push(var);
        Ok(())
    }

    fn analyze_if(&mut self, cond: &Expr, then: &Block, or: &Option<Block>) -> SemanticResult<()> {
        let cond_ty = self.analyze_expr(cond)?;
        if cond_ty != TypeKind::Bool {
            return Err(SemanticError::ExpressionTypeMismatch {
                expected: TypeKind::Bool,
                found: cond_ty,
                expr: cond.clone(),
            });
        }
        self.analyze_body(then)?;
        if let Some(or) = or {
            self.analyze_body(or)?;
        }
        Ok(())
    }

    fn analyze_loop(&mut self, body: &Block) -> SemanticResult<()> {
        self.loop_depth += 1;
        self.analyze_body(body)?;
        self.loop_depth -= 1;
        Ok(())
    }

    fn analyze_return(&mut self, statement: &Statement) -> SemanticResult<()> {
        let expr = match &statement.kind {
            StatementKind::Return(expr) => expr,
            _ => unreachable!("This function should only be called with an expression statement"),
        };

        let ret_ty = if let Some(expr) = expr {
            self.analyze_expr(expr)?
        } else {
            TypeKind::Void
        };

        if let Some(func) = self
            .current_function
            .as_ref()
            .and_then(|name| self.functions.get(name))
        {
            if let Some(func_ret) = &func.ret_ty {
                if ret_ty != func_ret.kind {
                    return Err(SemanticError::ReturnTypeMismatch {
                        expected: func_ret.kind.clone(),
                        found: ret_ty,
                        statement: statement.clone(),
                    });
                }
            } else if ret_ty != TypeKind::Void {
                return Err(SemanticError::ReturnTypeMismatch {
                    expected: TypeKind::Void,
                    found: ret_ty,
                    statement: statement.clone(),
                });
            }
        } else {
            return Err(SemanticError::ReturnOutsideOfFunction {
                stmt: statement.clone(),
            });
        }
        Ok(())
    }

    fn analyze_expr(&self, expr: &Expr) -> SemanticResult<TypeKind> {
        match &expr.kind {
            ExprKind::Literal(lit_kind) => match lit_kind {
                LiteralKind::Int(_) => Ok(TypeKind::Int),
                LiteralKind::Float(_) => Ok(TypeKind::Float),
                LiteralKind::Bool(_) => Ok(TypeKind::Bool),
                LiteralKind::String(_) => Ok(TypeKind::String),
            },
            ExprKind::Var(var) => {
                if let Some(var) = self.variables.get(&var.value) {
                    Ok(var.clone())
                } else {
                    Err(SemanticError::VariableNotFound {
                        ident: var.clone(),
                    })
                }
            }
            ExprKind::BinOp(_) => self.analyze_binop(expr),
            ExprKind::UnaryOp { .. } => self.analyze_unaryop(expr),
            ExprKind::Call { name, args } => {
                if let Some(func) = self.functions.get(&name.value) {
                    if func.args.len() != args.len() && !func.args.variadic {
                        return Err(SemanticError::ArgumentCountMismatch {
                            expected: func.args.len(),
                            found: args.len(),
                            args: args.clone(),
                        });
                    }
                    let diff = args.len() - func.args.len();
                    for (arg, expected) in args.iter().zip(func.args.iter()) {
                        let arg_ty = self.analyze_expr(arg)?;
                        if arg_ty != expected.ty.kind {
                            return Err(SemanticError::ArgumentTypeMismatch {
                                expected: expected.ty.kind.clone(),
                                found: arg_ty,
                                arg: arg.clone(),
                            });
                        }
                    }
                    Ok(func
                        .ret_ty
                        .as_ref()
                        .map(|ty| ty.kind.clone())
                        .unwrap_or(TypeKind::Void))
                } else {
                    Err(SemanticError::FunctionNotFound {
                        ident: name.clone(),
                    })
                }
            }
            ExprKind::FieldAccess { lhs, field } => {
                let lhs_ty = self.analyze_expr(lhs)?;
                let struct_ = match lhs_ty {
                    TypeKind::Struct(name) => {
                        if let Some(struct_) = self.structs.get(&name) {
                            struct_
                        } else {
                            panic!("struct not found")
                        }
                    }
                    _ => panic!("field access on non-struct type"),
                };
                if let Some(field) = struct_.fields.iter().find(|f| f.name.value == *field.value) {
                    Ok(field.ty.kind.clone())
                } else {
                    panic!("field not found")
                }
            }
            ExprKind::StructInit { name, fields } => {
                if let Some(struct_) = self.structs.get(&name.value) {
                    for (field, expr) in fields {
                        let field_ty = struct_
                            .fields
                            .iter()
                            .find(|f| f.name.value == *field)
                            .map(|f| f.ty.clone())
                            .unwrap();
                        let expr_ty = self.analyze_expr(expr)?;
                        if field_ty.kind != expr_ty {
                            return Err(SemanticError::ExpressionTypeMismatch {
                                expected: field_ty.kind,
                                found: expr_ty,
                                expr: expr.clone(),
                            });
                        }
                    }
                    Ok(TypeKind::Struct(name.value.clone()))
                } else {
                    panic!("struct not found")
                }
            }
        }
    }

    fn analyze_binop(&self, expr: &Expr) -> SemanticResult<TypeKind> {
        let binop = match expr {
            Expr {
                kind: ExprKind::BinOp(binop),
                ..
            } => binop,
            _ => unreachable!("This function should only be called with a binary operation"),
        };
        let lhs_ty = self.analyze_expr(&binop.lhs)?;
        let rhs_ty = self.analyze_expr(&binop.rhs)?;
        if lhs_ty != rhs_ty {
            return Err(SemanticError::ExpressionTypeMismatch {
                expected: lhs_ty,
                found: rhs_ty,
                expr: *binop.rhs.clone(),
            });
        }

        if lhs_ty == TypeKind::Float
            && (binop.kind == BinOpKind::And || binop.kind == BinOpKind::Or)
        {
            return Err(SemanticError::UnsupportedBinOp {
                ty: TypeKind::Float,
                expr: expr.clone(),
            });
        }

        let return_ty = match binop.kind {
            parsing::BinOpKind::Add
            | parsing::BinOpKind::Sub
            | parsing::BinOpKind::Mul
            | parsing::BinOpKind::Div
            | parsing::BinOpKind::Mod => lhs_ty,
            parsing::BinOpKind::Eq
            | parsing::BinOpKind::Neq
            | parsing::BinOpKind::Lt
            | parsing::BinOpKind::Leq
            | parsing::BinOpKind::Gt
            | parsing::BinOpKind::Geq => TypeKind::Bool,
            parsing::BinOpKind::And | parsing::BinOpKind::Or => TypeKind::Bool,
        };
        Ok(return_ty)
    }

    fn analyze_unaryop(&self, expr: &Expr) -> SemanticResult<TypeKind> {
        let (kind, rhs) = match expr {
            Expr {
                kind: ExprKind::UnaryOp { kind, rhs },
                ..
            } => (kind, rhs),
            _ => unreachable!("This function should only be called with a unary operation"),
        };
        let rhs_ty = self.analyze_expr(rhs)?;
        match kind {
            UnaryOpKind::Cast(ty) => Ok(ty.kind.clone()),
            _ => Ok(rhs_ty),
        }
    }
}
