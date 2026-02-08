use std::{collections::HashMap, path::Path};

use crate::{
    parsing::{
        Assignment, BinOp, BinOpKind, Block, Expr, ExprKind, Identifier, LiteralKind, Statement,
        StatementKind, Type, UnaryOpKind,
    },
    semantical::{ExtendedFunction, ExtendedStruct, Module},
};
use inkwell::{
    builder::Builder as LLVMBuilder,
    context::Context as LLVMContext,
    module::Module as LLVMModule,
    targets::{CodeModel, RelocMode, Target, TargetMachine},
    types::{BasicType, BasicTypeEnum},
    values::{BasicValue, PointerValue},
    AddressSpace, OptimizationLevel,
};

pub struct CodeGen<'a> {
    context: &'a LLVMContext,
    llvm_module: LLVMModule<'a>,
    llvm_builder: LLVMBuilder<'a>,
    variables: HashMap<String, (PointerValue<'a>, BasicTypeEnum<'a>)>,
    struct_types: HashMap<String, ExtendedStruct>,
    loop_exits: Vec<inkwell::basic_block::BasicBlock<'a>>,
}

impl Type {
    fn to_llvm<'a>(&self, context: &'a LLVMContext) -> inkwell::types::BasicTypeEnum<'a> {
        match self {
            Type::Int => context.i64_type().as_basic_type_enum(),
            Type::Float => context.f64_type().as_basic_type_enum(),
            Type::Bool => context.bool_type().as_basic_type_enum(),
            Type::Void => context.i64_type().as_basic_type_enum(),
            Type::String => context
                .ptr_type(AddressSpace::default())
                .as_basic_type_enum(),
            Type::Struct(name) => context.get_struct_type(name).unwrap().as_basic_type_enum(),
        }
    }
}

impl<'a> CodeGen<'a> {
    pub fn new(context: &'a LLVMContext, llvm_module: LLVMModule<'a>) -> Self {
        let llvm_builder = context.create_builder();
        CodeGen {
            context,
            llvm_module,
            llvm_builder,
            variables: HashMap::new(),
            struct_types: HashMap::new(),
            loop_exits: vec![],
        }
    }

    pub fn spit_out(&self) {
        self.llvm_module.print_to_stderr();
    }

    pub fn verify(&self) {
        let verify = self.llvm_module.verify();
        println!("{:?}", verify);
    }

    pub fn spit_out_object(&self, file_name: &str) {
        Target::initialize_all(&Default::default());
        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple).unwrap();
        let reloc_mode = RelocMode::PIC;
        let code_model = CodeModel::Default;
        let opt_level = OptimizationLevel::Aggressive;
        let target_machine = target
            .create_target_machine(
                &target_triple,
                "generic",
                "",
                opt_level,
                reloc_mode,
                code_model,
            )
            .unwrap();
        let object_file = Path::new(file_name).with_extension("o");
        let object_file = object_file.as_path();
        target_machine
            .write_to_file(
                &self.llvm_module,
                inkwell::targets::FileType::Object,
                object_file,
            )
            .unwrap();

        // CC

        let executable_file = Path::new(file_name).with_extension("out");

        std::process::Command::new("cc")
            .arg(object_file)
            .arg("-o")
            .arg(executable_file)
            .status()
            .unwrap();
    }

    pub fn declare_functions(&mut self, module: &Module) {
        // Generate builtin functions
        self.generate_printf();
        self.generate_iprint();
        self.generate_fprint();
        self.generate_bprint();

        // Declare all user functions
        for function in module.functions.iter() {
            let mut args = vec![];
            for arg in function.inner.sig.args.iter() {
                args.push(arg.1.to_llvm(self.context).into());
            }
            let ret_ty = function.inner.sig.ret_ty.to_llvm(self.context);
            let function_type = ret_ty.fn_type(&args, false);
            self.llvm_module
                .add_function(&function.inner.sig.name.value, function_type, None);
        }
    }

    pub fn declare_structs(&mut self, module: &Module) {
        for struct_decl in module.structs.iter() {
            let struct_ = struct_decl.1;
            self.context.opaque_struct_type(&struct_.inner.name.value);
        }
        for struct_decl in module.structs.iter() {
            let struct_ = struct_decl.1;
            let struct_type = self
                .context
                .get_struct_type(&struct_.inner.name.value)
                .unwrap();
            let mut members = vec![];
            for field in struct_.inner.fields.iter() {
                members.push(field.ty.to_llvm(self.context));
            }
            struct_type.set_body(&members, false);
        }
        self.struct_types = module.structs.clone();
    }

    pub fn generate(&mut self, module: Module) {
        self.declare_structs(&module);
        self.declare_functions(&module);
        for function in module.functions.iter() {
            self.generate_function(function);
        }
    }

    pub fn generate_printf(&mut self) {
        let string_type = self.context.ptr_type(AddressSpace::default());
        let function_type = self.context.i64_type().fn_type(&[string_type.into()], true);
        let _printf = self.llvm_module.add_function("printf", function_type, None);
    }

    pub fn generate_iprint(&mut self) {
        let printf = self.llvm_module.get_function("printf").unwrap();
        let int_type = Type::Int.to_llvm(self.context);
        let iprint_type = self.context.i64_type().fn_type(&[int_type.into()], false);
        let iprint = self.llvm_module.add_function("iprint", iprint_type, None);
        let basic_block = self.context.append_basic_block(iprint, "entry");
        self.llvm_builder.position_at_end(basic_block);
        let fmt = self
            .llvm_builder
            .build_global_string_ptr("%lld\n", "fmt_iprint")
            .unwrap();
        let arg = iprint.get_first_param().unwrap();
        let call = self
            .llvm_builder
            .build_call(printf, &[fmt.as_pointer_value().into(), arg.into()], "call")
            .unwrap();
        let call_value = call.try_as_basic_value().unwrap_basic();
        self.llvm_builder.build_return(Some(&call_value)).unwrap();
    }

    pub fn generate_fprint(&mut self) {
        let printf = self.llvm_module.get_function("printf").unwrap();
        let float_type = Type::Float.to_llvm(self.context);
        let fprint_type = self.context.i64_type().fn_type(&[float_type.into()], false);
        let fprint = self.llvm_module.add_function("fprint", fprint_type, None);
        let basic_block = self.context.append_basic_block(fprint, "entry");
        self.llvm_builder.position_at_end(basic_block);
        let fmt = self
            .llvm_builder
            .build_global_string_ptr("%lf\n", "fmt_fprint")
            .unwrap();
        let arg = fprint.get_first_param().unwrap();
        let call = self
            .llvm_builder
            .build_call(printf, &[fmt.as_pointer_value().into(), arg.into()], "call")
            .unwrap();
        let call_value = call.try_as_basic_value().unwrap_basic();
        self.llvm_builder.build_return(Some(&call_value)).unwrap();
    }

    pub fn generate_bprint(&mut self) {
        let printf = self.llvm_module.get_function("printf").unwrap();
        let bool_type = Type::Bool.to_llvm(self.context);
        let bprint_type = self.context.i64_type().fn_type(&[bool_type.into()], false);
        let bprint = self.llvm_module.add_function("bprint", bprint_type, None);
        let basic_block = self.context.append_basic_block(bprint, "entry");
        self.llvm_builder.position_at_end(basic_block);
        let fmt = self
            .llvm_builder
            .build_global_string_ptr("%s\n", "fmt_bprint")
            .unwrap();
        let arg = bprint.get_first_param().unwrap();
        let true_str = self
            .llvm_builder
            .build_global_string_ptr("true", "true")
            .unwrap();
        let false_str = self
            .llvm_builder
            .build_global_string_ptr("false", "false")
            .unwrap();
        let call = self
            .llvm_builder
            .build_select(
                arg.into_int_value(),
                true_str.as_pointer_value(),
                false_str.as_pointer_value(),
                "select",
            )
            .unwrap();
        let call = self
            .llvm_builder
            .build_call(
                printf,
                &[fmt.as_pointer_value().into(), call.into()],
                "call",
            )
            .unwrap();
        let call_value = call.try_as_basic_value().unwrap_basic();
        self.llvm_builder.build_return(Some(&call_value)).unwrap();
    }

    pub fn generate_function(&mut self, function: &ExtendedFunction) {
        let llvm_function = self
            .llvm_module
            .get_function(&function.inner.sig.name.value)
            .unwrap();
        let basic_block = self.context.append_basic_block(llvm_function, "entry");
        self.llvm_builder.position_at_end(basic_block);
        for (i, arg) in llvm_function.get_param_iter().enumerate() {
            let arg_name = &function.inner.sig.args[i].0;
            let alloca = self
                .llvm_builder
                .build_alloca(arg.get_type(), arg_name)
                .unwrap();
            self.llvm_builder.build_store(alloca, arg).unwrap();
            self.variables
                .insert(arg_name.to_string(), (alloca, arg.get_type()));
        }
        for variable in function.variables.iter() {
            let var_type = variable.ty.to_llvm(self.context);
            let var = self
                .llvm_builder
                .build_alloca(var_type, &variable.name)
                .unwrap();
            self.variables
                .insert(variable.name.clone(), (var, var_type));
        }
        for statement in function.inner.body.stmts.iter() {
            self.generate_statement(statement);
        }
    }

    pub fn generate_block(
        &mut self,
        block: &Block,
        basic_block: inkwell::basic_block::BasicBlock<'a>,
    ) -> bool {
        self.llvm_builder.position_at_end(basic_block);
        let mut exited_early = false;
        for statement in block.stmts.iter() {
            exited_early = self.generate_statement(statement);
        }
        exited_early
    }

    pub fn generate_statement(&mut self, statement: &Statement) -> bool {
        match &statement.kind {
            StatementKind::Return(expr) => {
                let value = if let Some(expr) = expr {
                    self.generate_expr::<false>(expr)
                } else {
                    self.context.i64_type().const_int(0, false).into()
                };
                self.llvm_builder.build_return(Some(&value)).unwrap();
                return true;
            }
            StatementKind::Expr(expr) => {
                self.generate_expr::<false>(expr);
            }
            StatementKind::Decl(assignment) => self.generate_assignment(assignment),
            StatementKind::Assign(assignment) => self.generate_assignment(assignment),
            StatementKind::If { cond, then, or } => self.generate_if(cond, then, or),
            StatementKind::Loop { body } => self.generate_loop(body),
            StatementKind::Break => {
                let current_block = self.llvm_builder.get_insert_block().unwrap();
                let end_block = self.loop_exits.last().unwrap();
                self.llvm_builder
                    .build_unconditional_branch(*end_block)
                    .unwrap();
                self.llvm_builder.position_at_end(current_block);
                return true;
            }
        }
        false
    }

    pub fn generate_expr<const RET_PTR: bool>(
        &mut self,
        expr: &Expr,
    ) -> inkwell::values::BasicValueEnum<'a> {
        match &expr.kind {
            ExprKind::Literal(lit) => self.generate_literal(lit),
            ExprKind::Var(ident) => self.generate_var(ident, RET_PTR),
            ExprKind::BinOp(binop) => self.generate_binop(binop),
            ExprKind::UnaryOp { kind, rhs } => self.generate_unaryop(kind, rhs),
            ExprKind::Call { name, args } => self.generate_call(name, args),
            ExprKind::FieldAccess { lhs, field } => self.generate_field_acces(lhs, field, RET_PTR),
            ExprKind::StructInit { name, fields } => self.generate_struct_init(name, fields),
        }
    }

    pub fn generate_literal(&mut self, lit: &LiteralKind) -> inkwell::values::BasicValueEnum<'a> {
        match lit {
            LiteralKind::Int(i) => self.context.i64_type().const_int(*i as u64, false).into(),
            LiteralKind::Float(f) => self.context.f64_type().const_float(*f).into(),
            LiteralKind::Bool(b) => self.context.bool_type().const_int(*b as u64, false).into(),
            LiteralKind::String(s) => {
                let string = self
                    .llvm_builder
                    .build_global_string_ptr(s, "string")
                    .unwrap();
                string.as_pointer_value().into()
            }
        }
    }

    pub fn generate_var(
        &mut self,
        ident: &Identifier,
        ret_ptr: bool,
    ) -> inkwell::values::BasicValueEnum<'a> {
        let (ptr, ty) = self.variables.get(&ident.value).unwrap();
        if ret_ptr {
            ptr.as_basic_value_enum()
        } else {
            self.llvm_builder
                .build_load(*ty, *ptr, &ident.value)
                .unwrap()
        }
    }

    pub fn generate_binop(&mut self, binop: &BinOp) -> inkwell::values::BasicValueEnum<'a> {
        let lhs = self.generate_expr::<false>(&binop.lhs);
        let rhs = self.generate_expr::<false>(&binop.rhs);
        let ty = lhs.get_type();
        match ty {
            inkwell::types::BasicTypeEnum::FloatType(_) => match &binop.kind {
                BinOpKind::Add => self
                    .llvm_builder
                    .build_float_add(lhs.into_float_value(), rhs.into_float_value(), "add")
                    .unwrap()
                    .into(),
                BinOpKind::Sub => self
                    .llvm_builder
                    .build_float_sub(lhs.into_float_value(), rhs.into_float_value(), "sub")
                    .unwrap()
                    .into(),
                BinOpKind::Mul => self
                    .llvm_builder
                    .build_float_mul(lhs.into_float_value(), rhs.into_float_value(), "mul")
                    .unwrap()
                    .into(),
                BinOpKind::Div => self
                    .llvm_builder
                    .build_float_div(lhs.into_float_value(), rhs.into_float_value(), "div")
                    .unwrap()
                    .into(),
                BinOpKind::Mod => self
                    .llvm_builder
                    .build_float_rem(lhs.into_float_value(), rhs.into_float_value(), "mod")
                    .unwrap()
                    .into(),
                BinOpKind::Eq => self
                    .llvm_builder
                    .build_float_compare(
                        inkwell::FloatPredicate::OEQ,
                        lhs.into_float_value(),
                        rhs.into_float_value(),
                        "eq",
                    )
                    .unwrap()
                    .into(),
                BinOpKind::Neq => self
                    .llvm_builder
                    .build_float_compare(
                        inkwell::FloatPredicate::ONE,
                        lhs.into_float_value(),
                        rhs.into_float_value(),
                        "neq",
                    )
                    .unwrap()
                    .into(),
                BinOpKind::Lt => self
                    .llvm_builder
                    .build_float_compare(
                        inkwell::FloatPredicate::OLT,
                        lhs.into_float_value(),
                        rhs.into_float_value(),
                        "lt",
                    )
                    .unwrap()
                    .into(),
                BinOpKind::Gt => self
                    .llvm_builder
                    .build_float_compare(
                        inkwell::FloatPredicate::OGT,
                        lhs.into_float_value(),
                        rhs.into_float_value(),
                        "gt",
                    )
                    .unwrap()
                    .into(),
                BinOpKind::Leq => self
                    .llvm_builder
                    .build_float_compare(
                        inkwell::FloatPredicate::OLE,
                        lhs.into_float_value(),
                        rhs.into_float_value(),
                        "leq",
                    )
                    .unwrap()
                    .into(),
                BinOpKind::Geq => self
                    .llvm_builder
                    .build_float_compare(
                        inkwell::FloatPredicate::OGE,
                        lhs.into_float_value(),
                        rhs.into_float_value(),
                        "geq",
                    )
                    .unwrap()
                    .into(),
                _ => unimplemented!("unsupported float operation"),
            },
            inkwell::types::BasicTypeEnum::IntType(_) => match &binop.kind {
                BinOpKind::Add => self
                    .llvm_builder
                    .build_int_add(lhs.into_int_value(), rhs.into_int_value(), "add")
                    .unwrap()
                    .into(),
                BinOpKind::Sub => self
                    .llvm_builder
                    .build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "sub")
                    .unwrap()
                    .into(),
                BinOpKind::Mul => self
                    .llvm_builder
                    .build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "mul")
                    .unwrap()
                    .into(),
                BinOpKind::Div => self
                    .llvm_builder
                    .build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "div")
                    .unwrap()
                    .into(),
                BinOpKind::Mod => self
                    .llvm_builder
                    .build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "mod")
                    .unwrap()
                    .into(),
                BinOpKind::Eq => self
                    .llvm_builder
                    .build_int_compare(
                        inkwell::IntPredicate::EQ,
                        lhs.into_int_value(),
                        rhs.into_int_value(),
                        "eq",
                    )
                    .unwrap()
                    .into(),
                BinOpKind::Neq => self
                    .llvm_builder
                    .build_int_compare(
                        inkwell::IntPredicate::NE,
                        lhs.into_int_value(),
                        rhs.into_int_value(),
                        "neq",
                    )
                    .unwrap()
                    .into(),
                BinOpKind::Lt => self
                    .llvm_builder
                    .build_int_compare(
                        inkwell::IntPredicate::SLT,
                        lhs.into_int_value(),
                        rhs.into_int_value(),
                        "lt",
                    )
                    .unwrap()
                    .into(),
                BinOpKind::Gt => self
                    .llvm_builder
                    .build_int_compare(
                        inkwell::IntPredicate::SGT,
                        lhs.into_int_value(),
                        rhs.into_int_value(),
                        "gt",
                    )
                    .unwrap()
                    .into(),
                BinOpKind::Leq => self
                    .llvm_builder
                    .build_int_compare(
                        inkwell::IntPredicate::SLE,
                        lhs.into_int_value(),
                        rhs.into_int_value(),
                        "leq",
                    )
                    .unwrap()
                    .into(),
                BinOpKind::Geq => self
                    .llvm_builder
                    .build_int_compare(
                        inkwell::IntPredicate::SGE,
                        lhs.into_int_value(),
                        rhs.into_int_value(),
                        "geq",
                    )
                    .unwrap()
                    .into(),
                _ => unimplemented!("unsupported int operation"),
            },
            _ => unimplemented!("unsupported type {:?}", ty),
        }
    }

    pub fn generate_unaryop(
        &mut self,
        kind: &UnaryOpKind,
        expr: &Expr,
    ) -> inkwell::values::BasicValueEnum<'a> {
        let value = self.generate_expr::<false>(expr);
        match kind {
            UnaryOpKind::Neg => match value.get_type() {
                inkwell::types::BasicTypeEnum::FloatType(_) => self
                    .llvm_builder
                    .build_float_neg(value.into_float_value(), "neg")
                    .unwrap()
                    .into(),
                inkwell::types::BasicTypeEnum::IntType(_) => self
                    .llvm_builder
                    .build_int_neg(value.into_int_value(), "neg")
                    .unwrap()
                    .into(),
                _ => unimplemented!("unsupported type"),
            },
            UnaryOpKind::Not => self
                .llvm_builder
                .build_not(value.into_int_value(), "not")
                .unwrap()
                .into(),
            UnaryOpKind::Cast(ty) => match (value.get_type(), ty) {
                (inkwell::types::BasicTypeEnum::FloatType(_), Type::Float) => value,
                (inkwell::types::BasicTypeEnum::IntType(_), Type::Int) => value,
                (inkwell::types::BasicTypeEnum::IntType(_), Type::Float) => self
                    .llvm_builder
                    .build_signed_int_to_float(
                        value.into_int_value(),
                        self.context.f64_type(),
                        "cast",
                    )
                    .unwrap()
                    .into(),
                (inkwell::types::BasicTypeEnum::FloatType(_), Type::Int) => self
                    .llvm_builder
                    .build_float_to_signed_int(
                        value.into_float_value(),
                        self.context.i64_type(),
                        "cast",
                    )
                    .unwrap()
                    .into(),
                _ => unimplemented!("unsupported cast"),
            },
        }
    }

    pub fn generate_call(
        &mut self,
        name: &str,
        args: &Vec<Expr>,
    ) -> inkwell::values::BasicValueEnum<'a> {
        let function = self.llvm_module.get_function(name).unwrap();
        let mut llvm_args = vec![];
        for arg in args {
            llvm_args.push(self.generate_expr::<false>(arg).into());
        }
        self.llvm_builder
            .build_call(function, &llvm_args, "call")
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic()
    }

    pub fn generate_assignment(&mut self, assignment: &Assignment) {
        let ptr = self.generate_expr::<true>(&assignment.assignee);
        let value = self.generate_expr::<false>(&assignment.value);
        self.llvm_builder
            .build_store(ptr.into_pointer_value(), value)
            .unwrap();
    }

    pub fn generate_if(&mut self, cond: &Expr, then: &Block, or: &Option<Block>) {
        let cond = self.generate_expr::<false>(cond);
        let current_block = self.llvm_builder.get_insert_block().unwrap();
        let function = current_block.get_parent().unwrap();
        let then_block = self.context.append_basic_block(function, "if_then");
        let exited_early = self.generate_block(then, then_block);
        if !exited_early {
            self.llvm_builder
                .build_unconditional_branch(then_block)
                .unwrap();
        }
        let end_block = self.context.append_basic_block(function, "if_end");
        if let Some(or) = or {
            let or_block = self.context.append_basic_block(function, "if_or");
            let exited_early = self.generate_block(or, or_block);
            println!("exited_early: {}", exited_early);
            if !exited_early {
                self.llvm_builder
                    .build_unconditional_branch(end_block)
                    .unwrap();
            }
            self.llvm_builder.position_at_end(current_block);
            self.llvm_builder
                .build_conditional_branch(cond.into_int_value(), then_block, or_block)
                .unwrap();
            self.llvm_builder.position_at_end(or_block);
        } else {
            self.llvm_builder.position_at_end(current_block);
            self.llvm_builder
                .build_conditional_branch(cond.into_int_value(), then_block, end_block)
                .unwrap();
        }
        self.llvm_builder.position_at_end(end_block);
    }

    fn generate_loop(&mut self, body: &Block) {
        let current_block = self.llvm_builder.get_insert_block().unwrap();
        let function = current_block.get_parent().unwrap();
        let loop_block = self.context.append_basic_block(function, "loop_block");
        self.llvm_builder
            .build_unconditional_branch(loop_block)
            .unwrap();
        let end_block = self.context.append_basic_block(function, "loop_exit");
        self.loop_exits.push(end_block);
        self.generate_block(body, loop_block);
        self.llvm_builder
            .build_unconditional_branch(loop_block)
            .unwrap();
        self.llvm_builder.position_at_end(end_block);
        self.loop_exits.pop();
    }

    fn generate_field_acces(
        &mut self,
        lhs: &Expr,
        field: &Identifier,
        ret_ptr: bool,
    ) -> inkwell::values::BasicValueEnum<'a> {
        let lhs_unused = self.generate_expr::<false>(lhs);
        let lhs = self.generate_expr::<true>(lhs);
        let struct_type = lhs_unused.get_type().into_struct_type();
        let struct_name = struct_type.get_name().unwrap().to_str().unwrap();
        let struct_ty = self.struct_types.get(struct_name).unwrap();
        let (field_ty, field_idx) = struct_ty.fields.get(&field.value).unwrap();
        let field_ptr = self
            .llvm_builder
            .build_struct_gep(
                struct_type,
                lhs.into_pointer_value(),
                *field_idx as u32,
                "field_access",
            )
            .unwrap();
        if ret_ptr {
            field_ptr.into()
        } else {
            self.llvm_builder
                .build_load(field_ty.to_llvm(self.context), field_ptr, "field_access")
                .unwrap()
        }
    }

    fn generate_struct_init(
        &mut self,
        name: &Identifier,
        fields: &Vec<(String, Expr)>,
    ) -> inkwell::values::BasicValueEnum<'a> {
        let struct_type = self.context.get_struct_type(&name.value).unwrap();
        let struct_val = self
            .llvm_builder
            .build_alloca(struct_type, "struct")
            .unwrap();
        for (i, field) in fields.iter().enumerate() {
            let field_val = self.generate_expr::<false>(&field.1);
            let field_ptr = self
                .llvm_builder
                .build_struct_gep(struct_type, struct_val, i as u32, "field_set")
                .unwrap();
            self.llvm_builder.build_store(field_ptr, field_val);
        }
        self.llvm_builder
            .build_load(struct_type, struct_val, "struct_init")
            .unwrap()
    }
}
