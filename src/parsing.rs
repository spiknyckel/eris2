// BNF:
// <ws> ::= (" " | "\t")*
// <number> ::= <digit> | <number> <digit>
// <digit> ::= [0-9]
// <letter> ::= "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z" | "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
// <binop> ::= "+" | "-" | "*" | "/" | "%"
// <unaryop> ::= "-"
// <binary> ::= <expr> <ws> <binop> <ws> <expr>
// <unary> ::= <unaryop> <ws> <expr>
// <ident> ::= <letter> | <ident> <letter> | <ident> "_"
// <bool> ::= "true" | "false"
// <literal> ::= <number> | <bool>
// <decl> ::= <ident> <ws> ":=" <ws> <expr>
// <assign> ::= <ident> <ws> "=" <ws> <expr>
// <expr> ::= "(" <ws> <expr> <ws> ")" | <ident> | <binary> | <unary> | <literal> | <assign> | <if> | <break> | <loop> | <call>
// <args> ::= <expr> | <args> "," <expr>
// <stmt> ::= <expr> ";" | <decl> ";"
// <block> ::= "{" <ws> <stmt>* <ws> "}"
// <sig_args> ::= <ident> | <sig_args> <ws> "," <ws> <ident>
// <sig> ::= <ident> <ws> "::" <ws> "(" <sig_args> ")"
// <function> ::= <sig> <ws> <block>
// <break> ::= "break"
// <if> ::= "if" <ws> <block> | "if" <ws> <block> <ws> "else" <ws> <block>
// <loop> ::= "loop" <ws> <block>
// <call> ::= <ident> <ws> "(" <args> ")"

use std::ops::Deref;

use itertools::{peek_nth, PeekNth};

use crate::tokenizing::{
    LiteralKind as TokenLiteralKind, NumberKind, SymbolKind, Token, TokenKind, TokenLocation,
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NodeLocation {
    start: TokenLocation,
    end: TokenLocation,
}

impl NodeLocation {
    pub fn new(start: TokenLocation, end: TokenLocation) -> Self {
        Self {
            start,
            end,
        }
    }
    pub fn merge(&self, other: &NodeLocation) -> NodeLocation {
        NodeLocation {
            start: self.start,
            end: other.end,
        }
    }

    pub fn line_span(&self) -> (u32, u32) {
        (self.start.line_span().0, self.end.line_span().1)
    }

    pub fn col_span(&self) -> (u32, u32) {
        (self.start.col_span().0, self.end.col_span().1)
    }

    pub fn start(&self) -> TokenLocation {
        self.start
    }

    pub fn end(&self) -> TokenLocation {
        self.end
    }

    pub fn file_id(&self) -> usize {
        self.start.file_id()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeKind {
    Int,
    Float,
    Bool,
    Void,
    String,
    Struct(String),
}

#[derive(Debug, Clone)]
pub struct Type {
    pub kind: TypeKind,
    pub span: NodeLocation,
}

impl Deref for Type {
    type Target = TypeKind;

    fn deref(&self) -> &Self::Target {
        &self.kind
    }
}

impl std::fmt::Display for TypeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match &self {
            TypeKind::Int => "int",
            TypeKind::Float => "float",
            TypeKind::Bool => "bool",
            TypeKind::Void => "void",
            TypeKind::String => "string",
            TypeKind::Struct(s) => s,
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug)]
pub struct Ast {
    pub nodes: Vec<Item>,
}

#[derive(Debug)]
pub enum ItemKind {
    Function(Function),
    Import(String),
    Struct(Struct),
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: Identifier,
    pub ty: Type,
    pub span: NodeLocation,
}

#[derive(Debug, Clone)]
pub struct Fields {
    pub fields: Vec<Field>,
    pub span: NodeLocation,
}

impl Deref for Fields {
    type Target = Vec<Field>;

    fn deref(&self) -> &Self::Target {
        &self.fields
    }
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub name: Identifier,
    pub fields: Fields,
    pub span: NodeLocation,
}

#[derive(Debug, Clone)]
pub struct Identifier {
    pub value: String,
    pub span: NodeLocation,
}

impl Deref for Identifier {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[derive(Debug)]
pub struct Item {
    pub kind: ItemKind,
}

#[derive(Debug, Clone)]
pub struct Arg {
    pub name: Identifier,
    pub ty: Type,
    pub span: NodeLocation,
}

#[derive(Debug, Clone)]
pub struct Args {
    pub args: Vec<Arg>,
    pub variadic: bool,
    pub span: NodeLocation,
}

impl Deref for Args {
    type Target = Vec<Arg>;

    fn deref(&self) -> &Self::Target {
        &self.args
    }
}

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: Identifier,
    pub args: Args,
    pub ret_ty: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub sig: FunctionSignature,
    pub body: Option<Block>,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Statement>,
    pub span: NodeLocation,
}

#[derive(Debug, Clone)]
pub enum StatementKind {
    Expr(Expr),
    Decl(Assignment),
    Assign(Assignment),
    If {
        cond: Box<Expr>,
        then: Block,
        or: Option<Block>,
    },
    Loop {
        body: Block,
    },
    Break,
    Return(Option<Box<Expr>>),
}

#[derive(Debug, Clone)]
pub struct Statement {
    pub kind: StatementKind,
    pub span: NodeLocation,
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub assignee: Expr,
    pub value: Expr,
    pub span: NodeLocation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    And,
    Or,
    Eq,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,
}

impl TryFrom<SymbolKind> for BinOpKind {
    type Error = ();

    fn try_from(kind: SymbolKind) -> Result<Self, Self::Error> {
        match kind {
            SymbolKind::Plus => Ok(Self::Add),
            SymbolKind::Minus => Ok(Self::Sub),
            SymbolKind::Star => Ok(Self::Mul),
            SymbolKind::Slash => Ok(Self::Div),
            SymbolKind::Percent => Ok(Self::Mod),
            SymbolKind::And => Ok(Self::And),
            SymbolKind::Or => Ok(Self::Or),
            SymbolKind::Equals => Ok(Self::Eq),
            SymbolKind::Neq => Ok(Self::Neq),
            SymbolKind::Lt => Ok(Self::Lt),
            SymbolKind::Gt => Ok(Self::Gt),
            SymbolKind::Leq => Ok(Self::Leq),
            SymbolKind::Geq => Ok(Self::Geq),
            _ => Err(()),
        }
    }
}

impl BinOpKind {
    pub fn precedence(&self) -> u8 {
        match self {
            Self::Or => 1,
            Self::And => 2,
            Self::Eq | Self::Neq | Self::Lt | Self::Gt | Self::Leq | Self::Geq => 3,
            Self::Add | Self::Sub => 4,
            Self::Mul | Self::Div | Self::Mod => 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinOp {
    pub kind: BinOpKind,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
}

#[derive(Debug, Clone)]
pub enum UnaryOpKind {
    Neg,
    Not,
    Cast(Type),
}

#[derive(Debug, Clone, PartialEq)]
pub enum LiteralKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone)]
pub struct CallArgs {
    pub args: Vec<Expr>,
    pub span: NodeLocation,
}

impl Deref for CallArgs {
    type Target = Vec<Expr>;

    fn deref(&self) -> &Self::Target {
        &self.args
    }
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Literal(LiteralKind),
    Var(Identifier),
    BinOp(BinOp),
    UnaryOp {
        kind: UnaryOpKind,
        rhs: Box<Expr>,
    },
    Call {
        name: Identifier,
        args: CallArgs,
    },
    FieldAccess {
        lhs: Box<Expr>,
        field: Identifier,
    },
    StructInit {
        name: Identifier,
        fields: Vec<(String, Expr)>,
    },
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: NodeLocation,
}

pub struct Parser<I: Iterator<Item = Token>> {
    file_id: usize,
    tokens: PeekNth<I>,
    last_token: Option<Token>,
}

#[derive(Debug)]
pub enum ParsingError {
    UnexpectedToken(Token),
    ExpectedAnotherToken { expected: TokenKind, got: Token },
    UnexpectedEndOfInput,
}

impl ParsingError {
    pub fn to_string(self, source: &str) -> String {
        let (msg, location) = match self {
            ParsingError::UnexpectedToken(token) => (
                format!(
                    "\x1b[31merror\x1b[0m: unexpected token at {}:{}\n",
                    token.location.line_start(), token.location.col_start()
                ),
                token.location,
            ),
            ParsingError::ExpectedAnotherToken { expected, got } => (
                format!(
                    "\x1b[31merror\x1b[0m: expected token {:?} at {}:{}\n",
                    expected, got.location.line_start(), got.location.col_start()
                ),
                got.location,
            ),
            ParsingError::UnexpectedEndOfInput => {
                return "\x1b[31merror\x1b[0m: unexpected end of input".to_string();
            }
        };
        let error = crate::error::error(
            location.line_span(), location.col_span(), source
        );
        format!("{}{}", msg, error)
    }

    pub fn file_id(&self) -> usize {
        match self {
            ParsingError::UnexpectedToken(token) => token.location.file_id(),
            ParsingError::ExpectedAnotherToken { got, .. } => got.location.file_id(),
            ParsingError::UnexpectedEndOfInput => 0,
        }
    }
}

pub type ParsingResult<T> = Result<T, ParsingError>;

impl<I: Iterator<Item = Token>> Parser<I> {
    pub fn new(tokens: I, file_id: usize) -> Self {
        Self {
            file_id,
            tokens: peek_nth(tokens),
            last_token: None,
        }
    }

    pub fn peek(&mut self) -> ParsingResult<&Token> {
        self.tokens.peek().ok_or(ParsingError::UnexpectedEndOfInput)
    }

    pub fn peek_nth(&mut self, n: usize) -> ParsingResult<&Token> {
        self.tokens
            .peek_nth(n)
            .ok_or(ParsingError::UnexpectedEndOfInput)
    }

    pub fn eat(&mut self) -> ParsingResult<Token> {
        let token = self
            .tokens
            .next()
            .ok_or(ParsingError::UnexpectedEndOfInput)?;
        self.last_token = Some(token.clone());
        Ok(token)
    }

    /// Eat a token and assert that it is of a certain kind
    pub fn eat_and_assert(&mut self, kind: TokenKind) -> ParsingResult<Token> {
        let token = self.eat()?;
        if token.kind == kind {
            Ok(token)
        } else {
            Err(ParsingError::ExpectedAnotherToken {
                expected: kind,
                got: token,
            })
        }
    }

    pub fn peek_and_assert(&mut self, kind: TokenKind) -> ParsingResult<&Token> {
        let token = self.peek()?;
        if token.kind == kind {
            Ok(token)
        } else {
            Err(ParsingError::ExpectedAnotherToken {
                expected: kind,
                got: token.clone(),
            })
        }
    }

    /// Returns the location of the next token
    /// without consuming it
    pub fn get_next_location(&mut self) -> ParsingResult<TokenLocation> {
        let token = self.peek()?;
        Ok(token.location)
    }

    /// Returns the location of the last eaten token
    ///
    /// If there is no last token, it will return the location of the next token
    pub fn get_location(&mut self) -> ParsingResult<TokenLocation> {
        let token = self.last_token.clone();
        match token {
            Some(token) => Ok(token.location),
            None => self.get_next_location(),
        }
    }

    pub fn parse_expr(&mut self, precedence: u8) -> ParsingResult<Expr> {
        let lhs_token = self.eat()?;
        let start = lhs_token.location;

        let lhs_kind = match lhs_token.kind {
            TokenKind::Literal(n) => {
                let kind = match n {
                    TokenLiteralKind::Number(n) => match n {
                        NumberKind::Int(n) => LiteralKind::Int(n),
                        NumberKind::Float(n) => LiteralKind::Float(n),
                    },
                    TokenLiteralKind::String(s) => LiteralKind::String(s),
                };
                ExprKind::Literal(kind)
            }
            TokenKind::Ident(ident) => {
                if ident == "true" || ident == "false" {
                    ExprKind::Literal(LiteralKind::Bool(ident == "true"))
                } else {
                    match self.peek()?.kind {
                        TokenKind::Symbol(SymbolKind::LParen) => {
                            let lparen = self.eat()?;
                            let mut args = Vec::new();
                            loop {
                                let token = self.peek()?;
                                match token.kind {
                                    TokenKind::Symbol(SymbolKind::RParen) => {
                                        break;
                                    }
                                    TokenKind::Symbol(SymbolKind::Comma) => {
                                        let _ = self.eat()?;
                                    }
                                    _ => {
                                        let expr = self.parse_expr(0)?;
                                        args.push(expr);
                                    }
                                }
                            }
                            let rparen = self.eat()?;
                            ExprKind::Call {
                                name: Identifier {
                                    value: ident,
                                    span: NodeLocation::new(
                                        lhs_token.location,
                                        lhs_token.location,
                                    ),
                                },
                                args: CallArgs {
                                    args,
                                    span: NodeLocation::new(
                                        lparen.location,
                                        rparen.location,
                                    ),
                                },
                            }
                        }
                        TokenKind::Symbol(SymbolKind::LBrace) => {
                            let _ = self.eat()?;
                            let mut fields = Vec::new();
                            loop {
                                let token = self.peek()?;
                                match token.kind {
                                    TokenKind::Symbol(SymbolKind::RBrace) => {
                                        let _ = self.eat()?;
                                        break;
                                    }
                                    TokenKind::Symbol(SymbolKind::Comma) => {
                                        let _ = self.eat()?;
                                    }
                                    _ => {
                                        let field_name = self.eat()?;
                                        let field_name = match field_name.kind {
                                            TokenKind::Ident(s) => s,
                                            _ => {
                                                return Err(ParsingError::UnexpectedToken(
                                                    field_name,
                                                ))
                                            }
                                        };
                                        let _colon = self.eat()?;
                                        if let TokenKind::Symbol(SymbolKind::Colon) = _colon.kind {
                                        } else {
                                            return Err(ParsingError::UnexpectedToken(_colon));
                                        }
                                        let expr = self.parse_expr(0)?;
                                        fields.push((field_name, expr));
                                    }
                                }
                            }
                            ExprKind::StructInit {
                                name: Identifier {
                                    value: ident,
                                    span: NodeLocation::new(start, start),
                                },
                                fields,
                            }
                        }
                        _ => ExprKind::Var(Identifier {
                            value: ident,
                            span: NodeLocation::new(start, start),
                        }),
                    }
                }
            }
            TokenKind::Symbol(ref symbol) => match symbol {
                SymbolKind::LParen => {
                    let expr = self.parse_expr(0)?;
                    let _ = self.eat()?;
                    expr.kind
                }
                SymbolKind::Minus => {
                    let rhs = self.parse_expr(6)?;
                    ExprKind::UnaryOp {
                        kind: UnaryOpKind::Neg,
                        rhs: Box::new(rhs),
                    }
                }
                SymbolKind::Not => {
                    let rhs = self.parse_expr(6)?;
                    ExprKind::UnaryOp {
                        kind: UnaryOpKind::Not,
                        rhs: Box::new(rhs),
                    }
                }
                SymbolKind::At => {
                    let ty = self.parse_type()?;
                    let rhs = self.parse_expr(6)?;
                    ExprKind::UnaryOp {
                        kind: UnaryOpKind::Cast(ty),
                        rhs: Box::new(rhs),
                    }
                }
                _ => return Err(ParsingError::UnexpectedToken(lhs_token)),
            },
        };

        let mut lhs = Expr {
            kind: lhs_kind,
            span: NodeLocation::new(start, self.get_location()?),
        };

        // Left hand recursion

        loop {
            let rhs_token = self.peek()?;
            if rhs_token.kind == TokenKind::Symbol(SymbolKind::Semicolon) {
                break;
            }
            let op = match rhs_token.kind {
                TokenKind::Symbol(SymbolKind::Dot) => {
                    let _dot = self.eat()?;
                    let field_token = self.eat()?;
                    let identifier = match field_token.kind {
                        TokenKind::Ident(s) => s,
                        _ => return Err(ParsingError::UnexpectedToken(field_token)),
                    };
                    lhs = Expr {
                        kind: ExprKind::FieldAccess {
                            lhs: Box::new(lhs),
                            field: Identifier {
                                value: identifier,
                                span: NodeLocation::new(
                                    field_token.location,
                                    field_token.location,
                                ),
                            },
                        },
                        span: NodeLocation::new(start, self.get_location()?),
                    };
                    continue;
                }
                TokenKind::Symbol(ref symbol) => match BinOpKind::try_from(symbol.clone()) {
                    Ok(op) => op,
                    Err(_) => break,
                },
                _ => break,
            };

            let rhs_precedence = op.precedence();
            if rhs_precedence < precedence {
                break;
            }

            let _ = self.eat()?;
            let rhs = self.parse_expr(rhs_precedence)?;
            lhs = Expr {
                kind: ExprKind::BinOp(BinOp {
                    kind: op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }),
                span: NodeLocation::new(start, self.get_location()?),
            };
        }

        Ok(lhs)
    }

    pub fn parse_assignment(&mut self, assignee: Expr) -> ParsingResult<Statement> {
        let equals = self.eat()?;
        let start = assignee.span.start;
        let kind = match equals.kind {
            TokenKind::Symbol(SymbolKind::Walrus) => {
                let value = self.parse_expr(0)?;
                let span = assignee.span.merge(&value.span);
                StatementKind::Decl(Assignment {
                    assignee,
                    value,
                    span,
                })
            }
            TokenKind::Symbol(SymbolKind::AssignEq) => {
                let value = self.parse_expr(0)?;
                let span = assignee.span.merge(&value.span);
                StatementKind::Assign(Assignment {
                    assignee,
                    value,
                    span,
                })
            }
            _ => return Err(ParsingError::UnexpectedToken(equals)),
        };
        Ok(Statement {
            kind,
            span: NodeLocation::new(start, self.get_location()?),
        })
    }

    pub fn parse_ident(&mut self) -> ParsingResult<Statement> {
        let start = self.get_next_location()?;
        let value = {
            let token = self.peek()?;
            match token.kind.clone() {
                TokenKind::Ident(s) => s,
                _ => return Err(ParsingError::UnexpectedToken(token.clone())),
            }
        };
        match value.as_str() {
            "if" => {
                let _if = self.eat()?;
                let _lparen = self.peek_and_assert(TokenKind::Symbol(SymbolKind::LParen))?;
                let cond = self.parse_expr(0)?;
                let then = self.parse_block()?;
                let or = if let Ok(token) = self.peek() {
                    if token.kind == TokenKind::Ident("else".to_string()) {
                        self.eat()?;
                        Some(self.parse_block()?)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let kind = StatementKind::If {
                    cond: Box::new(cond),
                    then,
                    or,
                };
                Ok(Statement {
                    kind,
                    span: NodeLocation::new(start, self.get_location()?),
                })
            }
            "loop" => {
                let _if = self.eat()?;
                let body = self.parse_block()?;
                let kind = StatementKind::Loop { body };
                Ok(Statement {
                    kind,
                    span: NodeLocation::new(start, self.get_location()?),
                })
            }
            "break" => {
                let _ = self.eat()?;
                Ok(Statement {
                    kind: StatementKind::Break,
                    span: NodeLocation::new(start, self.get_location()?),
                })
            }
            "return" => {
                let _ = self.eat()?;
                let value = if let Ok(token) = self.peek() {
                    if token.kind == TokenKind::Symbol(SymbolKind::Semicolon) {
                        None
                    } else {
                        Some(self.parse_expr(0)?)
                    }
                } else {
                    None
                };

                // Thank you clippy lol           vvvvvvvv
                let kind = StatementKind::Return(value.map(Box::new));
                Ok(Statement {
                    kind,
                    span: NodeLocation::new(start, self.get_location()?),
                })
            }
            _ => {
                let expr = self.parse_expr(0)?;
                match self.peek()?.kind {
                    TokenKind::Symbol(SymbolKind::AssignEq)
                    | TokenKind::Symbol(SymbolKind::Walrus) => self.parse_assignment(expr),
                    _ => {
                        let kind = StatementKind::Expr(expr);
                        Ok(Statement {
                            kind,
                            span: NodeLocation::new(start, self.get_location()?),
                        })
                    }
                }
            }
        }
    }

    pub fn parse_type(&mut self) -> ParsingResult<Type> {
        let token = self.eat()?;
        let kind = match &token.kind {
            TokenKind::Ident(s) => match s.as_str() {
                "int" => TypeKind::Int,
                "float" => TypeKind::Float,
                "bool" => TypeKind::Bool,
                "string" => TypeKind::String,
                _ => TypeKind::Struct(s.to_owned()),
            },
            _ => return Err(ParsingError::UnexpectedToken(token)),
        };
        Ok(Type {
            kind,
            span: NodeLocation::new(token.location, token.location),
        })
    }

    pub fn parse_field(&mut self) -> ParsingResult<Field> {
        let name = self.eat()?;
        let name = match name.kind {
            TokenKind::Ident(s) => Identifier {
                value: s,
                span: NodeLocation::new(name.location, name.location),
            },
            _ => return Err(ParsingError::UnexpectedToken(name)),
        };
        let colon = self.eat()?;
        if let TokenKind::Symbol(SymbolKind::Colon) = colon.kind {
        } else {
            return Err(ParsingError::UnexpectedToken(colon));
        }
        let ty = self.parse_type()?;
        let span = name.span.merge(&ty.span);
        Ok(Field { name, ty, span })
    }

    pub fn parse_fields(&mut self) -> ParsingResult<Fields> {
        let lbrace = self.eat_and_assert(TokenKind::Symbol(SymbolKind::LBrace))?;
        let mut fields = Vec::new();
        loop {
            match self.peek()?.kind {
                TokenKind::Ident(_) => {
                    fields.push(self.parse_field()?);
                    self.eat_and_assert(TokenKind::Symbol(SymbolKind::Semicolon))?;
                }
                TokenKind::Symbol(SymbolKind::RBrace) => {
                    break;
                }
                _ => return Err(ParsingError::UnexpectedToken(self.peek()?.clone())),
            }
        }
        let rbrace = self.eat()?;
        Ok(Fields {
            fields,
            span: NodeLocation::new(lbrace.location, rbrace.location),
        })
    }

    pub fn parse_struct(&mut self) -> ParsingResult<Struct> {
        let name = self.eat()?;
        let name = match name.kind {
            TokenKind::Ident(s) => Identifier {
                value: s,
                span: NodeLocation::new(name.location, name.location),
            },
            _ => return Err(ParsingError::UnexpectedToken(name)),
        };
        let _coloncolon = self.eat_and_assert(TokenKind::Symbol(SymbolKind::ColonColon))?;
        let _struct = self.eat_and_assert(TokenKind::Ident("struct".to_string()))?;
        let fields = self.parse_fields()?;
        let span = name.span.merge(&fields.span);

        Ok(Struct { name, fields, span })
    }

    pub fn parse_item_kind(&mut self) -> ParsingResult<ItemKind> {
        let token_kind = self.peek()?.kind.clone();
        match token_kind {
            TokenKind::Ident(ident) => {
                if ident == "import" {
                    let _ = self.eat()?;
                    let token = self.eat()?;
                    match token.kind {
                        TokenKind::Literal(TokenLiteralKind::String(s)) => {
                            return Ok(ItemKind::Import(s))
                        }
                        _ => return Err(ParsingError::UnexpectedToken(token)),
                    }
                }
                let nth_1 = self.peek_nth(1)?;
                if let TokenKind::Symbol(SymbolKind::ColonColon) = nth_1.kind {
                    let nth_2 = self.peek_nth(2)?;
                    match &nth_2.kind {
                        TokenKind::Ident(ident) => {
                            if ident == "struct" {
                                Ok(ItemKind::Struct(self.parse_struct()?))
                            } else {
                                return Err(ParsingError::UnexpectedToken(nth_2.clone()));
                            }
                        }
                        TokenKind::Symbol(SymbolKind::LParen) => {
                            Ok(ItemKind::Function(self.parse_function()?))
                        }
                        _ => return Err(ParsingError::UnexpectedToken(nth_2.clone())),
                    }
                } else {
                    Err(ParsingError::UnexpectedToken(nth_1.clone()))
                }
            }
            _ => Err(ParsingError::UnexpectedToken(self.peek()?.clone())),
        }
    }

    pub fn parse_item(&mut self) -> ParsingResult<Item> {
        let kind = self.parse_item_kind()?;
        Ok(Item { kind })
    }

    pub fn parse_identifier(&mut self) -> ParsingResult<Identifier> {
        let token = self.eat()?;
        match token.kind {
            TokenKind::Ident(s) => Ok(Identifier {
                value: s,
                span: NodeLocation::new(token.location, token.location),
            }),
            _ => Err(ParsingError::UnexpectedToken(token)),
        }
    }

    pub fn parse_function_arg(&mut self) -> ParsingResult<Arg> {
        let name = self.parse_identifier()?;
        let colon = self.eat_and_assert(TokenKind::Symbol(SymbolKind::Colon))?;
        if let TokenKind::Symbol(SymbolKind::Colon) = colon.kind {
        } else {
            return Err(ParsingError::UnexpectedToken(colon));
        }
        let ty = self.parse_type()?;
        let span = name.span.merge(&ty.span);
        Ok(Arg { name, ty, span })
    }

    pub fn parse_function_args(&mut self) -> ParsingResult<Args> {
        let lparen = self.eat()?;
        if let TokenKind::Symbol(SymbolKind::LParen) = lparen.kind {
        } else {
            return Err(ParsingError::UnexpectedToken(lparen));
        }
        let mut args = Vec::new();
        let mut variadic = false;
        loop {
            let token = self.peek()?;
            match token.kind {
                TokenKind::Symbol(SymbolKind::RParen) => {
                    let _rparen = self.eat()?;
                    break;
                }
                TokenKind::Symbol(SymbolKind::Comma) => {
                    let _ = self.eat()?;
                }
                TokenKind::Symbol(SymbolKind::Ellipsis) => {
                    if variadic {
                        return Err(ParsingError::UnexpectedToken(token.clone()));
                    }
                    variadic = true;
                    let _ = self.eat()?;
                }
                _ => {
                    let arg = self.parse_function_arg()?;
                    args.push(arg);
                }
            }
        }
        Ok(Args {
            args,
            variadic,
            span: NodeLocation::new(lparen.location, self.get_location()?),
        })
    }

    pub fn parse_fn_name(&mut self) -> ParsingResult<Identifier> {
        let token = self.eat()?;
        match token.kind {
            TokenKind::Ident(s) => Ok(Identifier {
                value: s,
                span: NodeLocation::new(token.location, token.location),
            }),
            _ => Err(ParsingError::UnexpectedToken(token)),
        }
    }

    pub fn parse_fn_ret_ty(&mut self) -> ParsingResult<Option<Type>> {
        match self.peek()?.kind {
            TokenKind::Symbol(SymbolKind::RArrow) => self.eat()?,
            _ => return Ok(None),
        };
        let token = self.eat()?;
        let kind = match &token.kind {
            TokenKind::Ident(s) => match s.as_str() {
                "int" => TypeKind::Int,
                "float" => TypeKind::Float,
                "bool" => TypeKind::Bool,
                "string" => TypeKind::String,
                _ => TypeKind::Struct(s.to_owned()),
            },
            _ => return Err(ParsingError::UnexpectedToken(token)),
        };
        Ok(Some(Type {
            kind,
            span: NodeLocation::new(token.location, token.location),
        }))
    }

    pub fn parse_fn_sig(&mut self) -> ParsingResult<FunctionSignature> {
        let name = self.parse_fn_name()?;

        let _coloncolon = self.eat_and_assert(TokenKind::Symbol(SymbolKind::ColonColon))?;
        let args = self.parse_function_args()?;
        let ret_ty = self.parse_fn_ret_ty()?;
        Ok(FunctionSignature { name, args, ret_ty })
    }

    pub fn parse_function(&mut self) -> ParsingResult<Function> {
        let sig = self.parse_fn_sig()?;
        match self.peek()?.kind {
            TokenKind::Symbol(SymbolKind::Semicolon) => {
                let _ = self.eat()?;
                return Ok(Function { sig, body: None });
            }
            _ => {}
        }
        let body = self.parse_block()?;

        Ok(Function {
            sig,
            body: Some(body),
        })
    }

    pub fn parse_block(&mut self) -> ParsingResult<Block> {
        let lbrace = self.eat_and_assert(TokenKind::Symbol(SymbolKind::LBrace))?;
        let mut stmts = Vec::new();
        while let Ok(t) = self.peek() {
            match t.kind {
                TokenKind::Symbol(SymbolKind::RBrace) => break,
                TokenKind::Ident(_) => stmts.push(self.parse_ident()?),
                TokenKind::Symbol(SymbolKind::Semicolon) => {
                    let _ = self.eat()?;
                }
                _ => return Err(ParsingError::UnexpectedToken(t.clone())),
            }
        }

        let rbrace = self.eat_and_assert(TokenKind::Symbol(SymbolKind::RBrace))?;
        Ok(Block {
            stmts,
            span: NodeLocation::new(lbrace.location, rbrace.location),
        })
    }

    pub fn parse(self) -> ParsingResult<Ast> {
        let mut nodes = Vec::new();
        for item in self {
            nodes.push(item?);
        }
        Ok(Ast { nodes })
    }
}

impl<I: Iterator<Item = Token>> Iterator for Parser<I> {
    type Item = ParsingResult<Item>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.peek().map(|tok| tok.kind.clone()) {
            Err(_) => None,
            Ok(_) => Some(self.parse_item()),
        }
    }
}
