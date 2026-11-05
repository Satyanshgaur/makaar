//! LaTeX math → Office Math (OMML), so Word shows editable equations. Covers what notes use:
//! scripts, fractions, roots, `\left…\right`, accents, `\text`, matrices/cases/aligned, and
//! symbol commands. Anything else is an error and the caller falls back to the source.

#[derive(Debug, PartialEq)]
pub struct OmmlError(pub String);

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Cmd(String),
    Open,
    Close,
    Sup,
    Sub,
    Amp,
    Newline,
    Space,
    Char(char),
}

fn tokenize(s: &str) -> Vec<Tok> {
    let mut out = vec![];
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        out.push(match c {
            '\\' => match it.next() {
                Some(n) if n.is_ascii_alphabetic() => {
                    let mut name = String::from(n);
                    while let Some(&m) = it.peek().filter(|m| m.is_ascii_alphabetic()) {
                        name.push(m);
                        it.next();
                    }
                    Tok::Cmd(name)
                }
                Some('\\') => Tok::Newline,
                Some(n) => Tok::Cmd(n.to_string()),
                None => continue,
            },
            '{' => Tok::Open,
            '}' => Tok::Close,
            '^' => Tok::Sup,
            '_' => Tok::Sub,
            '&' => Tok::Amp,
            c if c.is_whitespace() => Tok::Space,
            c => Tok::Char(c),
        });
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
enum Node {
    /// Math run; `upright` for operator names, `text` for `\text{}` (normal text font).
    Run {
        s: String,
        upright: bool,
        text: bool,
    },
    Group(Vec<Node>),
    Scripts {
        base: Box<Node>,
        sub: Option<Box<Node>>,
        sup: Option<Box<Node>>,
    },
    Frac(Box<Node>, Box<Node>),
    Root(Option<Box<Node>>, Box<Node>),
    Delim(String, String, Box<Node>),
    Accent(char, Box<Node>),
    Bar(bool, Box<Node>),
    Over(Box<Node>, Box<Node>),
    Matrix(Vec<Vec<Node>>),
}

const SYMBOLS: &[(&str, &str)] = &[
    ("alpha", "α"),
    ("beta", "β"),
    ("gamma", "γ"),
    ("delta", "δ"),
    ("epsilon", "ϵ"),
    ("varepsilon", "ε"),
    ("zeta", "ζ"),
    ("eta", "η"),
    ("theta", "θ"),
    ("vartheta", "ϑ"),
    ("iota", "ι"),
    ("kappa", "κ"),
    ("lambda", "λ"),
    ("mu", "μ"),
    ("nu", "ν"),
    ("xi", "ξ"),
    ("pi", "π"),
    ("rho", "ρ"),
    ("sigma", "σ"),
    ("tau", "τ"),
    ("upsilon", "υ"),
    ("phi", "ϕ"),
    ("varphi", "φ"),
    ("chi", "χ"),
    ("psi", "ψ"),
    ("omega", "ω"),
    ("Gamma", "Γ"),
    ("Delta", "Δ"),
    ("Theta", "Θ"),
    ("Lambda", "Λ"),
    ("Xi", "Ξ"),
    ("Pi", "Π"),
    ("Sigma", "Σ"),
    ("Phi", "Φ"),
    ("Psi", "Ψ"),
    ("Omega", "Ω"),
    ("le", "≤"),
    ("leq", "≤"),
    ("ge", "≥"),
    ("geq", "≥"),
    ("ne", "≠"),
    ("neq", "≠"),
    ("lt", "<"),
    ("gt", ">"),
    ("approx", "≈"),
    ("equiv", "≡"),
    ("sim", "∼"),
    ("simeq", "≃"),
    ("cong", "≅"),
    ("propto", "∝"),
    ("ll", "≪"),
    ("gg", "≫"),
    ("leqslant", "⩽"),
    ("geqslant", "⩾"),
    ("nleq", "≰"),
    ("ngeq", "≱"),
    ("in", "∈"),
    ("notin", "∉"),
    ("ni", "∋"),
    ("subset", "⊂"),
    ("subseteq", "⊆"),
    ("supset", "⊃"),
    ("supseteq", "⊇"),
    ("cup", "∪"),
    ("cap", "∩"),
    ("emptyset", "∅"),
    ("varnothing", "∅"),
    ("forall", "∀"),
    ("exists", "∃"),
    ("neg", "¬"),
    ("lnot", "¬"),
    ("land", "∧"),
    ("wedge", "∧"),
    ("lor", "∨"),
    ("vee", "∨"),
    ("oplus", "⊕"),
    ("otimes", "⊗"),
    ("to", "→"),
    ("rightarrow", "→"),
    ("leftarrow", "←"),
    ("gets", "←"),
    ("Rightarrow", "⇒"),
    ("Leftarrow", "⇐"),
    ("Leftrightarrow", "⇔"),
    ("iff", "⇔"),
    ("implies", "⇒"),
    ("leftrightarrow", "↔"),
    ("mapsto", "↦"),
    ("longrightarrow", "⟶"),
    ("uparrow", "↑"),
    ("downarrow", "↓"),
    ("times", "×"),
    ("div", "÷"),
    ("pm", "±"),
    ("mp", "∓"),
    ("cdot", "⋅"),
    ("ast", "∗"),
    ("star", "⋆"),
    ("circ", "∘"),
    ("bullet", "∙"),
    ("infty", "∞"),
    ("partial", "∂"),
    ("nabla", "∇"),
    ("prime", "′"),
    ("angle", "∠"),
    ("degree", "°"),
    ("therefore", "∴"),
    ("because", "∵"),
    ("mid", "∣"),
    ("parallel", "∥"),
    ("perp", "⊥"),
    ("ldots", "…"),
    ("dots", "…"),
    ("cdots", "⋯"),
    ("vdots", "⋮"),
    ("ddots", "⋱"),
    ("sum", "∑"),
    ("prod", "∏"),
    ("int", "∫"),
    ("iint", "∬"),
    ("oint", "∮"),
    ("bigcup", "⋃"),
    ("bigcap", "⋂"),
    ("langle", "⟨"),
    ("rangle", "⟩"),
    ("lfloor", "⌊"),
    ("rfloor", "⌋"),
    ("lceil", "⌈"),
    ("rceil", "⌉"),
    ("vert", "|"),
    ("Vert", "‖"),
    ("lbrace", "{"),
    ("rbrace", "}"),
    ("{", "{"),
    ("}", "}"),
    ("|", "‖"),
    ("%", "%"),
    ("_", "_"),
    ("&", "&"),
    ("#", "#"),
    ("$", "$"),
    (",", "\u{2009}"),
    (";", "\u{2005}"),
    (":", "\u{2005}"),
    ("!", ""),
    (" ", " "),
    ("quad", "\u{2003}"),
    ("qquad", "\u{2003}\u{2003}"),
];

const FUNCTIONS: &[&str] = &[
    "log", "ln", "lg", "exp", "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos",
    "arctan", "sinh", "cosh", "tanh", "lim", "max", "min", "sup", "inf", "det", "gcd", "deg",
    "dim", "ker", "mod", "bmod", "Pr", "arg",
];

const ACCENTS: &[(&str, char)] = &[
    ("hat", '\u{302}'),
    ("widehat", '\u{302}'),
    ("bar", '\u{305}'),
    ("vec", '\u{20D7}'),
    ("dot", '\u{307}'),
    ("ddot", '\u{308}'),
    ("tilde", '\u{303}'),
    ("widetilde", '\u{303}'),
];

/// Commands that only change the font; their argument is kept as is.
const FONTS: &[&str] = &[
    "mathbf",
    "mathit",
    "mathcal",
    "mathbb",
    "mathsf",
    "mathtt",
    "mathfrak",
    "boldsymbol",
    "bm",
    "displaystyle",
    "textstyle",
    "scriptstyle",
    "limits",
    "nolimits",
    "big",
    "Big",
    "bigg",
    "Bigg",
    "bigl",
    "bigr",
    "Bigl",
    "Bigr",
];

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

fn err<T>(msg: impl Into<String>) -> Result<T, OmmlError> {
    Err(OmmlError(msg.into()))
}

impl Parser {
    fn skip_space(&mut self) {
        while self.toks.get(self.pos) == Some(&Tok::Space) {
            self.pos += 1;
        }
    }

    /// Next token, skipping spaces (insignificant in math mode).
    fn peek(&mut self) -> Option<&Tok> {
        self.skip_space();
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        self.skip_space();
        self.next_raw()
    }

    fn next_raw(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    /// Parses until `}`, `&`, `\\`, `\right`, `\end` or the end, leaving the stopper unread.
    fn row(&mut self) -> Result<Vec<Node>, OmmlError> {
        let mut out = vec![];
        loop {
            match self.peek() {
                None | Some(Tok::Close | Tok::Amp | Tok::Newline) => return Ok(out),
                Some(Tok::Cmd(c)) if c == "right" || c == "end" => return Ok(out),
                Some(Tok::Sup | Tok::Sub) => {
                    let base = out.pop().unwrap_or(Node::Group(vec![]));
                    out.push(self.scripts(base)?);
                }
                _ => {
                    if let Some(n) = self.atom()? {
                        out.push(n);
                    }
                }
            }
        }
    }

    fn scripts(&mut self, base: Node) -> Result<Node, OmmlError> {
        let (mut sub, mut sup) = (None, None);
        while let Some(t @ (Tok::Sup | Tok::Sub)) = self.peek().cloned() {
            self.next();
            let arg = Box::new(self.arg()?);
            let slot = if t == Tok::Sup { &mut sup } else { &mut sub };
            if slot.replace(arg).is_some() {
                return err("double script");
            }
        }
        Ok(Node::Scripts {
            base: Box::new(base),
            sub,
            sup,
        })
    }

    /// One argument: a `{group}` or a single atom.
    fn arg(&mut self) -> Result<Node, OmmlError> {
        match self.atom()? {
            Some(n) => Ok(n),
            None => self.arg(),
        }
    }

    fn group(&mut self) -> Result<Node, OmmlError> {
        let r = self.row()?;
        match self.next() {
            Some(Tok::Close) => Ok(Node::Group(r)),
            _ => err("unbalanced {"),
        }
    }

    /// Raw characters of a `{…}` argument (for `\text`, environment names), spaces kept.
    fn raw_arg(&mut self) -> Result<String, OmmlError> {
        if self.next() != Some(Tok::Open) {
            return err("expected {");
        }
        let mut s = String::new();
        let mut depth = 0;
        loop {
            match self.next_raw() {
                Some(Tok::Close) if depth == 0 => return Ok(s),
                Some(Tok::Close) => {
                    depth -= 1;
                    s.push('}');
                }
                Some(Tok::Open) => {
                    depth += 1;
                    s.push('{');
                }
                Some(Tok::Char(c)) => s.push(c),
                Some(Tok::Space | Tok::Newline) => s.push(' '),
                Some(Tok::Cmd(c)) => match SYMBOLS.iter().find(|(n, _)| *n == c) {
                    Some((_, u)) => s.push_str(u),
                    None => {
                        s.push('\\');
                        s.push_str(&c);
                    }
                },
                Some(Tok::Sup) => s.push('^'),
                Some(Tok::Sub) => s.push('_'),
                Some(Tok::Amp) => s.push('&'),
                None => return err("unbalanced {"),
            }
        }
    }

    fn delimiter(&mut self) -> Result<String, OmmlError> {
        match self.next() {
            Some(Tok::Char('.')) => Ok(String::new()),
            Some(Tok::Char(c)) => Ok(c.to_string()),
            Some(Tok::Cmd(c)) => SYMBOLS
                .iter()
                .find(|(n, _)| *n == c)
                .map(|(_, u)| u.to_string())
                .ok_or(OmmlError(format!("bad delimiter \\{c}"))),
            _ => err("missing delimiter"),
        }
    }

    fn atom(&mut self) -> Result<Option<Node>, OmmlError> {
        let Some(t) = self.next() else {
            return err("missing argument");
        };
        Ok(Some(match t {
            Tok::Open => self.group()?,
            Tok::Char(c) => run(c.to_string()),
            Tok::Close | Tok::Amp | Tok::Newline | Tok::Sup | Tok::Sub | Tok::Space => {
                return err("unexpected token")
            }
            Tok::Cmd(c) => return self.command(&c),
        }))
    }

    fn command(&mut self, c: &str) -> Result<Option<Node>, OmmlError> {
        Ok(Some(match c {
            "frac" | "dfrac" | "tfrac" | "cfrac" => {
                let n = self.arg()?;
                Node::Frac(Box::new(n), Box::new(self.arg()?))
            }
            "binom" => {
                let n = self.arg()?;
                let d = self.arg()?;
                let m = Node::Matrix(vec![vec![n], vec![d]]);
                Node::Delim("(".into(), ")".into(), Box::new(m))
            }
            "sqrt" => {
                let deg = if self.peek() == Some(&Tok::Char('[')) {
                    self.next();
                    let mut d = vec![];
                    while self.peek().is_some_and(|t| *t != Tok::Char(']')) {
                        if let Some(n) = self.atom()? {
                            d.push(n);
                        }
                    }
                    self.next();
                    Some(Box::new(Node::Group(d)))
                } else {
                    None
                };
                Node::Root(deg, Box::new(self.arg()?))
            }
            "text" | "textrm" | "textbf" | "textit" | "mbox" | "mathrm" | "operatorname" => {
                let s = self.raw_arg()?;
                if c == "mathrm" || c == "operatorname" {
                    Node::Run {
                        s,
                        upright: true,
                        text: false,
                    }
                } else {
                    Node::Run {
                        s,
                        upright: true,
                        text: true,
                    }
                }
            }
            "left" => {
                let open = self.delimiter()?;
                let body = self.row()?;
                if self.next() != Some(Tok::Cmd("right".into())) {
                    return err("\\left without \\right");
                }
                let close = self.delimiter()?;
                Node::Delim(open, close, Box::new(Node::Group(body)))
            }
            "begin" => return self.environment().map(Some),
            "overline" => Node::Bar(true, Box::new(self.arg()?)),
            "underline" => Node::Bar(false, Box::new(self.arg()?)),
            "overset" | "stackrel" => {
                let top = self.arg()?;
                Node::Over(Box::new(self.arg()?), Box::new(top))
            }
            "not" => {
                let mut n = self.arg()?;
                if let Node::Run { s, .. } = &mut n {
                    s.push('\u{338}');
                }
                n
            }
            "phantom" | "hphantom" | "vphantom" | "label" | "tag" => {
                self.arg()?;
                return Ok(None);
            }
            _ if FONTS.contains(&c) => return Ok(None),
            _ => {
                if let Some((_, ch)) = ACCENTS.iter().find(|(n, _)| *n == c) {
                    Node::Accent(*ch, Box::new(self.arg()?))
                } else if FUNCTIONS.contains(&c) {
                    Node::Run {
                        s: c.trim_start_matches('b').into(),
                        upright: true,
                        text: false,
                    }
                } else if let Some((_, u)) = SYMBOLS.iter().find(|(n, _)| *n == c) {
                    if u.is_empty() {
                        return Ok(None);
                    }
                    run(u.to_string())
                } else {
                    return err(format!("unknown command \\{c}"));
                }
            }
        }))
    }

    fn environment(&mut self) -> Result<Node, OmmlError> {
        let name = self.raw_arg()?;
        if name == "array" || name == "tabular" {
            self.raw_arg()?; // column spec
        }
        let mut rows = vec![];
        let mut row = vec![];
        loop {
            row.push(Node::Group(self.row()?));
            match self.next() {
                Some(Tok::Amp) => {}
                Some(Tok::Newline) => rows.push(std::mem::take(&mut row)),
                Some(Tok::Cmd(c)) if c == "end" => {
                    self.raw_arg()?;
                    break;
                }
                _ => return err(format!("unterminated {name}")),
            }
        }
        if row.iter().any(|n| *n != Node::Group(vec![])) {
            rows.push(row);
        }
        let m = Node::Matrix(rows);
        Ok(match name.trim_end_matches('*') {
            "pmatrix" => Node::Delim("(".into(), ")".into(), Box::new(m)),
            "bmatrix" => Node::Delim("[".into(), "]".into(), Box::new(m)),
            "vmatrix" => Node::Delim("|".into(), "|".into(), Box::new(m)),
            "cases" => Node::Delim("{".into(), String::new(), Box::new(m)),
            _ => m,
        })
    }
}

fn run(s: String) -> Node {
    Node::Run {
        s,
        upright: false,
        text: false,
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn write(n: &Node, out: &mut String) {
    let e = |n: &Node, out: &mut String| {
        out.push_str("<m:e>");
        write(n, out);
        out.push_str("</m:e>");
    };
    match n {
        Node::Run { s, upright, text } => {
            out.push_str("<m:r>");
            if *text {
                out.push_str("<m:rPr><m:nor/></m:rPr>");
            } else if *upright || !s.chars().all(char::is_alphabetic) {
                out.push_str("<m:rPr><m:sty m:val=\"p\"/></m:rPr>");
            }
            out.push_str(&format!(
                "<m:t xml:space=\"preserve\">{}</m:t></m:r>",
                esc(s)
            ));
        }
        Node::Group(v) => v.iter().for_each(|c| write(c, out)),
        Node::Scripts { base, sub, sup } => {
            let tag = match (sub, sup) {
                (Some(_), Some(_)) => "sSubSup",
                (Some(_), None) => "sSub",
                (None, Some(_)) => "sSup",
                (None, None) => return write(base, out),
            };
            out.push_str(&format!("<m:{tag}>"));
            e(base, out);
            for (name, part) in [("sub", sub), ("sup", sup)] {
                if let Some(p) = part {
                    out.push_str(&format!("<m:{name}>"));
                    write(p, out);
                    out.push_str(&format!("</m:{name}>"));
                }
            }
            out.push_str(&format!("</m:{tag}>"));
        }
        Node::Frac(a, b) => {
            out.push_str("<m:f><m:num>");
            write(a, out);
            out.push_str("</m:num><m:den>");
            write(b, out);
            out.push_str("</m:den></m:f>");
        }
        Node::Root(deg, body) => {
            out.push_str("<m:rad>");
            match deg {
                Some(d) => {
                    out.push_str("<m:deg>");
                    write(d, out);
                    out.push_str("</m:deg>");
                }
                None => out.push_str("<m:radPr><m:degHide m:val=\"1\"/></m:radPr><m:deg/>"),
            }
            e(body, out);
            out.push_str("</m:rad>");
        }
        Node::Delim(o, c, body) => {
            out.push_str(&format!(
                "<m:d><m:dPr><m:begChr m:val=\"{}\"/><m:endChr m:val=\"{}\"/></m:dPr>",
                esc(o).replace('"', "&quot;"),
                esc(c).replace('"', "&quot;")
            ));
            e(body, out);
            out.push_str("</m:d>");
        }
        Node::Accent(ch, body) => {
            out.push_str(&format!(
                "<m:acc><m:accPr><m:chr m:val=\"{ch}\"/></m:accPr>"
            ));
            e(body, out);
            out.push_str("</m:acc>");
        }
        Node::Bar(top, body) => {
            let pos = if *top { "top" } else { "bot" };
            out.push_str(&format!(
                "<m:bar><m:barPr><m:pos m:val=\"{pos}\"/></m:barPr>"
            ));
            e(body, out);
            out.push_str("</m:bar>");
        }
        Node::Over(body, top) => {
            out.push_str("<m:limUpp>");
            e(body, out);
            out.push_str("<m:lim>");
            write(top, out);
            out.push_str("</m:lim></m:limUpp>");
        }
        Node::Matrix(rows) => {
            out.push_str("<m:m>");
            for r in rows {
                out.push_str("<m:mr>");
                for c in r {
                    e(c, out);
                }
                out.push_str("</m:mr>");
            }
            out.push_str("</m:m>");
        }
    }
}

/// `<m:oMath>` for a LaTeX formula (without `$` or `\(` delimiters).
pub fn omml(latex: &str) -> Result<String, OmmlError> {
    let mut p = Parser {
        toks: tokenize(latex),
        pos: 0,
    };
    let mut rows = vec![];
    loop {
        rows.push(p.row()?);
        match p.next() {
            None => break,
            // top-level `\\` or `&` (aligned display math without an environment)
            Some(Tok::Newline) => {}
            Some(Tok::Amp) => rows.last_mut().unwrap().push(run(String::new())),
            _ => return err("unbalanced }"),
        }
    }
    let mut out = String::from("<m:oMath>");
    if rows.len() > 1 {
        write(
            &Node::Matrix(rows.into_iter().map(|r| vec![Node::Group(r)]).collect()),
            &mut out,
        );
    } else {
        write(&Node::Group(rows.pop().unwrap_or_default()), &mut out);
    }
    out.push_str("</m:oMath>");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(latex: &str) -> String {
        let s = omml(latex).unwrap();
        s["<m:oMath>".len()..s.len() - "</m:oMath>".len()].to_string()
    }

    #[test]
    fn runs_and_scripts() {
        assert_eq!(body("x"), "<m:r><m:t xml:space=\"preserve\">x</m:t></m:r>");
        let s = body("x_1^{2}");
        assert!(
            s.starts_with("<m:sSubSup><m:e>") && s.contains("<m:sub>") && s.contains("<m:sup>"),
            "{s}"
        );
        assert!(body("2^n").starts_with("<m:sSup>"));
    }

    #[test]
    fn structures() {
        assert!(body("\\frac{a}{b+c}").starts_with("<m:f><m:num>"));
        assert!(body("\\sqrt{x}").contains("degHide"));
        assert!(body("\\sqrt[3]{x}").contains("<m:deg><m:r>"));
        assert!(body("\\left( x \\right]").contains("begChr m:val=\"(\"/><m:endChr m:val=\"]\""));
        assert!(
            body("\\begin{cases} 1 & x>0 \\\\ 0 & \\text{else} \\end{cases}")
                .contains("<m:m><m:mr>")
        );
        assert!(body("\\text{for all } n")
            .contains("<m:nor/></m:rPr><m:t xml:space=\"preserve\">for all </m:t>"));
        assert!(body("a < b").contains("&lt;"));
        assert!(body("\\log n \\le \\Theta(n)").contains("Θ"));
        assert!(body("\\overline{AB}").contains("<m:bar>"));
    }

    #[test]
    fn errors_fall_back() {
        assert!(omml("\\frac{a}{b").is_err());
        assert!(omml("a}").is_err());
        assert!(omml("\\foo{x}").is_err());
        assert!(omml("x^1^2").is_err());
        assert!(omml("\\left( x").is_err());
    }
}
