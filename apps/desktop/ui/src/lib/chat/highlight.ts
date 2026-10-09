/**
 * Syntax highlighting as tokens (kind + text), rendered as text spans: no markup is generated.
 * Small and fast enough to re-run on every streamed chunk of a code block.
 */

export type TokenKind = "kw" | "str" | "num" | "com" | "fn" | "plain";
export interface Token {
  k: TokenKind;
  v: string;
}

const KEYWORDS: Record<string, readonly string[]> = {
  js: ["const", "let", "var", "function", "return", "if", "else", "for", "while", "of", "in", "new", "class", "extends", "import", "export", "from", "async", "await", "try", "catch", "throw", "true", "false", "null", "undefined", "typeof", "interface", "type"],
  py: ["def", "return", "if", "elif", "else", "for", "while", "in", "import", "from", "as", "class", "try", "except", "raise", "with", "lambda", "None", "True", "False", "and", "or", "not", "async", "await", "yield", "pass"],
  rs: ["fn", "let", "mut", "pub", "struct", "enum", "impl", "trait", "use", "mod", "match", "if", "else", "for", "while", "loop", "return", "self", "Self", "true", "false", "Some", "None", "Ok", "Err", "async", "await", "move", "where", "const", "static", "unsafe"],
  sh: ["if", "then", "else", "fi", "for", "do", "done", "while", "case", "esac", "function", "export", "echo", "cd", "set"],
  c: ["int", "char", "void", "return", "if", "else", "for", "while", "struct", "static", "const", "unsigned", "include", "define", "auto", "class", "public", "private", "template", "namespace", "using"],
};

const ALIAS: Record<string, string> = {
  javascript: "js", ts: "js", typescript: "js", tsx: "js", jsx: "js", json: "js",
  python: "py", rust: "rs", bash: "sh", shell: "sh", powershell: "sh", ps1: "sh", zsh: "sh",
  cpp: "c", "c++": "c", h: "c", hpp: "c", cs: "c", java: "c", go: "c",
};

export function langFamily(lang: string): string | null {
  const l = ALIAS[lang] ?? lang;
  return KEYWORDS[l] ? l : null;
}

export function highlight(code: string, lang: string): Token[] {
  const fam = langFamily(lang);
  if (!fam) return [{ k: "plain", v: code }];
  const kw = new Set(KEYWORDS[fam]);
  const hashComment = fam === "py" || fam === "sh";
  const re = hashComment
    ? /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"?|'(?:[^'\\\n]|\\.)*'?)|(\b\d[\d_.]*\b)|([A-Za-z_]\w*)(?=\s*\()|([A-Za-z_]\w*)/g
    : /(\/\/[^\n]*|\/\*[\s\S]*?(?:\*\/|$))|("(?:[^"\\\n]|\\.)*"?|'(?:[^'\\\n]|\\.)*'?|`(?:[^`\\]|\\.)*`?)|(\b\d[\d_.xa-fA-F]*\b)|([A-Za-z_]\w*)(?=\s*\()|([A-Za-z_]\w*)/g;
  const out: Token[] = [];
  let last = 0;
  for (let m = re.exec(code); m; m = re.exec(code)) {
    if (m.index > last) out.push({ k: "plain", v: code.slice(last, m.index) });
    const [all, com, str, num, fn, word] = m;
    if (com) out.push({ k: "com", v: all });
    else if (str) out.push({ k: "str", v: all });
    else if (num) out.push({ k: "num", v: all });
    else if (fn) out.push({ k: kw.has(fn) ? "kw" : "fn", v: all });
    else if (word) out.push({ k: kw.has(word) ? "kw" : "plain", v: all });
    last = m.index + all.length;
    if (all.length === 0) re.lastIndex++;
  }
  if (last < code.length) out.push({ k: "plain", v: code.slice(last) });
  return out;
}
