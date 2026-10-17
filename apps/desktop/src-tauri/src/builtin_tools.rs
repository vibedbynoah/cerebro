//! Built-in tools: the model's own file and command access, limited to the user's workspace folders.
//!
//! These tools are always offered to the model in Chat (server `cerebro`), alongside the MCP
//! servers' tools; the Tools screen gates each one through the usual allow / ask / deny rules.
//! Everything a built-in tool touches must live inside one of the folders the user added on the
//! Tools screen: every path is resolved (symlinks and `..` included) and then must land under a
//! folder root, so a model cannot read or write anything the user did not open to it.
//!
//! `run_command` spawns a real PowerShell in the first workspace folder. To make sure a command can
//! never leave processes behind — including processes the command itself starts — it runs inside a
//! Job Object with kill-on-close (crates/cerebro-job), terminated when it times out or ends.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::Manager;

use crate::engine::{Access, ToolContext, ToolHost};
use crate::toolcall::ToolSpec;

/// The server name the model sees these tools under (Chat shows "from cerebro").
pub const SERVER: &str = "cerebro";

/// Shown whenever a tool needs a workspace folder and the user has added none yet.
const NO_FOLDER: &str = "No workspace folder is set. Add one on the Tools screen first.";

/// Longest a `run_command` may take before its whole process tree is killed.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(60);
/// Most command output kept (stdout and stderr together); the rest is read and dropped.
const MAX_COMMAND_BYTES: usize = 16 * 1024;
/// Most bytes read from one file.
const MAX_FILE_BYTES: usize = 128 * 1024;
/// Most directory entries listed.
const MAX_ENTRIES: usize = 500;
/// Most search hits returned.
const MAX_MATCHES: usize = 100;
/// Most bytes read from one fetched page.
const MAX_PAGE_BYTES: usize = 2 * 1024 * 1024;
/// Most workspace folders.
const MAX_FOLDERS: usize = 32;
/// Longest one folder path may be.
const MAX_FOLDER_LEN: usize = 1024;
/// Bytes the whole search may read across files before it stops early.
const MAX_SEARCH_BYTES: usize = 32 * 1024 * 1024;

// ---- Workspace folders --------------------------------------------------------------------------

/// The folders the user has opened to the built-in tools, shared between the Tools screen commands
/// and the engine's tool host. Kept in memory here and saved to the app's config folder as JSON.
pub struct WorkspaceFolders {
    folders: Mutex<Vec<PathBuf>>,
}

/// The on-disk shape: `{"folders": ["C:\\one", "D:\\two"]}`.
#[derive(Default, Serialize, Deserialize)]
struct StoredFolders {
    #[serde(default)]
    folders: Vec<PathBuf>,
}

fn folders_file<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Option<PathBuf> {
    app.path().app_config_dir().ok().map(|d| d.join("workspace-folders.json"))
}

impl WorkspaceFolders {
    pub fn new(folders: Vec<PathBuf>) -> Self {
        Self { folders: Mutex::new(folders) }
    }

    /// Reads the saved folders (nothing on first run, or after the file is damaged).
    pub fn load<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Self {
        let folders = folders_file(app)
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice::<StoredFolders>(&b).ok())
            .map(|s| s.folders)
            .unwrap_or_default();
        Self::new(folders)
    }

    /// The folder roots. Empty means every built-in tool refuses.
    pub fn paths(&self) -> Vec<PathBuf> {
        self.folders.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The folders as display strings (no `\\?\` prefix).
    pub fn list(&self) -> Vec<String> {
        self.paths().iter().map(|p| display(p)).collect()
    }

    pub fn replace(&self, folders: Vec<PathBuf>) {
        *self.folders.lock().unwrap_or_else(|e| e.into_inner()) = folders;
    }

    /// Validates the Tools screen's folder list before anything is saved or replaced: each path
    /// must exist, be a folder, and be at most [`MAX_FOLDER_LEN`] characters. Duplicates collapse
    /// (one canonical entry each).
    pub fn checked(input: &[String]) -> Result<Vec<PathBuf>, String> {
        if input.len() > MAX_FOLDERS {
            return Err(format!("At most {MAX_FOLDERS} folders can be added."));
        }
        let mut out: Vec<PathBuf> = Vec::new();
        for raw in input {
            let raw = raw.trim();
            if raw.is_empty() {
                return Err("A folder path cannot be empty.".to_owned());
            }
            if raw.chars().count() > MAX_FOLDER_LEN {
                return Err(format!("A folder path cannot be longer than {MAX_FOLDER_LEN} characters."));
            }
            let path = PathBuf::from(raw);
            let meta = std::fs::metadata(&path).map_err(|_| format!("{raw} was not found."))?;
            if !meta.is_dir() {
                return Err(format!("{raw} is not a folder."));
            }
            let canonical = std::fs::canonicalize(&path).map_err(|_| format!("{raw} could not be opened."))?;
            if !out.iter().any(|p| same_path(p, &canonical)) {
                out.push(canonical);
            }
        }
        Ok(out)
    }
}

/// Saves the folder list atomically (temp file + rename), like the other settings files.
pub fn save_folders<R: tauri::Runtime>(app: &tauri::AppHandle<R>, folders: &[PathBuf]) -> Result<(), String> {
    let path = folders_file(app).ok_or("The app's config folder is not available.")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Could not save the folders: {e}."))?;
    }
    let data = serde_json::to_vec_pretty(&StoredFolders { folders: folders.to_vec() }).map_err(|e| format!("Could not save the folders: {e}."))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, data).map_err(|e| format!("Could not save the folders: {e}."))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("Could not save the folders: {e}."))
}

// ---- Tools --------------------------------------------------------------------------------------

/// The built-in tools, in the order the model sees them.
pub fn tools() -> Vec<ToolSpec> {
    let dir = serde_json::json!({ "type": "string", "description": "Path inside a workspace folder." });
    let obj = |props: serde_json::Value, required: &[&str]| serde_json::json!({ "type": "object", "properties": props, "required": required });
    vec![
        ToolSpec {
            server: SERVER.to_owned(),
            name: "read_file".to_owned(),
            description: "Read a text file from a workspace folder.".to_owned(),
            input_schema: obj(serde_json::json!({ "path": dir }), &["path"]),
        },
        ToolSpec {
            server: SERVER.to_owned(),
            name: "list_directory".to_owned(),
            description: "List the entries of a folder in the workspace.".to_owned(),
            input_schema: obj(serde_json::json!({ "path": dir }), &["path"]),
        },
        ToolSpec {
            server: SERVER.to_owned(),
            name: "search_files".to_owned(),
            description: "Find lines containing a pattern in the workspace files.".to_owned(),
            input_schema: obj(
                serde_json::json!({
                    "pattern": { "type": "string", "description": "Text every returned line must contain." },
                    "path": { "type": "string", "description": "Folder to search (default: every workspace folder)." },
                }),
                &["pattern"],
            ),
        },
        ToolSpec {
            server: SERVER.to_owned(),
            name: "write_file".to_owned(),
            description: "Write (create or replace) a text file in a workspace folder.".to_owned(),
            input_schema: obj(
                serde_json::json!({
                    "path": dir,
                    "content": { "type": "string", "description": "The file's new content." },
                }),
                &["path", "content"],
            ),
        },
        ToolSpec {
            server: SERVER.to_owned(),
            name: "edit_file".to_owned(),
            description: "Replace one exact occurrence of old_text in a file.".to_owned(),
            input_schema: obj(
                serde_json::json!({
                    "path": dir,
                    "old_text": { "type": "string", "description": "The text to replace; must appear exactly once." },
                    "new_text": { "type": "string", "description": "The replacement text." },
                }),
                &["path", "old_text", "new_text"],
            ),
        },
        ToolSpec {
            server: SERVER.to_owned(),
            name: "run_command".to_owned(),
            description: "Run a PowerShell command in the first workspace folder.".to_owned(),
            input_schema: obj(serde_json::json!({ "command": { "type": "string", "description": "The PowerShell command line." } }), &["command"]),
        },
        ToolSpec {
            server: SERVER.to_owned(),
            name: "fetch_url".to_owned(),
            description: "Fetch a web page as text.".to_owned(),
            input_schema: obj(serde_json::json!({ "url": { "type": "string", "description": "An http(s) address." } }), &["url"]),
        },
    ]
}

/// The same tools as plain JSON, in the shape `mcp_tools` returns (`server`, `name`, `description`,
/// `input_schema`), so the Tools screen can render both kinds with one component.
pub fn tools_json() -> serde_json::Value {
    serde_json::Value::Array(
        tools()
            .iter()
            .map(|t| {
                serde_json::json!({
                    "server": t.server,
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect(),
    )
}

/// The engine's tool host: built-in tools plus the MCP servers' tools, everything else untouched.
pub struct BuiltinTools {
    mcp: Arc<dyn ToolHost>,
    folders: Arc<WorkspaceFolders>,
    /// What each file looked like when a tool last read or wrote it (read-before-write).
    observed: Observed,
    /// A folder read_file may open besides the workspace: where long tool results are saved whole.
    spill_dir: Option<PathBuf>,
}

impl BuiltinTools {
    pub fn new(mcp: Arc<dyn ToolHost>, folders: Arc<WorkspaceFolders>) -> Self {
        Self { mcp, folders, observed: Observed::default(), spill_dir: None }
    }

    pub fn with_spill_dir(mut self, dir: PathBuf) -> Self {
        self.spill_dir = Some(dir);
        self
    }
}

/// Deletes saved tool results older than `age` (called at start-up).
pub fn prune_spills(dir: &Path, age: Duration) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|el| el > age);
        if old && e.path().extension().is_some_and(|x| x == "txt") {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// A file's version as last seen by a tool: its size and modification time (or absent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seen {
    Absent,
    Present(u64, Option<std::time::SystemTime>),
}

fn seen_now(path: &Path) -> Seen {
    match std::fs::metadata(path) {
        Ok(m) => Seen::Present(m.len(), m.modified().ok()),
        Err(_) => Seen::Absent,
    }
}

/// Files the tools have read or written this session, by normalized path.
#[derive(Default)]
pub struct Observed(Mutex<std::collections::HashMap<String, Seen>>);

impl Observed {
    fn record(&self, path: &Path) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).insert(norm(path), seen_now(path));
    }

    /// May `path` be changed now? A new file may always be created; an existing one must have been
    /// read, and not have changed since. The error texts end in the step that fixes them.
    fn check_write(&self, path: &Path) -> Result<(), String> {
        let now = seen_now(path);
        if now == Seen::Absent {
            return Ok(());
        }
        match self.0.lock().unwrap_or_else(|e| e.into_inner()).get(&norm(path)) {
            None => Err(format!("Changing \"{}\" requires reading it first — read the file, then retry.", display(path))),
            Some(seen) if *seen != now => Err(format!("\"{}\" changed since it was read — re-read the file, then retry.", display(path))),
            Some(_) => Ok(()),
        }
    }
}

/// Everything a built-in tool needs for one call.
struct Scope<'a> {
    folders: &'a [PathBuf],
    ctx: ToolContext,
    observed: Option<&'a Observed>,
    extra_read: Option<&'a Path>,
}

impl Scope<'_> {
    /// The path a tool may use: inside the workspace folders (or, for reads, the results folder),
    /// or anywhere when access is full.
    fn reach(&self, raw: &str, write: bool) -> Result<PathBuf, String> {
        if write && self.ctx.access == Access::ReadOnly {
            return Err(SANDBOX_READ_ONLY.to_owned());
        }
        if self.ctx.access == Access::Full {
            let raw = raw.trim();
            if raw.is_empty() {
                return Err("A path is required.".to_owned());
            }
            let input = Path::new(raw);
            let joined = if input.is_absolute() {
                input.to_path_buf()
            } else {
                self.folders.first().ok_or_else(|| format!("{raw} is relative and no workspace folder is set; give a full path."))?.join(input)
            };
            return resolve_ancestor(&joined);
        }
        match confined(self.folders, raw) {
            Ok(p) => Ok(p),
            Err(e) => {
                if let (false, Some(extra)) = (write, self.extra_read) {
                    let resolved = resolve_ancestor(Path::new(raw.trim()))?;
                    let root = std::fs::canonicalize(extra).unwrap_or_else(|_| extra.to_path_buf());
                    if is_within(&resolved, &root) {
                        return Ok(resolved);
                    }
                }
                Err(e)
            }
        }
    }

    fn before_write(&self, path: &Path) -> Result<(), String> {
        match (self.ctx.read_before_write, self.observed) {
            (true, Some(o)) => o.check_write(path),
            _ => Ok(()),
        }
    }

    fn saw(&self, path: &Path) {
        if let Some(o) = self.observed {
            o.record(path);
        }
    }
}

/// The refusal a write gets under read-only access, worded so a model knows not to retry.
const SANDBOX_READ_ONLY: &str = "[sandbox: file access denied under read-only mode] Tool access is Read only (Tools screen): files cannot be changed and commands cannot run. Do not retry; tell the user what you would change.";

impl ToolHost for BuiltinTools {
    fn tools(&self) -> Vec<ToolSpec> {
        let mut all = tools();
        all.extend(self.mcp.tools());
        all
    }

    fn call(&self, server: &str, tool: &str, arguments: serde_json::Value, ctx: ToolContext) -> Result<String, String> {
        if server == SERVER {
            let folders = self.folders.paths();
            let scope = Scope { folders: &folders, ctx, observed: Some(&self.observed), extra_read: self.spill_dir.as_deref() };
            return run(tool, &arguments, &scope);
        }
        self.mcp.call(server, tool, arguments, ctx)
    }
}

/// One built-in tool by name, with the workspace folders current at call time.
fn run(tool: &str, args: &serde_json::Value, scope: &Scope) -> Result<String, String> {
    match tool {
        "read_file" => read_file(scope, args),
        "list_directory" => list_directory(scope, args),
        "search_files" => search_files(scope, args),
        "write_file" => write_file(scope, args),
        "edit_file" => edit_file(scope, args),
        "run_command" => run_command(scope, args, COMMAND_TIMEOUT),
        "fetch_url" => fetch_url(args, scope.ctx.offline, scope.ctx.allow_local_fetch),
        other => Err(format!("There is no built-in tool named {other}.")),
    }
}

/// A required non-empty string argument.
fn text_arg(args: &serde_json::Value, name: &str) -> Result<String, String> {
    match args.get(name).and_then(serde_json::Value::as_str) {
        Some(s) if !s.trim().is_empty() => Ok(s.to_owned()),
        _ => Err(format!("{name} must be a non-empty string.")),
    }
}

/// An argument that must be a string but may be empty (file contents, replacement text).
fn raw_text_arg(args: &serde_json::Value, name: &str) -> Result<String, String> {
    match args.get(name).and_then(serde_json::Value::as_str) {
        Some(s) => Ok(s.to_owned()),
        None => Err(format!("{name} must be a string.")),
    }
}

// ---- Path confinement ---------------------------------------------------------------------------

/// A path as the user sees it: canonical Windows paths carry a `\\?\` prefix that means nothing
/// to them.
fn display(p: &Path) -> String {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        rest.to_owned()
    } else {
        s.into_owned()
    }
}

/// Case-insensitive comparison form (Windows paths differ only by case).
fn norm(p: &Path) -> String {
    let s = p.to_string_lossy().replace('/', "\\");
    if cfg!(windows) {
        s.to_lowercase()
    } else {
        s
    }
}

fn same_path(a: &Path, b: &Path) -> bool {
    norm(a) == norm(b)
}

/// The resolved path is strictly inside the root (a sibling whose name merely starts with the
/// root's name does not count).
fn is_within(path: &Path, root: &Path) -> bool {
    let (p, r) = (norm(path), norm(root));
    p.len() > r.len() && p.starts_with(&r) && p.as_bytes()[r.len()] == b'\\'
}

/// Resolves a path that may not exist yet: the nearest existing ancestor is canonicalized (which
/// resolves symlinks) and the missing tail is re-appended. A `..` in the missing tail is refused,
/// because it cannot be resolved safely.
fn resolve_ancestor(path: &Path) -> Result<PathBuf, String> {
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = path.to_path_buf();
    loop {
        match std::fs::canonicalize(&cur) {
            Ok(base) => {
                let mut out = base;
                for part in tail.iter().rev() {
                    out.push(part);
                }
                return Ok(out);
            }
            Err(_) => {
                let Some(name) = cur.file_name() else {
                    return Err(format!("{} is not a valid path.", display(path)));
                };
                if name == ".." {
                    return Err(format!("{} is not a valid path.", display(path)));
                }
                tail.push(name.to_owned());
                if !cur.pop() {
                    return Err(format!("{} is not a valid path.", display(path)));
                }
            }
        }
    }
}

/// Resolves `raw` (relative paths start in the first workspace folder) and checks it lands inside
/// one of the folded roots — the single gate every path-taking tool goes through.
fn confined(folders: &[PathBuf], raw: &str) -> Result<PathBuf, String> {
    if folders.is_empty() {
        return Err(NO_FOLDER.to_owned());
    }
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("A path is required.".to_owned());
    }
    let input = Path::new(raw);
    let joined = if input.is_absolute() { input.to_path_buf() } else { folders[0].join(input) };
    let resolved = resolve_ancestor(&joined)?;
    for f in folders {
        let root = std::fs::canonicalize(f).unwrap_or_else(|_| f.clone());
        if same_path(&resolved, &root) || is_within(&resolved, &root) {
            return Ok(resolved);
        }
    }
    let list = folders.iter().map(|f| display(f)).collect::<Vec<_>>().join(", ");
    Err(format!("{raw} is outside the workspace folders ({list})."))
}

/// The path relative to its workspace folder, with forward slashes for the model; the full path
/// when nothing matches.
fn rel_to(folders: &[PathBuf], path: &Path) -> String {
    for f in folders {
        for root in [f.clone(), std::fs::canonicalize(f).unwrap_or_else(|_| f.clone())] {
            if let Ok(rel) = path.strip_prefix(&root) {
                let s = rel.to_string_lossy().replace('\\', "/");
                if !s.is_empty() {
                    return s;
                }
            }
        }
    }
    display(path).replace('\\', "/")
}

// ---- File tools ---------------------------------------------------------------------------------

/// Reads up to `max` bytes; `Ok((text, cut))` with `cut` true when the file was longer. Refuses
/// binary files (a NUL byte in the first `max` bytes).
fn read_limited(path: &Path, max: usize) -> Result<(String, bool), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Could not read {}: {e}.", display(path)))?;
    let cut = bytes.len() > max;
    let bytes = &bytes[..bytes.len().min(max)];
    if bytes.contains(&0) {
        return Err(format!("{} is not a text file.", display(path)));
    }
    Ok((String::from_utf8_lossy(bytes).into_owned(), cut))
}

fn read_file(scope: &Scope, args: &serde_json::Value) -> Result<String, String> {
    let path = scope.reach(&text_arg(args, "path")?, false)?;
    let (text, cut) = read_limited(&path, MAX_FILE_BYTES)?;
    scope.saw(&path);
    let mut out = text;
    if cut {
        out.push_str("\n[... cut at 128 KB]");
    }
    Ok(out)
}

fn list_directory(scope: &Scope, args: &serde_json::Value) -> Result<String, String> {
    let path = scope.reach(&text_arg(args, "path")?, false)?;
    let entries = std::fs::read_dir(&path).map_err(|e| format!("Could not list {}: {e}.", display(&path)))?;
    let mut names: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        names.push(if dir { format!("{name}/") } else { name });
    }
    names.sort();
    if names.is_empty() {
        return Ok("(empty folder)".to_owned());
    }
    let more = names.len().saturating_sub(MAX_ENTRIES);
    names.truncate(MAX_ENTRIES);
    if more > 0 {
        names.push(format!("[... {more} more entries]"));
    }
    Ok(names.join("\n"))
}

fn search_files(scope: &Scope, args: &serde_json::Value) -> Result<String, String> {
    let folders = scope.folders;
    let pattern = text_arg(args, "pattern")?;
    let roots = match args.get("path").and_then(serde_json::Value::as_str) {
        Some(p) if !p.trim().is_empty() => vec![scope.reach(p, false)?],
        _ if folders.is_empty() => return Err(NO_FOLDER.to_owned()),
        _ => folders.to_vec(),
    };
    let mut budget = MAX_SEARCH_BYTES;
    let mut hits: Vec<String> = Vec::new();
    let mut stop: Option<&str> = None;
    let mut stack: Vec<PathBuf> = roots.iter().rev().cloned().collect();
    'walk: while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_symlink() {
                continue;
            }
            if kind.is_dir() {
                stack.push(path);
                continue;
            }
            if budget == 0 {
                stop = Some("[... stopped: the search budget was reached]");
                break 'walk;
            }
            let Ok(data) = std::fs::read(&path) else { continue };
            let take = data.len().min(MAX_FILE_BYTES).min(budget);
            budget -= take;
            let bytes = &data[..take];
            if bytes.contains(&0) {
                continue;
            }
            let text = String::from_utf8_lossy(bytes);
            for (i, line) in text.lines().enumerate() {
                if line.contains(&pattern) {
                    let short: String = line.trim().chars().take(200).collect();
                    hits.push(format!("{}:{}: {short}", rel_to(folders, &path), i + 1));
                    if hits.len() >= MAX_MATCHES {
                        stop = Some("[... stopped: 100 matches were found]");
                        break 'walk;
                    }
                }
            }
        }
    }
    if hits.is_empty() {
        return Ok(format!("No line matches \"{pattern}\"."));
    }
    if let Some(note) = stop {
        hits.push(note.to_owned());
    }
    Ok(hits.join("\n"))
}

fn write_file(scope: &Scope, args: &serde_json::Value) -> Result<String, String> {
    let path = scope.reach(&text_arg(args, "path")?, true)?;
    let content = raw_text_arg(args, "content")?;
    scope.before_write(&path)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Could not create {}: {e}.", display(parent)))?;
    }
    std::fs::write(&path, content.as_bytes()).map_err(|e| format!("Could not write {}: {e}.", display(&path)))?;
    scope.saw(&path);
    Ok(format!("Wrote {} bytes to {}.", content.len(), display(&path)))
}

fn edit_file(scope: &Scope, args: &serde_json::Value) -> Result<String, String> {
    let path = scope.reach(&text_arg(args, "path")?, true)?;
    let old = text_arg(args, "old_text")?;
    let new = raw_text_arg(args, "new_text")?;
    scope.before_write(&path)?;
    let (text, cut) = read_limited(&path, MAX_FILE_BYTES)?;
    if cut {
        return Err(format!("{} is larger than 128 KB, too large to edit precisely.", display(&path)));
    }
    match text.matches(&old).count() {
        0 => return Err(format!("old_text was not found in {}.", display(&path))),
        1 => {}
        n => return Err(format!("old_text appears {n} times in {}; include more of the text so it matches exactly once.", display(&path))),
    }
    let updated = text.replacen(&old, &new, 1);
    std::fs::write(&path, updated.as_bytes()).map_err(|e| format!("Could not write {}: {e}.", display(&path)))?;
    scope.saw(&path);
    Ok(format!("Edited {}.", display(&path)))
}

// ---- run_command --------------------------------------------------------------------------------

/// A command's stdout and stderr, merged into one bounded buffer by two reader threads. The pipe
/// keeps being drained after the cap so a chatty command can never block on a full pipe.
struct PipeOut {
    data: Mutex<Vec<u8>>,
    cut: AtomicBool,
    readers: AtomicUsize,
}

impl PipeOut {
    fn new(readers: usize) -> Self {
        Self { data: Mutex::new(Vec::new()), cut: AtomicBool::new(false), readers: AtomicUsize::new(readers) }
    }

    fn push(&self, chunk: &[u8]) {
        let mut data = self.data.lock().unwrap_or_else(|e| e.into_inner());
        let room = MAX_COMMAND_BYTES.saturating_sub(data.len());
        let take = room.min(chunk.len());
        data.extend_from_slice(&chunk[..take]);
        if take < chunk.len() {
            self.cut.store(true, Ordering::SeqCst);
        }
    }

    /// The bytes read so far, waiting up to `grace` for the reader threads to finish (the pipes
    /// close when every process holding them — the job is terminated at this point — is gone).
    fn drain(&self, grace: Duration) -> Vec<u8> {
        let deadline = Instant::now() + grace;
        while self.readers.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        self.data.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

fn spawn_reader<R: Read + Send + 'static>(mut pipe: R, out: Arc<PipeOut>) {
    std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match pipe.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => out.push(&buf[..n]),
            }
        }
        out.readers.fetch_sub(1, Ordering::SeqCst);
    });
}

/// Runs one PowerShell command. The `timeout` is a parameter (not only the constant above) so the
/// tests can exercise the kill path quickly.
#[cfg(windows)]
fn run_command(scope: &Scope, args: &serde_json::Value, timeout: Duration) -> Result<String, String> {
    use std::os::windows::process::CommandExt;

    if scope.ctx.access == Access::ReadOnly {
        return Err(SANDBOX_READ_ONLY.to_owned());
    }
    let cwd = match (scope.folders.first(), scope.ctx.access) {
        (Some(f), _) => f.clone(),
        (None, Access::Full) => std::env::var_os("USERPROFILE").map(PathBuf::from).ok_or_else(|| NO_FOLDER.to_owned())?,
        (None, _) => return Err(NO_FOLDER.to_owned()),
    };
    let command = text_arg(args, "command")?;
    // Windows PowerShell writes redirected output in the console code page unless told otherwise;
    // UTF-8 without BOM keeps non-ASCII text intact for the model.
    let script = format!("$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false);\n{command}");
    let mut cmd = std::process::Command::new("powershell.exe");
    cmd.arg("-NoProfile")
        .arg("-NonInteractive")
        .arg("-Command")
        .arg(&script)
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // CREATE_NO_WINDOW: no console window may flash while the engine runs a command.
        .creation_flags(0x0800_0000);
    if scope.ctx.scrub_env {
        for (k, _) in std::env::vars_os() {
            if k.to_str().is_some_and(looks_secret) {
                cmd.env_remove(&k);
            }
        }
    }
    let job = cerebro_job::Job::create().map_err(|e| format!("Could not create the process containment job: {e}."))?;
    let mut child = cmd.spawn().map_err(|e| format!("Could not start the command: {e}"))?;
    if let Err(e) = job.assign(&child) {
        let _ = child.kill();
        return Err(format!("Could not contain the command: {e}."));
    }
    let out = Arc::new(PipeOut::new(2));
    if let Some(stdout) = child.stdout.take() {
        spawn_reader(stdout, out.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_reader(stderr, out.clone());
    }
    let started = Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(e) => {
                let _ = job.terminate();
                return Err(format!("Could not wait for the command: {e}."));
            }
        }
        if started.elapsed() >= timeout {
            timed_out = true;
            break None;
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    if timed_out {
        let _ = job.terminate();
        let _ = child.kill();
    }
    let _ = child.wait();
    // Even a clean exit can leave processes the command started; the job ends them all before the
    // readers are waited on, so their pipes close and no orphan survives.
    let _ = job.terminate();
    let data = out.drain(Duration::from_secs(2));
    let mut text = String::from_utf8_lossy(&data).trim_end().to_owned();
    if timed_out {
        return Err(format!("The command did not finish within {} s and was stopped.\n{text}", timeout.as_secs()).trim_end().to_owned());
    }
    if out.cut.load(Ordering::SeqCst) {
        text.push_str("\n[... output cut at 16 KB]");
    }
    if let Some(status) = status {
        if !status.success() {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&format!("[exit code {}]", status.code().unwrap_or(-1)));
        }
    }
    if text.is_empty() {
        text = "(no output)".to_owned();
    }
    Ok(text)
}

/// Environment variable names that usually hold credentials (API keys, tokens, passwords). A
/// command a model runs does not inherit them unless the user turns "Hide secrets from tools" off.
pub fn looks_secret(name: &str) -> bool {
    let n = name.to_ascii_uppercase();
    ["KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "PRIVATE", "SESSION", "COOKIE"].iter().any(|w| n.contains(w))
}

#[cfg(not(windows))]
fn run_command(_scope: &Scope, _args: &serde_json::Value, _timeout: Duration) -> Result<String, String> {
    Err("Running commands is available on Windows only.".to_owned())
}

// ---- fetch_url ----------------------------------------------------------------------------------

/// An address a model's fetch may reach: the public internet only, unless the user allowed local
/// ones. Loopback, private, link-local (cloud metadata), CGNAT, unique-local and unspecified
/// addresses are this computer or its network.
fn public_ip(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            !(v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_unspecified() || v4.is_broadcast() || v4.is_multicast() || o[0] == 0 || (o[0] == 100 && (o[1] & 0xc0) == 64))
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return public_ip(IpAddr::V4(v4));
            }
            let s = v6.segments()[0];
            !(v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() || (s & 0xfe00) == 0xfc00 || (s & 0xffc0) == 0xfe80)
        }
    }
}

/// Resolves `url`'s host and returns the address to connect to, refusing local ones unless
/// allowed. Every address the name resolves to must pass, so a name cannot mix in a local one.
fn checked_address(url: &reqwest::Url, allow_local: bool) -> Result<std::net::SocketAddr, String> {
    use std::net::ToSocketAddrs;
    let host = url.host_str().ok_or_else(|| format!("{url} has no host."))?;
    let port = url.port_or_known_default().unwrap_or(80);
    let addrs: Vec<std::net::SocketAddr> = (host.trim_start_matches('[').trim_end_matches(']'), port).to_socket_addrs().map_err(|e| format!("Could not find {host}: {e}."))?.collect();
    let first = *addrs.first().ok_or_else(|| format!("Could not find {host}."))?;
    if !allow_local && addrs.iter().any(|a| !public_ip(a.ip())) {
        return Err(format!("{host} is on this computer or your local network, which web fetches may not reach (Settings > Protections)."));
    }
    Ok(first)
}

/// Fetches one http(s) page as text: non-text pages are refused, HTML is reduced to readable text,
/// and the body is capped. Offline mode refuses before any connection is made.
fn fetch_url(args: &serde_json::Value, offline: bool, allow_local: bool) -> Result<String, String> {
    if offline {
        return Err("Offline mode is on, so no web page can be fetched.".to_owned());
    }
    let url = text_arg(args, "url")?;
    let parsed = reqwest::Url::parse(&url).map_err(|_| format!("{url} is not a URL."))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!("{url} is not an http(s) URL."));
    }
    // Redirects are followed here, one hop at a time, so every hop's address is checked and the
    // connection is pinned to the checked address (a second DNS answer cannot swap it).
    let mut current = parsed;
    let mut resp = None;
    for _ in 0..=5 {
        let addr = checked_address(&current, allow_local)?;
        let host = current.host_str().unwrap_or_default().to_owned();
        let client = reqwest::blocking::Client::builder()
            .user_agent(concat!("cerebro/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::none())
            .resolve(&host, addr)
            .build()
            .map_err(|e| format!("Could not start the request: {e}."))?;
        let r = client.get(current.clone()).send().map_err(|e| format!("Could not fetch {current}: {e}."))?;
        if r.status().is_redirection() {
            let loc = r.headers().get(reqwest::header::LOCATION).and_then(|v| v.to_str().ok()).ok_or_else(|| format!("{current} redirected without an address."))?;
            let next = current.join(loc).map_err(|_| format!("{current} redirected to an invalid address."))?;
            if !matches!(next.scheme(), "http" | "https") {
                return Err(format!("{url} redirected to a non-http address."));
            }
            current = next;
            continue;
        }
        resp = Some(r);
        break;
    }
    let mut resp = resp.ok_or_else(|| format!("{url} redirected more than 5 times."))?;
    let final_url = resp.url().clone();
    if !resp.status().is_success() {
        return Err(format!("{final_url} answered {} ({}).", resp.status().as_u16(), resp.status().canonical_reason().unwrap_or("")));
    }
    let mime = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let is_text = mime.is_empty() || mime.starts_with("text/") || mime == "application/json" || mime == "application/xml";
    if !is_text {
        return Err(format!("{final_url} is {mime}, not a text page."));
    }
    let mut body = Vec::new();
    let mut buf = [0u8; 8192];
    let mut cut = false;
    loop {
        let n = resp.read(&mut buf).map_err(|e| format!("Could not read {final_url}: {e}."))?;
        if n == 0 {
            break;
        }
        let room = MAX_PAGE_BYTES.saturating_sub(body.len());
        let take = room.min(n);
        body.extend_from_slice(&buf[..take]);
        if take < n {
            cut = true;
            break;
        }
    }
    let page = String::from_utf8_lossy(&body).into_owned();
    let page = if mime == "text/html" { html_to_text(&page) } else { page };
    let mut text = page.trim().to_owned();
    if cut {
        text.push_str("\n[... cut at 2 MB]");
    }
    Ok(text)
}

/// Tags pushed as their own line, so text runs between them stay on one line.
const BLOCK_TAGS: &[&str] = &[
    "p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header", "footer", "blockquote", "pre", "table", "ul", "ol", "nav", "main", "aside", "hr", "title",
];

/// A small HTML-to-text pass (no parser dependency): drops script/style/head bodies, turns block
/// tags into line breaks and decodes the entities that appear in ordinary pages.
fn html_to_text(html: &str) -> String {
    let mut text = String::with_capacity(html.len().min(MAX_PAGE_BYTES));
    let mut rest = html;
    while let Some(start) = rest.find('<') {
        text.push_str(&decode_entities(&rest[..start]));
        let tail = &rest[start..];
        let Some(end) = tail.find('>') else {
            rest = "";
            break;
        };
        let tag = &tail[1..end];
        let lname = tag.trim_start_matches('/').split_whitespace().next().unwrap_or("").trim_end_matches('/').to_ascii_lowercase();
        if !tag.starts_with('/') && matches!(lname.as_str(), "script" | "style" | "head") {
            let close = format!("</{lname}>");
            match tail.to_ascii_lowercase().find(&close) {
                Some(c) => {
                    rest = &tail[c + close.len()..];
                    continue;
                }
                None => {
                    rest = "";
                    break;
                }
            }
        }
        if BLOCK_TAGS.contains(&lname.as_str()) {
            text.push('\n');
        }
        rest = &tail[end + 1..];
    }
    text.push_str(&decode_entities(rest));
    collapse(&text)
}

/// Collapses HTML's arbitrary whitespace: single spaces inside lines, single blank lines between
/// blocks.
fn collapse(s: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in s.lines() {
        let joined = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if joined.is_empty() {
            if lines.last().is_some_and(|l| !l.is_empty()) {
                lines.push(String::new());
            }
        } else {
            lines.push(joined);
        }
    }
    lines.join("\n").trim().to_owned()
}

/// The handful of named entities common pages use, plus `&#NNN;` / `&#xHH;`. Unknown entities stay
/// as written.
fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let Some(end) = tail[..tail.len().min(34)].find(';') else {
            out.push('&');
            rest = &tail[1..];
            continue;
        };
        let name = &tail[1..end];
        let decoded = match name {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            "mdash" => Some('—'),
            "ndash" => Some('–'),
            "hellip" => Some('…'),
            "copy" => Some('©'),
            "reg" => Some('®'),
            "trade" => Some('™'),
            "deg" => Some('°'),
            "middot" => Some('·'),
            "laquo" => Some('«'),
            "raquo" => Some('»'),
            "times" => Some('×'),
            "divide" => Some('÷'),
            "eacute" => Some('é'),
            "egrave" => Some('è'),
            "agrave" => Some('à'),
            "ccedil" => Some('ç'),
            "uuml" => Some('ü'),
            "ouml" => Some('ö'),
            "auml" => Some('ä'),
            "szlig" => Some('ß'),
            _ => {
                if let Some(hex) = name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
                    u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
                } else if let Some(dec) = name.strip_prefix('#') {
                    dec.parse::<u32>().ok().and_then(char::from_u32)
                } else {
                    None
                }
            }
        };
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &tail[end + 1..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

// ---- Tests --------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Default settings, no read-before-write bookkeeping: the tools' own behaviour.
    fn ws(folders: &[PathBuf]) -> Scope<'_> {
        Scope { folders, ctx: ToolContext::default(), observed: None, extra_read: None }
    }

    /// A unique temp folder, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!("cerebro-builtin-{tag}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
            std::fs::create_dir_all(&dir).expect("temp dir");
            TempDir(std::fs::canonicalize(&dir).expect("canonicalize"))
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn tool_names_are_stable_and_complete() {
        let specs = tools();
        let names: Vec<&str> = specs.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["read_file", "list_directory", "search_files", "write_file", "edit_file", "run_command", "fetch_url"]);
        assert!(tools().iter().all(|t| t.server == SERVER && t.input_schema["type"] == "object"));
        let json = tools_json();
        assert_eq!(json.as_array().map(Vec::len), Some(7));
        assert_eq!(json[0]["server"], SERVER);
        assert_eq!(json[0]["name"], "read_file");
    }

    #[test]
    fn paths_outside_the_folders_are_refused() {
        let root = TempDir::new("confine");
        assert_eq!(confined(&[], "notes.txt").unwrap_err(), NO_FOLDER);
        let err = confined(&[root.0.clone()], "../elsewhere.txt").unwrap_err();
        assert!(err.contains("outside the workspace"), "{err}");
        let escape = root.0.join("..").join("escape.txt");
        assert!(confined(&[root.0.clone()], &escape.to_string_lossy()).is_err());
        // A path that should exist but does not is still confined to the folder (so write_file can
        // create it), while the same name with a `..` tail is refused.
        let inside = confined(&[root.0.clone()], "sub/new.txt").unwrap();
        assert!(is_within(&inside, &root.0));
        assert!(confined(&[root.0.clone()], "sub/../../up.txt").is_err());
    }

    #[cfg(windows)]
    #[test]
    fn symlinks_cannot_escape_the_folders() {
        let root = TempDir::new("symlink");
        let outside = TempDir::new("symlink-out");
        std::fs::write(outside.0.join("secret.txt"), "secret").expect("write");
        if std::os::windows::fs::symlink_dir(&outside.0, root.0.join("link")).is_err() {
            eprintln!("skipping: creating symlinks needs Developer Mode");
            return;
        }
        let target = root.0.join("link").join("secret.txt");
        let err = confined(&[root.0.clone()], &target.to_string_lossy()).unwrap_err();
        assert!(err.contains("outside the workspace"), "{err}");
    }

    #[test]
    fn write_list_read_roundtrip() {
        let root = TempDir::new("roundtrip");
        let folders = [root.0.clone()];
        let wrote = write_file(&ws(&folders), &serde_json::json!({ "path": "notes/hello.txt", "content": "héllo world" })).unwrap();
        assert!(wrote.starts_with("Wrote 12 bytes to"), "{wrote}");
        let listed = list_directory(&ws(&folders), &serde_json::json!({ "path": "notes" })).unwrap();
        assert_eq!(listed, "hello.txt");
        let read = read_file(&ws(&folders), &serde_json::json!({ "path": "notes/hello.txt" })).unwrap();
        assert_eq!(read, "héllo world");
        std::fs::create_dir(root.0.join("empty")).unwrap();
        assert_eq!(list_directory(&ws(&folders), &serde_json::json!({ "path": "empty" })).unwrap(), "(empty folder)");
    }

    #[test]
    fn edit_file_replaces_only_a_unique_match() {
        let root = TempDir::new("edit");
        let folders = [root.0.clone()];
        std::fs::write(root.0.join("a.txt"), "one two one").unwrap();
        let err = edit_file(&ws(&folders), &serde_json::json!({ "path": "a.txt", "old_text": "one", "new_text": "1" })).unwrap_err();
        assert!(err.contains("2 times"), "{err}");
        let err = edit_file(&ws(&folders), &serde_json::json!({ "path": "a.txt", "old_text": "zero", "new_text": "1" })).unwrap_err();
        assert!(err.contains("was not found"), "{err}");
        let ok = edit_file(&ws(&folders), &serde_json::json!({ "path": "a.txt", "old_text": "two", "new_text": "2" })).unwrap();
        assert!(ok.starts_with("Edited"), "{ok}");
        assert_eq!(std::fs::read_to_string(root.0.join("a.txt")).unwrap(), "one 2 one");
    }

    #[test]
    fn search_reports_file_line_and_text() {
        let root = TempDir::new("search");
        let folders = [root.0.clone()];
        std::fs::write(root.0.join("a.txt"), "alpha\nneedle here\n").unwrap();
        std::fs::create_dir(root.0.join("sub")).unwrap();
        std::fs::write(root.0.join("sub").join("b.txt"), "no match\nanother needle\n").unwrap();
        let out = search_files(&ws(&folders), &serde_json::json!({ "pattern": "needle" })).unwrap();
        assert!(out.contains("a.txt:2: needle here"), "{out}");
        assert!(out.contains("sub/b.txt:2: another needle"), "{out}");
        assert_eq!(search_files(&ws(&folders), &serde_json::json!({ "pattern": "zzz" })).unwrap(), "No line matches \"zzz\".");
    }

    #[test]
    fn long_files_are_cut_with_a_note() {
        let root = TempDir::new("cut");
        let folders = [root.0.clone()];
        std::fs::write(root.0.join("big.txt"), "a".repeat(MAX_FILE_BYTES + 1)).unwrap();
        let read = read_file(&ws(&folders), &serde_json::json!({ "path": "big.txt" })).unwrap();
        assert!(read.ends_with("[... cut at 128 KB]"), "{}", read.len());
    }

    #[cfg(windows)]
    #[test]
    fn run_command_output_exit_code_cwd_and_utf8() {
        let root = TempDir::new("cmd");
        let folders = [root.0.clone()];
        let out = run_command(&ws(&folders), &serde_json::json!({ "command": "Write-Output 'hello'; Write-Output 'café'" }), Duration::from_secs(30)).unwrap();
        assert!(out.contains("hello"), "{out}");
        assert!(out.contains("café"), "UTF-8 must survive the pipe: {out}");
        let fail = run_command(&ws(&folders), &serde_json::json!({ "command": "Write-Output 'bad'; exit 3" }), Duration::from_secs(30)).unwrap();
        assert!(fail.contains("bad") && fail.contains("[exit code 3]"), "{fail}");
        let cwd = run_command(&ws(&folders), &serde_json::json!({ "command": "(Get-Location).Path" }), Duration::from_secs(30)).unwrap();
        let want = display(&root.0).to_lowercase();
        assert!(cwd.to_lowercase().contains(&want), "{cwd} vs {want}");
    }

    #[cfg(windows)]
    #[test]
    fn run_command_timeout_kills_the_whole_tree() {
        let root = TempDir::new("kill");
        let folders = [root.0.clone()];
        // The grandchild writes marker.txt (in the command's working folder) after 4 s if it
        // survives; the parent then sleeps for two minutes.
        let cmd = "Start-Process cmd -ArgumentList '/c','ping -n 5 127.0.0.1 >nul & echo done> marker.txt' -WindowStyle Hidden; Start-Sleep 120";
        let started = Instant::now();
        let err = run_command(&ws(&folders), &serde_json::json!({ "command": cmd }), Duration::from_secs(2)).unwrap_err();
        assert!(err.contains("did not finish"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(15), "the timeout must kill the tree, not wait for it");
        std::thread::sleep(Duration::from_secs(5));
        assert!(!root.0.join("marker.txt").exists(), "a grandchild survived the job object");
    }

    #[test]
    fn fetch_url_offline_and_bad_urls_fail_without_network() {
        let err = fetch_url(&serde_json::json!({ "url": "https://example.com" }), true, false).unwrap_err();
        assert!(err.contains("Offline mode"), "{err}");
        let offline = ToolContext { offline: true, ..ToolContext::default() };
        let scope = Scope { folders: &[], ctx: offline, observed: None, extra_read: None };
        assert!(run("fetch_url", &serde_json::json!({ "url": "https://example.com" }), &scope).unwrap_err().contains("Offline mode"));
        let err = fetch_url(&serde_json::json!({ "url": "ftp://example.com/x" }), false, false).unwrap_err();
        assert!(err.contains("not an http"), "{err}");
        let err = fetch_url(&serde_json::json!({ "url": "no spaces allowed" }), false, false).unwrap_err();
        assert!(err.contains("is not a URL"), "{err}");
    }

    #[test]
    fn local_and_private_addresses_are_not_public() {
        let ip = |s: &str| s.parse::<std::net::IpAddr>().unwrap();
        for local in ["127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1", "fd00::1", "fe80::1", "::ffff:192.168.0.1", "224.0.0.1"] {
            assert!(!public_ip(ip(local)), "{local}");
        }
        for public in ["1.1.1.1", "93.184.216.34", "2606:4700::1111", "172.32.0.1", "100.128.0.1"] {
            assert!(public_ip(ip(public)), "{public}");
        }
    }

    /// The guard on real requests: this computer, the cloud metadata address and a LAN address are
    /// refused before any connection; turning the protection off lets a local page through.
    #[test]
    fn fetches_to_this_network_are_refused_unless_allowed() {
        for url in ["http://127.0.0.1:9/", "http://localhost/", "http://169.254.169.254/latest/meta-data/", "http://192.168.1.1/", "http://[::1]/"] {
            let err = fetch_url(&serde_json::json!({ "url": url }), false, false).unwrap_err();
            assert!(err.contains("local network"), "{url}: {err}");
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            use std::io::Write;
            if let Ok((mut s, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = std::io::Read::read(&mut s, &mut buf);
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\nConnection: close\r\n\r\nlocal");
            }
        });
        let url = format!("http://127.0.0.1:{port}/");
        assert!(fetch_url(&serde_json::json!({ "url": url }), false, false).is_err());
        assert_eq!(fetch_url(&serde_json::json!({ "url": url }), false, true).unwrap(), "local");
    }

    #[test]
    fn read_only_refuses_every_change_and_full_reaches_outside() {
        let root = TempDir::new("ro");
        let outside = TempDir::new("ro-outside");
        std::fs::write(outside.0.join("o.txt"), "outside").unwrap();
        let folders = vec![root.0.clone()];
        let ro = Scope { folders: &folders, ctx: ToolContext { access: Access::ReadOnly, ..ToolContext::default() }, observed: None, extra_read: None };
        for (tool, args) in [
            ("write_file", serde_json::json!({ "path": "a.txt", "content": "x" })),
            ("edit_file", serde_json::json!({ "path": "a.txt", "old_text": "x", "new_text": "y" })),
            ("run_command", serde_json::json!({ "command": "Write-Output hi" })),
        ] {
            assert!(run(tool, &args, &ro).unwrap_err().starts_with("[sandbox: file access denied under read-only mode]"), "{tool}");
        }
        assert!(!root.0.join("a.txt").exists());
        let outside_path = outside.0.join("o.txt").to_string_lossy().into_owned();
        assert!(read_file(&ws(&folders), &serde_json::json!({ "path": outside_path })).is_err(), "workspace access stays inside");
        let full = Scope { folders: &folders, ctx: ToolContext { access: Access::Full, ..ToolContext::default() }, observed: None, extra_read: None };
        assert_eq!(read_file(&full, &serde_json::json!({ "path": outside_path })).unwrap(), "outside");
    }

    #[test]
    fn writes_need_a_fresh_read_first() {
        let root = TempDir::new("rbw");
        let folders = vec![root.0.clone()];
        let observed = Observed::default();
        let scope = Scope { folders: &folders, ctx: ToolContext::default(), observed: Some(&observed), extra_read: None };
        // A new file may be created without reading; then it counts as seen.
        write_file(&scope, &serde_json::json!({ "path": "new.txt", "content": "one" })).unwrap();
        edit_file(&scope, &serde_json::json!({ "path": "new.txt", "old_text": "one", "new_text": "two" })).unwrap();
        // A file nobody read may not be replaced or edited.
        std::fs::write(root.0.join("old.txt"), "keep").unwrap();
        let err = write_file(&scope, &serde_json::json!({ "path": "old.txt", "content": "lost" })).unwrap_err();
        assert!(err.ends_with("— read the file, then retry."), "{err}");
        assert_eq!(std::fs::read_to_string(root.0.join("old.txt")).unwrap(), "keep");
        read_file(&scope, &serde_json::json!({ "path": "old.txt" })).unwrap();
        // Changed by someone else after the read.
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(root.0.join("old.txt"), "keep, edited elsewhere").unwrap();
        let err = edit_file(&scope, &serde_json::json!({ "path": "old.txt", "old_text": "keep", "new_text": "x" })).unwrap_err();
        assert!(err.ends_with("— re-read the file, then retry."), "{err}");
        read_file(&scope, &serde_json::json!({ "path": "old.txt" })).unwrap();
        edit_file(&scope, &serde_json::json!({ "path": "old.txt", "old_text": "keep", "new_text": "kept" })).unwrap();
        assert_eq!(std::fs::read_to_string(root.0.join("old.txt")).unwrap(), "kept, edited elsewhere");
        // The protection is a setting.
        std::fs::write(root.0.join("free.txt"), "a").unwrap();
        let off = Scope { ctx: ToolContext { read_before_write: false, ..ToolContext::default() }, ..scope };
        write_file(&off, &serde_json::json!({ "path": "free.txt", "content": "b" })).unwrap();
    }

    #[test]
    fn results_folder_is_readable_but_not_writable() {
        let root = TempDir::new("spill-ws");
        let spill = TempDir::new("spill-dir");
        std::fs::write(spill.0.join("r1.txt"), "whole result").unwrap();
        let folders = vec![root.0.clone()];
        let scope = Scope { folders: &folders, ctx: ToolContext::default(), observed: None, extra_read: Some(&spill.0) };
        let p = spill.0.join("r1.txt").to_string_lossy().into_owned();
        assert_eq!(read_file(&scope, &serde_json::json!({ "path": p })).unwrap(), "whole result");
        assert!(write_file(&scope, &serde_json::json!({ "path": p, "content": "x" })).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn commands_do_not_inherit_secrets_unless_allowed() {
        let root = TempDir::new("env");
        let folders = vec![root.0.clone()];
        std::env::set_var("CEREBRO_TEST_API_KEY", "sk-should-not-leak");
        let cmd = serde_json::json!({ "command": "Write-Output \"[$env:CEREBRO_TEST_API_KEY]\"" });
        assert_eq!(run_command(&ws(&folders), &cmd, Duration::from_secs(30)).unwrap(), "[]");
        let shown = Scope { folders: &folders, ctx: ToolContext { scrub_env: false, ..ToolContext::default() }, observed: None, extra_read: None };
        assert_eq!(run_command(&shown, &cmd, Duration::from_secs(30)).unwrap(), "[sk-should-not-leak]");
        assert!(looks_secret("GITHUB_TOKEN") && looks_secret("aws_secret_access_key") && !looks_secret("PATH") && !looks_secret("USERPROFILE"));
    }

    #[test]
    fn html_becomes_readable_text() {
        let html = "<html><head><title>t</title><style>p{}</style></head><body><h1>Hi &amp; bye</h1><p>one</p><p>two<br>three</p><script>var x=1;</script><p>&#233;t&#233; &#x2713; &unknown;</p></body></html>";
        let text = html_to_text(html);
        assert!(text.starts_with("Hi & bye"), "{text}");
        assert!(text.contains("one\n\ntwo\nthree"), "{text}");
        assert!(!text.contains("var x"), "{text}");
        assert!(text.contains("été ✓ &unknown;"), "{text}");
    }

    #[test]
    fn folder_validation_rejects_bad_input() {
        let root = TempDir::new("checked");
        std::fs::write(root.0.join("file.txt"), "x").unwrap();
        let missing = root.0.join("nope").to_string_lossy().into_owned();
        assert!(WorkspaceFolders::checked(&[missing]).is_err());
        let file = root.0.join("file.txt").to_string_lossy().into_owned();
        assert!(WorkspaceFolders::checked(&[file]).is_err());
        assert!(WorkspaceFolders::checked(&["   ".to_owned()]).is_err());
        let path = root.0.to_string_lossy().into_owned();
        let ok = WorkspaceFolders::checked(&[path.clone(), path]).unwrap();
        assert_eq!(ok.len(), 1, "duplicates collapse");
        assert!(ok[0].is_dir());
    }

    #[test]
    fn the_host_lists_builtins_first_and_routes_calls() {
        struct FakeMcp;
        impl ToolHost for FakeMcp {
            fn tools(&self) -> Vec<ToolSpec> {
                vec![ToolSpec { server: "mcp".into(), name: "x".into(), description: "d".into(), input_schema: serde_json::json!({ "type": "object" }) }]
            }
            fn call(&self, _server: &str, _tool: &str, _args: serde_json::Value, _ctx: ToolContext) -> Result<String, String> {
                Ok("mcp ran".to_owned())
            }
        }
        let root = TempDir::new("host");
        let host = BuiltinTools::new(Arc::new(FakeMcp), Arc::new(WorkspaceFolders::new(vec![root.0.clone()])));
        let names: Vec<String> = ToolHost::tools(&host).iter().map(|t| t.name.clone()).collect();
        assert_eq!(names.len(), 8);
        assert_eq!(names[0], "read_file");
        assert_eq!(names[7], "x");
        std::fs::write(root.0.join("a.txt"), "hi").unwrap();
        assert_eq!(host.call(SERVER, "read_file", serde_json::json!({ "path": "a.txt" }), ToolContext::default()).unwrap(), "hi");
        assert_eq!(host.call("mcp", "x", serde_json::json!({}), ToolContext::default()).unwrap(), "mcp ran");
    }
}
