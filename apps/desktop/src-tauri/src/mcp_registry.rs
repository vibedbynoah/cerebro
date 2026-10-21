//! The official MCP registry (registry.modelcontextprotocol.io) as a live source for the Tools
//! screen's "Browse servers": a search returns servers in the same shape as the bundled catalogue
//! (ui/src/lib/mcp-catalogue.ts), so a registry server is added with the same form. npm packages run
//! with `npx -y`, PyPI packages with `uvx`, container images with `docker run -i --rm`; hosted
//! servers are their URL. Only the search request leaves the computer, and only when the user
//! searches.

use serde::Serialize;
use serde_json::Value;
use std::time::Duration;

const REGISTRY: &str = "https://registry.modelcontextprotocol.io/v0/servers";

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum Install {
    Command { command: String, args: Vec<String> },
    Url { url: String },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EnvKey {
    pub name: String,
    pub label: String,
    pub secret: bool,
    pub url: String,
}

/// One registry server, shaped like a catalogue entry.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Entry {
    pub id: String,
    pub name: String,
    pub category: String,
    pub description: String,
    pub install: Install,
    pub env: Vec<EnvKey>,
    pub risk: String,
    pub offline: bool,
    pub homepage: String,
    pub version: String,
}

fn s<'a>(v: &'a Value, camel: &str, snake: &str) -> Option<&'a Value> {
    v.get(camel).or_else(|| v.get(snake))
}

fn text(v: &Value, camel: &str, snake: &str) -> String {
    s(v, camel, snake).and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// A short id for mcp.json from a registry name such as "io.github.owner/server-name".
fn short_id(name: &str) -> String {
    let last = name.rsplit('/').next().unwrap_or(name);
    let id: String = last.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c.to_ascii_lowercase() } else { '-' }).collect();
    id.trim_matches('-').chars().take(40).collect::<String>()
}

/// Every server in a registry page, in order; servers this app cannot start are skipped.
pub fn parse(page: &Value) -> Vec<Entry> {
    let items = page["servers"].as_array().cloned().unwrap_or_default();
    let mut out = Vec::new();
    for item in items {
        // Newer pages wrap each server as { server: {...}, _meta: {...} }.
        let srv = if item.get("server").is_some() { item["server"].clone() } else { item };
        let name = text(&srv, "name", "name");
        if name.is_empty() {
            continue;
        }
        let title = srv.get("title").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| name.rsplit('/').next().unwrap_or(&name).to_owned());
        let homepage = srv.get("repository").and_then(|r| r.get("url")).and_then(Value::as_str).or(srv.get("websiteUrl").and_then(Value::as_str)).unwrap_or("https://registry.modelcontextprotocol.io").to_owned();
        let version = text(&srv, "version", "version");
        let packages = s(&srv, "packages", "packages").and_then(Value::as_array).cloned().unwrap_or_default();
        let mut install = None;
        let mut env = Vec::new();
        for p in &packages {
            let kind = text(p, "registryType", "registry_type");
            let ident = text(p, "identifier", "name");
            let ver = text(p, "version", "version");
            let transport = p.get("transport").and_then(|t| t.get("type")).and_then(Value::as_str).unwrap_or("stdio");
            if ident.is_empty() || transport != "stdio" {
                continue;
            }
            let pinned = |sep: &str| if ver.is_empty() || ver == "latest" { ident.clone() } else { format!("{ident}{sep}{ver}") };
            install = match kind.as_str() {
                "npm" => Some(Install::Command { command: "npx".into(), args: vec!["-y".into(), pinned("@")] }),
                "pypi" => Some(Install::Command { command: "uvx".into(), args: vec![pinned("==")] }),
                "oci" | "docker" => Some(Install::Command { command: "docker".into(), args: vec!["run".into(), "-i".into(), "--rm".into(), pinned(":")] }),
                _ => None,
            };
            if install.is_some() {
                for e in s(p, "environmentVariables", "environment_variables").and_then(Value::as_array).into_iter().flatten() {
                    let n = text(e, "name", "name");
                    if n.is_empty() {
                        continue;
                    }
                    let d = text(e, "description", "description");
                    env.push(EnvKey { label: if d.is_empty() { n.clone() } else { d }, secret: s(e, "isSecret", "is_secret").and_then(Value::as_bool).unwrap_or(false), url: homepage.clone(), name: n });
                }
                break;
            }
        }
        if install.is_none() {
            install = srv.get("remotes").and_then(Value::as_array).and_then(|r| r.iter().find_map(|x| x.get("url").and_then(Value::as_str))).map(|u| Install::Url { url: u.to_owned() });
        }
        let Some(install) = install else { continue };
        let risk = if matches!(install, Install::Url { .. }) { "account" } else { "run" };
        out.push(Entry {
            id: short_id(&name),
            name: title,
            category: "Registry".into(),
            description: text(&srv, "description", "description").chars().take(300).collect(),
            install,
            env,
            risk: risk.into(),
            offline: false,
            homepage,
            version,
        });
    }
    out
}

/// Searches the registry (at most 50 results).
pub fn search(query: &str) -> Result<Vec<Entry>, String> {
    let q = query.trim();
    if q.chars().count() > 100 {
        return Err("Use a shorter search.".to_owned());
    }
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(20)).user_agent(concat!("Cerebro/", env!("CARGO_PKG_VERSION"))).build().map_err(|e| e.to_string())?;
    let url = reqwest::Url::parse_with_params(REGISTRY, &[("search", q), ("limit", "50")]).map_err(|e| e.to_string())?;
    let r = client.get(url).send().map_err(|e| if e.is_connect() || e.is_timeout() { "Could not reach the MCP registry. Check your connection.".to_owned() } else { format!("The MCP registry search failed: {e}") })?;
    if !r.status().is_success() {
        return Err(format!("The MCP registry answered with an error (HTTP {}).", r.status().as_u16()));
    }
    let page: Value = r.json().map_err(|_| "The MCP registry sent something this app could not read.".to_owned())?;
    Ok(parse(&page))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn current_and_older_page_shapes_both_parse() {
        let page = json!({ "servers": [
            { "server": { "name": "io.github.acme/weather-mcp", "title": "Weather", "description": "Forecasts.", "version": "1.2.0",
                "repository": { "url": "https://github.com/acme/weather-mcp" },
                "packages": [{ "registryType": "npm", "identifier": "@acme/weather-mcp", "version": "1.2.0", "transport": { "type": "stdio" },
                    "environmentVariables": [{ "name": "WEATHER_KEY", "description": "Your API key", "isSecret": true }] }] }, "_meta": {} },
            { "name": "io.github.x/py-tool", "description": "A Python one.", "version": "0.3.1",
                "packages": [{ "registry_type": "pypi", "identifier": "py-tool", "version": "0.3.1",
                    "environment_variables": [{ "name": "TOKEN", "is_secret": false }] }] },
            { "server": { "name": "com.example/hosted", "description": "Runs at the vendor.", "remotes": [{ "type": "streamable-http", "url": "https://mcp.example.com/mcp" }] } },
            { "server": { "name": "com.example/nothing-runnable", "packages": [{ "registryType": "nuget", "identifier": "X" }] } },
        ]});
        let e = parse(&page);
        assert_eq!(e.len(), 3);
        assert_eq!(e[0].id, "weather-mcp");
        assert_eq!(e[0].name, "Weather");
        assert_eq!(e[0].install, Install::Command { command: "npx".into(), args: vec!["-y".into(), "@acme/weather-mcp@1.2.0".into()] });
        assert_eq!(e[0].env[0], EnvKey { name: "WEATHER_KEY".into(), label: "Your API key".into(), secret: true, url: "https://github.com/acme/weather-mcp".into() });
        assert_eq!(e[1].install, Install::Command { command: "uvx".into(), args: vec!["py-tool==0.3.1".into()] });
        assert!(!e[1].env[0].secret);
        assert_eq!(e[2].install, Install::Url { url: "https://mcp.example.com/mcp".into() });
        assert_eq!(e[2].risk, "account");
    }

    #[test]
    fn ids_are_safe_for_mcp_json() {
        assert_eq!(short_id("io.github.Owner/My Server!"), "my-server");
        assert_eq!(short_id("plain"), "plain");
    }

    /// The real registry answers a search (needs the internet).
    /// `cargo test -p cerebro-desktop mcp_registry::tests::live -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_search_returns_servers() {
        let r = search("filesystem").expect("search");
        println!("{} servers, first: {:?}", r.len(), r.first().map(|e| (&e.name, &e.install)));
        assert!(!r.is_empty());
    }
}
