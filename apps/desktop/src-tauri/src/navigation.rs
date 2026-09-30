//! The webview may only display the bundled app. Any navigation to another origin (a link, a
//! redirect, `location = ...`) is refused before the webview starts it.

use tauri::Url;

/// True only for the bundled app's own origin: `tauri://localhost` (macOS, Linux) or
/// `http(s)://tauri.localhost` (Windows WebView2), on the default port, with no credentials.
/// Tauri itself admits its isolation iframe before this runs. Same-document hash changes (the
/// router) are not navigations and never reach it.
pub fn is_app_url(url: &Url) -> bool {
    let plain = url.port().is_none() && url.username().is_empty() && url.password().is_none();
    let host = url.host_str();
    plain
        && match url.scheme() {
            "tauri" => host == Some("localhost"),
            "http" | "https" => host == Some("tauri.localhost"),
            _ => false,
        }
}

#[cfg(test)]
mod tests {
    use super::is_app_url;
    use tauri::Url;

    fn allowed(s: &str) -> bool {
        is_app_url(&Url::parse(s).expect("test URLs are valid"))
    }

    #[test]
    fn the_bundled_app_is_allowed() {
        assert!(allowed("http://tauri.localhost/index.html"));
        assert!(allowed("http://tauri.localhost/#/settings"));
        assert!(allowed("https://tauri.localhost/"));
        assert!(allowed("tauri://localhost/index.html"));
    }

    #[test]
    fn every_other_origin_is_refused() {
        for url in [
            "https://example.com/",
            "http://127.0.0.1:8181/v1/models",
            "http://localhost/",
            "http://tauri.localhost.evil.example/",
            "http://evil.tauri.localhost/",
            "http://tauri.localhost:8080/",
            "http://user@tauri.localhost/",
            "http://user:pw@tauri.localhost/",
            "file:///C:/Windows/System32/",
            "javascript:alert(1)",
            "data:text/html,hi",
            "ipc://localhost/",
            "tauri://evil/",
            "tauri://localhost:1/",
        ] {
            assert!(!allowed(url), "{url} must be refused");
        }
    }
}
