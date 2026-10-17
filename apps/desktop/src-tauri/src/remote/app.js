/* Cerebro phone console: status dashboard and optional chat.
   One IIFE, nothing persisted except a "signed in before" flag. */
(function () {
  "use strict";

  /* Tuning. */
  const POLL_MS = 333;           // status poll while signed in
  const SIGNED_OUT_MS = 1000;    // gentler poll while the sign-in view shows
  const BACKOFF_MAX_MS = 5000;   // ceiling for the network backoff
  const STALE_MS = 3000;         // older than this and the link is suspect
  const PAINT_MS = 1000;         // per-second readouts
  const MAX_REQS = 8;
  const DASH = "\u2014";
  const SEEN_KEY = "cerebro.seen";

  const $ = function (id) { return document.getElementById(id); };

  const dom = {
    pill: $("pill"), pillText: $("pillText"), fresh: $("fresh"), live: $("live"),
    signIn: $("viewSignIn"), form: $("signInForm"), password: $("password"),
    device: $("device"), error: $("signInError"), retry: $("retryNote"),
    submit: $("signInSubmit"), pairHint: $("pairHint"),
    dash: $("viewDash"), signOut: $("signOut"),
    tabStatus: $("tabStatus"), tabChat: $("tabChat"),
    panelStatus: $("panelStatus"), panelChat: $("panelChat"),
    modelName: $("modelName"), modelQuant: $("modelQuant"),
    decode: $("decode"), prefill: $("prefill"),
    ctxText: $("ctxText"), ctxBar: $("ctxBar"),
    gpuUtil: $("gpuUtil"), gpuTemp: $("gpuTemp"), gpuPower: $("gpuPower"),
    vramText: $("vramText"), vramBar: $("vramBar"),
    cpu: $("cpu"), ramText: $("ramText"), ramBar: $("ramBar"), deviceName: $("deviceName"),
    requests: $("requests"), stopButton: $("stopButton"),
    chatLog: $("chatLog"), chatForm: $("chatForm"), chatInput: $("chatInput"),
    send: $("sendButton"), chatStop: $("chatStop"), chatNote: $("chatNote"),
    scrim: $("stopScrim"), stopCancel: $("stopCancel"), stopGo: $("stopConfirm")
  };

  const freshChat = function () {
    return { turns: [], id: null, from: 0, text: "", node: null, thinking: null, streamBusy: false, streamFails: 0 };
  };

  const state = {
    pair: null,
    signedIn: false,
    lastAt: 0,
    failures: 0,
    busy: false,
    checked: false,
    timer: 0,
    countdown: false,
    retryTimer: 0,
    reqSig: "",
    paintedAt: 0,
    returnFocus: null,
    pending: { decode: null, prefill: null, force: false },
    status: null,
    chat: freshChat()
  };

  /* ---------------------------------------------------------- helpers */

  function setText(node, text) {
    if (node && node.textContent !== text) node.textContent = text;
  }
  function show(node, visible) { if (node) node.hidden = !visible; }
  function announce(text) { setText(dom.live, text); }
  function missing(value) { return value === null || value === undefined; }

  // Bars read a custom property, written through the CSSOM rather than a
  // style attribute, so they work under the strict style-src policy.
  function setFill(node, ratio) {
    if (!node) return;
    const pct = Math.max(0, Math.min(100, (ratio || 0) * 100));
    node.style.setProperty("--fill", pct.toFixed(1) + "%");
  }
  function num(value, digits) {
    if (missing(value) || Number.isNaN(value)) return DASH;
    const d = digits || 0;
    return Number(value).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function pctText(value) { return missing(value) ? DASH : num(value, 0) + "%"; }
  function ago(ms) {
    const s = Math.max(0, Math.round((Date.now() - ms) / 1000));
    if (s < 60) return s + "s ago";
    if (s < 3600) return Math.round(s / 60) + "m ago";
    return Math.round(s / 3600) + "h ago";
  }
  function seenBefore() {
    try { return window.localStorage.getItem(SEEN_KEY) === "1"; } catch (err) { return false; }
  }
  function remember() {
    try { window.localStorage.setItem(SEEN_KEY, "1"); } catch (err) { /* private mode */ }
  }

  /* -------------------------------------------------------------- api */

  async function api(path, options) {
    const opts = options || {};
    opts.credentials = "same-origin";
    opts.cache = "no-store";
    opts.headers = Object.assign({ "X-Cerebro": "1" }, opts.headers || {});
    // Every POST is JSON, even an empty one: the server refuses any other POST (CSRF guard).
    if (opts.method === "POST") {
      opts.headers["Content-Type"] = "application/json";
      if (opts.body === undefined) opts.body = "{}";
    }
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch (err) { data = null; }
    if (!res.ok) {
      const err = new Error((data && data.error) || "Request failed (" + res.status + ")");
      err.status = res.status;
      if (data && data.retryAfterS) err.retryAfterS = data.retryAfterS;
      throw err;
    }
    return data;
  }

  function takePairCode() {
    const hash = window.location.hash || "";
    const found = /(?:^|[#&])pair=([^&]+)/.exec(hash);
    if (!found) return null;
    // Clear the address bar first so the code never sits in history.
    window.history.replaceState(null, "", "/");
    return decodeURIComponent(found[1] || "") || null;
  }

  function guessDevice() {
    const ua = navigator.userAgent || "";
    if (/iPhone|iPod/i.test(ua)) return "iPhone";
    if (/iPad/i.test(ua)) return "iPad";
    if (/Android/i.test(ua)) return /Mobile/i.test(ua) ? "Android phone" : "Android tablet";
    if (/Macintosh|Mac OS X/i.test(ua)) return "Mac";
    if (/Windows/i.test(ua)) return "Windows";
    if (/Linux/i.test(ua)) return "Linux";
    return "Phone";
  }

  /* -------------------------------------------------------- rendering */

  const ENGINE_LABEL = { idle: "Idle", loading: "Loading", generating: "Generating", error: "Error" };

  function onStatus(s) {
    state.status = s;
    state.lastAt = Date.now();
    setEngine(s.engine && s.engine.state ? s.engine.state : "idle");
    renderModel(s.model);
    state.pending.decode = s.decodeTps;
    state.pending.prefill = s.prefillTps;
    paintRates();
    renderContext(s.model);
    renderGpu(s.gpu);
    renderSystem(s);
    renderRequests(s.requests);
    syncChat(s.chatAllowed);
    renderStop(s.engine);
    paintFresh();
    if (state.failures > 0) {
      state.failures = 0;
      announce("Connected again.");
    }
  }

  function setEngine(name) {
    setText(dom.pillText, ENGINE_LABEL[name] || "Unknown");
    if (dom.pill.dataset.state !== name) dom.pill.dataset.state = name;
  }

  function renderModel(model) {
    setText(dom.modelName, model && model.name ? model.name : "No model loaded");
    setText(dom.modelQuant, model && model.quant ? model.quant : DASH);
  }

  /* Decode and prefill repaint once a second; the rest repaints live. */
  function paintRates() {
    const now = Date.now();
    if (!state.pending.force && now - state.paintedAt < PAINT_MS) return;
    state.pending.force = false;
    state.paintedAt = now;
    // One decimal below 100 tok/s, whole numbers above: 23,215 reads better than 23,215.0.
    const rate = function (v) { return num(v, !missing(v) && v >= 100 ? 0 : 1); };
    setText(dom.decode, rate(state.pending.decode));
    setText(dom.prefill, rate(state.pending.prefill));
  }

  function renderContext(model) {
    if (!model || !model.contextMax) {
      setText(dom.ctxText, DASH);
      setFill(dom.ctxBar, 0);
      return;
    }
    setText(dom.ctxText, num(model.contextUsed) + " / " + num(model.contextMax));
    setFill(dom.ctxBar, model.contextUsed / model.contextMax);
  }

  function renderGpu(gpu) {
    setText(dom.gpuUtil, gpu ? pctText(gpu.utilPct) : DASH);
    setText(dom.gpuTemp, gpu && !missing(gpu.tempC) ? num(gpu.tempC, 0) + " °C" : DASH);
    setText(dom.gpuPower, gpu && !missing(gpu.powerW) ? num(gpu.powerW, 0) + " W" : DASH);
    if (!gpu || !gpu.vramTotalGiB) {
      setText(dom.vramText, DASH);
      setFill(dom.vramBar, 0);
      return;
    }
    setText(dom.vramText, num(gpu.vramUsedGiB, 1) + " / " + num(gpu.vramTotalGiB, 1) + " GiB");
    setFill(dom.vramBar, gpu.vramUsedGiB / gpu.vramTotalGiB);
  }

  function renderSystem(s) {
    setText(dom.cpu, pctText(s.cpuPct));
    setText(dom.deviceName, s.device ? s.device : DASH);
    if (!s.ramTotalGiB) {
      setText(dom.ramText, DASH);
      setFill(dom.ramBar, 0);
      return;
    }
    setText(dom.ramText, num(s.ramUsedGiB, 1) + " / " + num(s.ramTotalGiB, 1) + " GiB");
    setFill(dom.ramBar, s.ramUsedGiB / s.ramTotalGiB);
  }

  function renderRequests(list) {
    const rows = (Array.isArray(list) ? list : []).slice(0, MAX_REQS);
    const sig = rows.map(function (r) {
      return [r.id, r.state, r.outputTokens, Math.round(r.decodeTps || 0)].join(":");
    }).join("|");
    if (sig === state.reqSig) return;   // keep quiet between ticks
    state.reqSig = sig;
    dom.requests.textContent = "";
    if (!rows.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No requests yet.";
      dom.requests.appendChild(li);
      return;
    }
    rows.forEach(function (r) {
      const li = document.createElement("li");
      const line = document.createElement("div");
      line.className = "req-line";
      const id = document.createElement("span");
      id.className = "req-id";
      id.textContent = String(r.id);
      const st = document.createElement("span");
      st.className = "req-state";
      st.dataset.state = r.state;
      st.textContent = r.state;
      line.appendChild(id);
      line.appendChild(st);
      const meta = document.createElement("div");
      meta.className = "req-meta";
      meta.textContent = ago(r.startedAt) + " · " + num(r.outputTokens) + " tok" +
        (r.decodeTps ? " · " + num(r.decodeTps, 1) + " tok/s" : "");
      li.appendChild(line);
      li.appendChild(meta);
      dom.requests.appendChild(li);
    });
  }

  function paintFresh() {
    if (!state.lastAt) return;
    const age = Date.now() - state.lastAt;
    setText(dom.fresh, "Updated " + Math.round(age / 1000) + "s ago");
    const stale = age > STALE_MS;
    if (dom.fresh.dataset.stale !== String(stale)) dom.fresh.dataset.stale = String(stale);
    if (stale && state.signedIn) {
      setText(dom.pillText, "Reconnecting");
      dom.pill.dataset.state = "offline";
    }
  }

  /* ---------------------------------------------------------- polling */

  function nextDelay() {
    if (state.failures > 0) return Math.min(BACKOFF_MAX_MS, POLL_MS * Math.pow(2, state.failures));
    return state.signedIn ? POLL_MS : SIGNED_OUT_MS;
  }
  function schedule() {
    window.clearTimeout(state.timer);
    state.timer = window.setTimeout(tick, nextDelay());
  }

  async function tick() {
    if (!state.signedIn && state.checked) return;     // signed out: nothing to poll until sign-in
    if (document.hidden) { schedule(); return; }      // paused in the background
    if (!state.busy) {                                // never overlap requests
      state.busy = true;
      try {
        const s = await api("/api/status");
        if (!state.signedIn) goSignedIn();
        onStatus(s);
      } catch (err) {
        onStatusError(err);
      } finally {
        state.busy = false;
        state.checked = true;
      }
    }
    if (state.chat.id) pollChunk();
    schedule();
  }

  function onStatusError(err) {
    if (err.status === 401) {
      // Revoked, expired or never signed in: show the form once, without wiping a message on it.
      if (state.signedIn || !state.checked) goSignedOut(state.signedIn ? "This phone was signed out. Scan a new QR code to sign in again." : "");
      return;
    }
    if (!err.status || err.status >= 500) state.failures += 1;
  }

  /* ------------------------------------------------------ views + auth */

  function goSignedIn() {
    state.signedIn = true;
    show(dom.signIn, false);
    show(dom.dash, true);
    setSignInError("");
  }

  function goSignedOut(message) {
    state.signedIn = false;
    state.chat = freshChat();
    dom.chatLog.textContent = "";
    setText(dom.chatNote, "");
    show(dom.stopButton, false);
    show(dom.chatStop, false);
    show(dom.dash, false);
    show(dom.signIn, true);
    setText(dom.pillText, "Signed out");
    dom.pill.dataset.state = "idle";
    if (state.lastAt) setText(dom.fresh, "Last update " + ago(state.lastAt));
    setSignInError(message);
  }

  function setSignInError(message) {
    setText(dom.error, message || "");
    show(dom.error, Boolean(message));
  }

  function startRetryCountdown(seconds) {
    window.clearInterval(state.retryTimer);
    state.countdown = true;
    dom.submit.disabled = true;
    let left = Math.max(1, Math.ceil(seconds || 1));
    const step = function () {
      setText(dom.retry, "Too many attempts. Try again in " + left + "s.");
      show(dom.retry, true);
      left -= 1;
      if (left <= 0) {
        window.clearInterval(state.retryTimer);
        state.retryTimer = 0;
        state.countdown = false;
        dom.submit.disabled = false;
        show(dom.retry, false);
      }
    };
    step();
    state.retryTimer = window.setInterval(step, 1000);
  }

  dom.form.addEventListener("submit", async function (event) {
    event.preventDefault();
    const payload = { password: dom.password.value, device: dom.device.value.trim() || guessDevice() };
    if (state.pair) payload.pair = state.pair;
    setSignInError("");
    show(dom.retry, false);
    dom.submit.disabled = true;
    try {
      await api("/api/login", { method: "POST", body: JSON.stringify(payload) });
      state.pair = null;            // a pairing code is good for one sign-in
      dom.password.value = "";      // the password is never kept anywhere
      remember();
      goSignedIn();
      announce("Signed in.");
      schedule();
    } catch (err) {
      if (err.status === 401) setSignInError(err.message || "Wrong password, or the QR code has expired.");
      else if (err.status === 429) startRetryCountdown(err.retryAfterS);
      else setSignInError("Cannot reach Cerebro. Check the address and try again.");
    } finally {
      if (!state.countdown) dom.submit.disabled = false;
    }
  });

  dom.signOut.addEventListener("click", async function () {
    try { await api("/api/logout", { method: "POST" }); } catch (err) { /* drop locally anyway */ }
    goSignedOut("");
    announce("Signed out.");
  });

  /* -------------------------------------------------------- stop flow */

  function renderStop(engine) {
    const busy = Boolean(state.chat.id) || (engine && engine.state === "generating");
    show(dom.stopButton, Boolean(busy) && state.signedIn);
    show(dom.chatStop, Boolean(state.chat.id));
  }

  function openConfirm() {
    state.returnFocus = document.activeElement;
    show(dom.scrim, true);
    dom.stopGo.focus();
  }
  function closeConfirm() {
    show(dom.scrim, false);
    if (state.returnFocus && state.returnFocus.focus) state.returnFocus.focus();
  }

  dom.stopButton.addEventListener("click", openConfirm);
  dom.chatStop.addEventListener("click", openConfirm);
  dom.stopCancel.addEventListener("click", closeConfirm);
  dom.scrim.addEventListener("click", function (event) {
    if (event.target === dom.scrim) closeConfirm();
  });
  dom.stopGo.addEventListener("click", async function () {
    closeConfirm();
    try {
      await api("/api/stop", { method: "POST" });
      announce("Stopping.");
    } catch (err) {
      announce("Stop failed.");
    }
  });
  document.addEventListener("keydown", function (event) {
    if (dom.scrim.hidden) return;
    if (event.key === "Escape") { closeConfirm(); return; }
    if (event.key !== "Tab") return;
    if (event.shiftKey && document.activeElement === dom.stopCancel) {
      event.preventDefault();
      dom.stopGo.focus();
    } else if (!event.shiftKey && document.activeElement === dom.stopGo) {
      event.preventDefault();
      dom.stopCancel.focus();
    }
  });

  /* ------------------------------------------------------------- chat */

  function syncChat(allowed) {
    show(dom.tabChat, Boolean(allowed));
    if (!allowed) {
      show(dom.panelChat, false);
      selectTab("status");
    }
    dom.send.disabled = !allowed || Boolean(state.chat.id);
  }

  function addMessage(role, text) {
    const wrap = document.createElement("div");
    wrap.className = "msg " + role;
    const who = document.createElement("p");
    who.className = "who";
    who.textContent = role === "user" ? "You" : "Model";
    const body = document.createElement("p");
    body.className = "body";
    body.textContent = text;          // model output is only ever set as text
    wrap.appendChild(who);
    wrap.appendChild(body);
    dom.chatLog.appendChild(wrap);
    scrollChat();
    return { wrap: wrap, body: body };
  }

  function scrollChat() {
    dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
  }

  function thinkingBox(msg) {
    const details = document.createElement("details");
    details.className = "thinking";
    const summary = document.createElement("summary");
    summary.textContent = "Thinking";
    const body = document.createElement("p");
    body.className = "body";
    details.appendChild(summary);
    details.appendChild(body);
    msg.wrap.insertBefore(details, msg.body);
    return body;
  }

  dom.chatForm.addEventListener("submit", function (event) {
    event.preventDefault();
    sendChat();
  });
  dom.chatInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendChat();
    }
  });

  async function sendChat() {
    const chat = state.chat;
    const text = dom.chatInput.value.trim();
    if (!text || chat.id) return;
    const turns = chat.turns.concat([{ role: "user", text: text }]);
    addMessage("user", text);
    dom.chatInput.value = "";
    setText(dom.chatNote, "");
    try {
      const res = await api("/api/chat", { method: "POST", body: JSON.stringify({ turns: turns }) });
      chat.turns = turns;
      chat.id = res.id;
      chat.from = 0;
      chat.text = "";
      chat.node = addMessage("assistant", "");
      chat.streamFails = 0;
      dom.send.disabled = true;
      renderStop(state.status && state.status.engine);
      announce("Model is replying.");
    } catch (err) {
      if (err.status === 403) setText(dom.chatNote, "Chat is switched off in Cerebro.");
      else if (err.status === 409) setText(dom.chatNote, "The model is busy. Try again in a moment.");
      else setText(dom.chatNote, "That message could not be sent.");
    }
  }

  async function pollChunk() {
    const chat = state.chat;
    if (!chat.id || chat.streamBusy) return;   // never overlap requests
    chat.streamBusy = true;
    try {
      const res = await api("/api/chat/" + encodeURIComponent(chat.id) + "?from=" + chat.from);
      chat.streamFails = 0;
      if (typeof res.next === "number") chat.from = res.next;
      else chat.from += (res.text || "").length;
      if (res.text) {
        chat.text += res.text;
        setText(chat.node.body, chat.text);
        scrollChat();
      }
      if (res.thinking) {
        if (!chat.thinking && chat.node) chat.thinking = thinkingBox(chat.node);
        setText(chat.thinking, typeof res.thinking === "string" ? res.thinking : "Working it out");
      }
      if (res.error) {
        setText(dom.chatNote, res.error);
        endChat();
      } else if (res.done) {
        endChat();
      }
    } catch (err) {
      chat.streamFails += 1;
      if (chat.streamFails > 4) {
        setText(dom.chatNote, "Lost the connection while replying.");
        endChat();
      }
    } finally {
      chat.streamBusy = false;
    }
  }

  function endChat() {
    const chat = state.chat;
    if (chat.text) chat.turns = chat.turns.concat([{ role: "assistant", text: chat.text }]);
    chat.id = null;
    chat.text = "";
    chat.node = null;
    chat.thinking = null;
    dom.send.disabled = false;
    renderStop(state.status && state.status.engine);
    announce("Reply finished.");
  }

  /* ------------------------------------------------------------- tabs */

  function selectTab(which) {
    const onStatus = which !== "chat";
    dom.tabStatus.setAttribute("aria-selected", String(onStatus));
    dom.tabStatus.tabIndex = onStatus ? 0 : -1;
    dom.tabChat.setAttribute("aria-selected", String(!onStatus));
    dom.tabChat.tabIndex = onStatus ? -1 : 0;
    show(dom.panelStatus, onStatus);
    show(dom.panelChat, !onStatus && !dom.tabChat.hidden);
  }
  dom.tabStatus.addEventListener("click", function () { selectTab("status"); });
  dom.tabChat.addEventListener("click", function () { selectTab("chat"); });

  /* -------------------------------------------------------------- init */

  dom.device.value = guessDevice();
  state.pair = takePairCode();
  if (state.pair || seenBefore()) {
    setText(dom.pairHint, "Enter the password for this computer. A pairing code works once.");
  }
  syncChat(false);
  selectTab("status");

  window.setInterval(function () {
    state.pending.force = true;
    paintRates();
    paintFresh();
  }, PAINT_MS);

  window.addEventListener("online", function () {
    state.failures = 0;
    if (state.signedIn) schedule();
  });
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden && state.signedIn) schedule();
  });

  tick();   // one loop drives everything, so requests never overlap
})();
