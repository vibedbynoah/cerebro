"use strict";
/**
 * Cerebro theme + motion init. The one classic, render-blocking script on a Cerebro page: load it
 * as `<script src="./theme-init.js">` in `<head>`, before the stylesheets (external, so it works
 * under `script-src 'self'`). Module scripts run after first paint, which is too late for either
 * job below.
 *
 * 1. Applies the visitor's stored explicit theme (`cerebro-theme`) before first paint. With no
 *    stored choice it sets nothing and tokens.css follows `prefers-color-scheme`.
 * 2. Adds `motion-pending` when full motion will run, so motion.css can hide reveal/count-up start
 *    states before first paint instead of flashing the final state. motion.ts's initMotion()
 *    removes it; a CSS failsafe un-hides everything after 2.5 s if that never happens. Only
 *    include this script on pages that call initMotion().
 */
(function () {
    var html = document.documentElement;
    try {
        var stored = window.localStorage.getItem("cerebro-theme");
        if (stored === "light" || stored === "dark")
            html.setAttribute("data-theme", stored);
    }
    catch (_err) {
        // Storage blocked: the OS preference decides, same as a first visit.
    }
    try {
        if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches && "IntersectionObserver" in window) {
            html.classList.add("motion-pending");
        }
    }
    catch (_err) {
        // No matchMedia: leave content visible.
    }
})();
