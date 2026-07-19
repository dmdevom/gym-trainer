import { API_BASE_URL } from "./api";

// Anonymous, fire-and-forget usage beacon. Mirrors the backend's telemetry posture: it can
// only ever add signal, never break or slow the page. No cookies, no identity — just a
// random per-browser id so we can tell a returning visitor from a new one.

const CID_KEY = "trainer_cid";

/** A stable anonymous id for this browser: random (not tied to any identity), kept in
 *  localStorage. Returns "" when storage is blocked (e.g. private mode) — events are then
 *  simply id-less rather than failing. */
export function getClientId(): string {
  if (typeof window === "undefined") return "";
  try {
    let cid = window.localStorage.getItem(CID_KEY);
    if (!cid) {
      cid = crypto.randomUUID();
      window.localStorage.setItem(CID_KEY, cid);
    }
    return cid;
  } catch {
    return "";
  }
}

/** Send one usage event. Never throws, never blocks, never awaited — a tracking hiccup must
 *  not touch the UI. Prefers sendBeacon (survives page unload); the text/plain body keeps it
 *  a CORS-"simple" request (no preflight, which sendBeacon can't do), and the server reads
 *  the raw body regardless of content-type. Falls back to a keepalive fetch. */
export function track(event: string, props?: Record<string, unknown>): void {
  if (typeof window === "undefined") return;
  try {
    const body = JSON.stringify({ event, cid: getClientId(), props });
    const url = `${API_BASE_URL}/e`;
    const blob = new Blob([body], { type: "text/plain" });
    if (navigator.sendBeacon?.(url, blob)) return;
    void fetch(url, { method: "POST", body, keepalive: true }).catch(() => undefined);
  } catch {
    // best-effort: swallow everything
  }
}
