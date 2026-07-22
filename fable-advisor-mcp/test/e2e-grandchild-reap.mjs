#!/usr/bin/env node
// Regression test for the process-group-kill fix: on timeout, the server
// must kill the whole process group, not just the direct child. The
// existing e2e-hang.mjs only checks that the advisor *promise* settles in
// time — it doesn't check whether anything the child forked is still
// running afterward. This test drives a fake `claude` that forks a real
// background grandchild, forces the timeout path, and asserts the
// grandchild's pid is actually dead afterward.
import { spawn } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

if (process.platform === "win32") {
  console.log("SKIP: e2e-grandchild-reap is a POSIX-only regression test (bash fake-claude)");
  process.exit(0);
}

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

const fakeDir = mkdtempSync(join(tmpdir(), "fable-advisor-reap-test-"));
const logDir = mkdtempSync(join(tmpdir(), "fable-advisor-reap-test-log-"));
const fakeClaude = join(fakeDir, "fake-claude");
const pidFile = join(fakeDir, "grandchild.pid");
// Forks a detached grandchild that outlives this script, then hangs forever
// itself so the server's timeout path fires.
writeFileSync(
  fakeClaude,
  `#!/bin/bash
cat > /dev/null
(sleep 300 &
echo $! > "${pidFile}") &
disown
sleep 300
`,
);
chmodSync(fakeClaude, 0o755);

const server = spawn("node", [join(repoRoot, "dist", "index.js")], {
  env: {
    ...process.env,
    FABLE_ADVISOR_CLAUDE_BIN: fakeClaude,
    FABLE_ADVISOR_LOG_DIR: logDir,
    FABLE_ADVISOR_TIMEOUT_MS: "2000",
  },
  stdio: ["pipe", "pipe", "inherit"],
});

const isAlive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

const cleanup = (exitCode) => {
  server.kill();
  rmSync(fakeDir, { recursive: true, force: true });
  rmSync(logDir, { recursive: true, force: true });
  process.exit(exitCode);
};

const started = Date.now();
let buf = "";
server.stdout.on("data", (chunk) => {
  buf += chunk.toString();
  for (const line of buf.split("\n")) {
    if (!line.trim()) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      continue;
    }
    if (msg.id !== 2) continue;
    const secs = ((Date.now() - started) / 1000).toFixed(1);
    if (!existsSync(pidFile)) {
      console.log(`FAIL: grandchild pid file never appeared after ${secs}s`);
      cleanup(1);
      return;
    }
    const grandchildPid = Number(readFileSync(pidFile, "utf8").trim());
    // Give the group-kill a beat to land, then check.
    setTimeout(() => {
      const alive = isAlive(grandchildPid);
      console.log(
        `${alive ? "FAIL" : "PASS"}: grandchild pid=${grandchildPid} ${alive ? "still alive" : "reaped"} ${secs}s after advisor call settled`,
      );
      cleanup(alive ? 1 : 0);
    }, 500);
  }
});

const send = (obj) => server.stdin.write(JSON.stringify(obj) + "\n");
send({
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "e2e-reap-test", version: "0" } },
});
setTimeout(() => send({ jsonrpc: "2.0", method: "notifications/initialized" }), 200);
setTimeout(
  () => send({ jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "advisor", arguments: { task: "t", context: "c" } } }),
  400,
);
setTimeout(() => {
  console.log("FAIL: advisor call still hung after 30s");
  cleanup(1);
}, 30_000);
