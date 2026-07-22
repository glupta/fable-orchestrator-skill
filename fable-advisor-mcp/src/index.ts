#!/usr/bin/env node
/**
 * fable-advisor-mcp-server
 *
 * Gives an executor agent (OpenAI Codex CLI) an `advisor` tool backed by
 * Claude Fable 5 — a client-side port of Anthropic's server-side advisor
 * tool pattern (executor model consults a higher-intelligence advisor
 * mid-task for strategic guidance).
 *
 * Default backend shells out to Claude Code headless mode (`claude -p`),
 * so advisor calls run on the user's existing Claude plan login — no API
 * key. Set FABLE_ADVISOR_BACKEND=api + ANTHROPIC_API_KEY to bill the
 * Anthropic API directly instead.
 *
 * Unlike the native API feature, the executor's transcript is NOT
 * auto-forwarded: the executor must pass its task, progress, and question
 * explicitly. The tool description and ~/.codex/AGENTS.md guidance steer
 * Codex to do that.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import Anthropic from "@anthropic-ai/sdk";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { z } from "zod";

const ADVISOR_MODEL = process.env.FABLE_ADVISOR_MODEL ?? "claude-fable-5";
const ADVISOR_BACKEND = process.env.FABLE_ADVISOR_BACKEND ?? "claude-cli";
// API backend only: cap covers thinking + text. Fable thinks on every call
// and its tokenizer runs ~30% heavier than Opus-tier, so this sits well
// above the 2048 the native advisor tool recommends.
const ADVISOR_MAX_TOKENS = Number(process.env.FABLE_ADVISOR_MAX_TOKENS ?? 8192);
const ADVISOR_EFFORT = process.env.FABLE_ADVISOR_EFFORT as
  | "low"
  | "medium"
  | "high"
  | "xhigh"
  | "max"
  | undefined;
// depth:"deep" effort. Default xhigh, NOT max: a max-effort advisor_verify
// over a large repo can outlive ADVISOR_TIMEOUT_MS and return nothing —
// the most expensive consults were the ones most likely to be lost. Set
// FABLE_ADVISOR_DEEP_EFFORT=max to restore the old behavior deliberately.
const ADVISOR_DEEP_EFFORT =
  (process.env.FABLE_ADVISOR_DEEP_EFFORT as typeof ADVISOR_EFFORT) ?? "xhigh";
// Keep under Codex's tool_timeout_sec (600) so we return a readable error
// instead of Codex killing the call.
const ADVISOR_TIMEOUT_MS = Number(process.env.FABLE_ADVISOR_TIMEOUT_MS ?? 570_000);

// advisor_verify: read-only tools the grounded advisor may use.
const VERIFY_TOOLS = "Read,Grep,Glob";

// Consult log + review-cadence state live outside any project repo so they
// never show up as untracked files in governance-audited checkouts.
const GOVERNANCE_DIR =
  process.env.FABLE_ADVISOR_LOG_DIR ?? join(homedir(), ".fable-advisor");
const CONSULT_LOG_PATH = join(GOVERNANCE_DIR, "consults.jsonl");
const GOVERNANCE_STATE_PATH = join(GOVERNANCE_DIR, "state.json");
// Automatic review-cadence nagging is opt-in. With the env vars unset, the count/age
// thresholds never fire; the only way the [advisor-governance] notice
// appears is the user-controlled {"reviewRequested": true} in state.json
// (any agent can write it on the user's request). Significant-consult
// counting still runs — consultsSinceReview stays a useful signal for the
// user deciding WHEN to trigger a review. Set FABLE_ADVISOR_REVIEW_CONSULTS
// and/or FABLE_ADVISOR_REVIEW_DAYS to re-enable automatic nagging.
function envThreshold(name: string): number {
  const raw = process.env[name];
  const value = raw ? Number(raw) : NaN;
  return Number.isFinite(value) && value > 0 ? value : Infinity;
}
const REVIEW_CONSULT_THRESHOLD = envThreshold("FABLE_ADVISOR_REVIEW_CONSULTS");
const REVIEW_AGE_DAYS_THRESHOLD = envThreshold("FABLE_ADVISOR_REVIEW_DAYS");
// Decision-class language that marks a consult as review-debt-accumulating.
// Deliberately excludes ordinary engineering vocabulary (spec, contract,
// gate) that fired on routine consults and inflated the counter into noise.
const SIGNIFICANT_CONSULT_PATTERN =
  /freez[ei]|frozen|seal|launch|authoriz|approv|promot|go[\s/-]?no[\s/-]?go|preregist|sign[\s-]?off/i;

// Optional standing project brief (goal, current stage, known red flags)
// prepended to the system prompt. Read at call time so orchestrator edits
// apply without a server restart.
const BRIEF_PATH = process.env.FABLE_ADVISOR_BRIEF;

const SHARED_RULES = `Produce a plan or course correction, not the implementation:
- Begin your response with a single plain-text line (no bold, no heading markup) reading exactly "VERDICT: proceed", "VERDICT: revise", or "VERDICT: stop" — proceed = the executor's current plan/work is sound as-is; revise = continue, but change something specific you name; stop = a blocking problem must be resolved before any further progress. Pick the verdict for the executor's NEXT step, not for the whole project.
- If the executor asked a specific question, answer it directly first — but if the question presupposes a frame (e.g. "how do I make this gate pass?"), first assess whether the frame itself is sound (should this gate exist? is this the right experiment?). A well-executed step inside a wrong frame is still wrong.
- Identify the highest-leverage next steps and the failure modes the executor has not ruled out.
- Be concrete: name files, commands, APIs, invariants, and tests where the provided context allows.
- The executor's account may be incomplete or subtly wrong. Challenge assumptions its evidence does not support. Watch for inflated labels: a component's name in the executor's narrative (e.g. "CFR teacher", "independent evaluator") is a claim, not a fact.
- Keep guidance under roughly 300 words unless genuine complexity demands more. Give the shape of the solution and the tricky parts; never write the full implementation.
- End EVERY response with a section titled exactly "UNVERIFIED CLAIMS RELIED ON:" listing each load-bearing claim from the executor's context that you accepted without direct evidence (or "none"). Keep each item to one line. This section is mandatory — it marks the trust boundary for the executor and the human.`;

const ADVISOR_SYSTEM = `You are the advisor: a higher-intelligence model that a faster coding agent ("the executor" — OpenAI Codex) consults mid-task for strategic guidance. The executor sends you its task, a self-reported account of its progress (files read, commands run, results, errors), and optionally a specific question. Answer directly from the provided context; do not use tools.

One hard cap on your verdict: if the consult would authorize a gated milestone, promotion, launch, expensive (>~30 min), costly-to-undo, or irreversible action and the only support is the executor's own self-report, you may NOT issue "VERDICT: proceed". Cap it at "VERDICT: revise" and name the missing independent evidence or explicit user decision. Routine task/subtask completion, even final completion, is not an authorization event and may proceed from a strong evidence packet. A bare completion claim without direct excerpts (test output, diff hunks, or command results) must receive "VERDICT: revise" asking for the specific missing evidence; do not require advisor_verify solely because the task is final.

${SHARED_RULES}`;

const ADVISOR_VERIFY_SYSTEM = `You are the grounded advisor: a higher-intelligence model that a faster coding agent ("the executor" — OpenAI Codex) consults mid-task, with read-only access (Read, Grep, Glob) to the project working directory. The executor sends you its task, a self-reported account of its progress, and optionally a specific question.

Before advising, VERIFY the load-bearing claims in the executor's account against the actual files. Do not accept the executor's labels for what a component is or does — open the file and check. Prioritize: (1) claims the advice would be built on, (2) names/labels of components ("X is a solver", "Y is independent", "Z is frozen"), (3) any metric, gate, or test the executor says passes — read the code that computes it and confirm it can actually fail. Cite file paths and line numbers for what you verified. Budget your reading: verify the few claims that matter, not everything.

${SHARED_RULES.replace(
  'that you accepted without direct evidence (or "none")',
  'that you could not verify with the available tools (or "none"); claims you did verify belong in the body with file:line citations',
)}`;

interface AdvisorInput {
  task: string;
  project_state?: string;
  context: string;
  question?: string;
  depth?: "quick" | "deep";
}

// ---------------------------------------------------------------------------
// Governance: project brief, consult log, review-cadence notice.
// ---------------------------------------------------------------------------

/** Prepend the standing project brief (if configured and readable). */
function withBrief(system: string): string {
  if (!BRIEF_PATH) return system;
  try {
    const brief = readFileSync(BRIEF_PATH, "utf8").trim();
    if (!brief) return system;
    return `# Standing project brief (operator-maintained background; not direct evidence — prefer current quoted evidence when they conflict)\n${brief}\n\n${system}`;
  } catch {
    return system; // Missing/unreadable brief must never break a consult.
  }
}

interface GovernanceState {
  consultsSinceReview: number;
  lastReviewISO: string | null;
  reviewRequested: boolean;
}

function readGovernanceState(): GovernanceState {
  try {
    const raw = JSON.parse(readFileSync(GOVERNANCE_STATE_PATH, "utf8"));
    return {
      consultsSinceReview: Number(raw.consultsSinceReview) || 0,
      lastReviewISO: typeof raw.lastReviewISO === "string" ? raw.lastReviewISO : null,
      reviewRequested: raw.reviewRequested === true,
    };
  } catch {
    return { consultsSinceReview: 0, lastReviewISO: null, reviewRequested: false };
  }
}

/**
 * Record the consult, increment the counter for significant consults only,
 * and return the standing notice when a consolidated review is overdue or
 * user-requested. State/log I/O must never fail a consult.
 */
/**
 * Parse the mandatory "VERDICT: proceed|revise|stop" opener, if present.
 * Tolerates markdown decoration ("**VERDICT: stop**", "# VERDICT: revise")
 * despite the prompt forbidding it — models bold it anyway.
 */
function extractVerdict(advice: string): string | null {
  const match = /^[#*_>\s]*VERDICT:\s*[*_]*(proceed|revise|stop)\b/im.exec(advice);
  return match ? match[1].toLowerCase() : null;
}

interface ConsultTelemetry {
  backend: "claude-cli" | "api";
  inputTokens?: number;
  cacheCreationInputTokens?: number;
  cacheReadInputTokens?: number;
  outputTokens?: number;
  numTurns?: number;
  durationMs?: number;
  totalCostUsd?: number;
}

interface ConsultResult {
  advice: string;
  telemetry: ConsultTelemetry;
}

function recordConsultAndGetNotice(entry: {
  tool: string;
  task: string;
  projectState?: string;
  context: string;
  question?: string;
  projectDir?: string;
  depth?: string;
  advice: string;
  telemetry: ConsultTelemetry;
}): string | null {
  try {
    mkdirSync(GOVERNANCE_DIR, { recursive: true });
    const state = readGovernanceState();
    const significant =
      entry.depth === "deep" ||
      SIGNIFICANT_CONSULT_PATTERN.test(`${entry.task}\n${entry.question ?? ""}`);
    if (significant) state.consultsSinceReview += 1;
    if (!state.lastReviewISO) state.lastReviewISO = new Date().toISOString();
    writeFileSync(GOVERNANCE_STATE_PATH, JSON.stringify(state, null, 2) + "\n");

    appendFileSync(
      CONSULT_LOG_PATH,
      JSON.stringify({
        ts: new Date().toISOString(),
        tool: entry.tool,
        projectDir: entry.projectDir ?? null,
        taskPreview: entry.task.slice(0, 300),
        projectStateChars: entry.projectState?.length ?? 0,
        projectStateHash: entry.projectState
          ? createHash("sha256").update(entry.projectState).digest("hex").slice(0, 16)
          : null,
        contextChars: entry.context.length,
        questionPreview: entry.question?.slice(0, 300) ?? null,
        significant,
        verdict: extractVerdict(entry.advice),
        telemetry: entry.telemetry,
        advice: entry.advice,
        consultsSinceReview: state.consultsSinceReview,
        serverVersion: SERVER_VERSION,
      }) + "\n",
    );

    if (state.reviewRequested) {
      return (
        "[advisor-governance] The user has requested a consolidated adversarial review. " +
        "Surface this to the user verbatim and do not treat any new spec, seal, or launch as approved until that review completes and resets this flag."
      );
    }
    const ageDays =
      (Date.now() - Date.parse(state.lastReviewISO)) / 86_400_000;
    if (
      state.consultsSinceReview >= REVIEW_CONSULT_THRESHOLD ||
      ageDays >= REVIEW_AGE_DAYS_THRESHOLD
    ) {
      return (
        `[advisor-governance] ${state.consultsSinceReview} significant (decision-class) consults since the last consolidated review` +
        (state.lastReviewISO ? ` (last: ${state.lastReviewISO.slice(0, 10)})` : "") +
        ". A full-repo adversarial review is overdue — surface this notice to the user verbatim and recommend scheduling one before further frozen specs, seals, or launch approvals. Per-consult advice cannot catch whole-program drift."
      );
    }
    return null;
  } catch (error) {
    console.error("[fable-advisor] governance bookkeeping failed:", error);
    return null;
  }
}

function buildAdvisorPrompt(params: AdvisorInput): string {
  const sections = [
    "# Task the executor is working on",
    params.task,
  ];
  if (params.project_state) {
    sections.push("", "# Compact project state", params.project_state);
  }
  sections.push(
    "",
    "# Direct evidence, progress, and current interpretation",
    params.context,
  );
  if (params.question) {
    sections.push("", "# Specific question for you", params.question);
  }
  sections.push(
    "",
    "(Advisor: keep your guidance focused — a plan and the tricky parts, not a comprehensive essay.)",
  );
  return sections.join("\n");
}

// ---------------------------------------------------------------------------
// Backend 1 (default): Claude Code headless mode — runs on the user's plan.
// ---------------------------------------------------------------------------

function resolveClaudeBin(): string {
  if (process.env.FABLE_ADVISOR_CLAUDE_BIN) {
    return process.env.FABLE_ADVISOR_CLAUDE_BIN;
  }
  // Codex may spawn this server with a PATH that misses user bin dirs.
  const wellKnown = join(homedir(), ".local", "bin", "claude");
  for (const candidate of process.platform === "win32"
    ? [`${wellKnown}.exe`, `${wellKnown}.cmd`, wellKnown]
    : [wellKnown]) {
    if (existsSync(candidate)) return candidate;
  }
  return "claude";
}

interface ProcessResult {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

function runProcess(
  bin: string,
  args: string[],
  input: string,
  timeoutMs: number,
  cwd?: string,
): Promise<ProcessResult> {
  return new Promise((resolve, reject) => {
    // detached: true makes the child its own process-group leader (pgid ===
    // child.pid), so any grandchild it forks inherits that group and
    // `kill(-pgid)` below reaches the whole tree instead of only the direct
    // child.
    const child = spawn(bin, args, { stdio: ["pipe", "pipe", "pipe"], cwd, detached: true });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;
    let flushTimer: NodeJS.Timeout | undefined;

    // Liveness invariant: this promise MUST settle exactly once no matter
    // which combination of 'exit'/'close'/'error'/timeout fires. Resolving
    // only on 'close' deadlocks when the child leaves behind a grandchild
    // that inherited its stdio pipes: the child exits, 'close' never fires,
    // and the advisor call hangs past its own timeout.
    const settle = (code: number | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (flushTimer) clearTimeout(flushTimer);
      resolve({ code, stdout, stderr, timedOut });
    };

    const timer = setTimeout(() => {
      timedOut = true;
      // Kill the whole process group, not just the direct child. Signaling
      // only the child stops the *promise* from hanging but leaves any
      // grandchild running indefinitely, holding the stdio pipes open.
      // Across a long-lived MCP server those accumulate and can exhaust
      // memory/FDs, taking the parent worker down with them.
      try {
        process.kill(-child.pid!, "SIGKILL");
      } catch {
        child.kill("SIGKILL");
      }
      // Don't wait for 'close' after the kill — if the pipes are held open
      // elsewhere it never comes. Give the streams a beat, then settle.
      setTimeout(() => settle(null), 2_000);
    }, timeoutMs);

    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
    // An unhandled 'error' on any of these streams (e.g. EPIPE from a killed
    // child mid-read) throws and crashes this whole Node process — taking
    // the MCP server, and the worker blocked reading its stdio transport,
    // down with it. Swallow them all; the exit/close handlers below still
    // report what happened.
    child.stdin.on("error", () => {});
    child.stdout.on("error", () => {});
    child.stderr.on("error", () => {});
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(error);
    });
    // Preferred path: 'close' means stdio fully drained — settle immediately.
    child.on("close", (code) => settle(code));
    // Fallback: 'exit' always fires when the process ends. Allow 1.5s for
    // any remaining stdout to drain toward 'close'; if 'close' never comes
    // (orphaned grandchild holding the pipe), settle with what we have.
    child.on("exit", (code) => {
      flushTimer = setTimeout(() => settle(code), 1_500);
    });

    child.stdin.write(input);
    child.stdin.end();
  });
}

interface CliConsultOptions {
  system: string;
  tools: string;
  cwd?: string;
  /** Extra directories the advisor may read (claude --add-dir). */
  additionalDirs?: string[];
  /** Per-call effort override ("quick" → low, "deep" → max). */
  depth?: "quick" | "deep";
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function cliConsultResult(
  advice: string,
  telemetry: Partial<ConsultTelemetry> = {},
): ConsultResult {
  return { advice, telemetry: { backend: "claude-cli", ...telemetry } };
}

async function consultViaClaudeCli(
  params: AdvisorInput,
  opts: CliConsultOptions,
): Promise<ConsultResult> {
  const bin = resolveClaudeBin();
  // Node >=18.20/20.12 (CVE-2024-27980 fix) refuses to spawn .cmd/.bat
  // without shell:true, and shell:true cannot safely carry our multiline
  // system-prompt args through cmd.exe quoting. Fail with instructions
  // instead of an opaque EINVAL.
  if (process.platform === "win32" && /\.(cmd|bat)$/i.test(bin)) {
    return cliConsultResult(
      `Advisor error (claude_bin_is_cmd_shim): '${bin}' is a .cmd shim, which Node cannot spawn safely. ` +
      "Tell the user to point FABLE_ADVISOR_CLAUDE_BIN at the claude .exe instead " +
      "(typically next to the shim, e.g. claude.exe in the same directory). " +
      "Continue the task without advice."
    );
  }
  const args = [
    "-p",
    "--model",
    ADVISOR_MODEL,
    "--system-prompt",
    opts.system,
    "--tools",
    opts.tools,
    "--strict-mcp-config",
    "--no-session-persistence",
    "--output-format",
    "json",
  ];
  for (const dir of opts.additionalDirs ?? []) {
    if (existsSync(dir)) args.push("--add-dir", dir);
  }
  const effort =
    opts.depth === "quick"
      ? "low"
      : opts.depth === "deep"
        ? ADVISOR_DEEP_EFFORT
        : ADVISOR_EFFORT;
  if (effort) {
    args.push("--effort", effort);
  }

  let result: ProcessResult;
  try {
    result = await runProcess(
      bin,
      args,
      buildAdvisorPrompt(params),
      ADVISOR_TIMEOUT_MS,
      opts.cwd,
    );
  } catch (error) {
    const err = error as NodeJS.ErrnoException;
    if (err.code === "ENOENT") {
      return cliConsultResult(
        `Advisor error (claude_cli_not_found): '${bin}' is not on this server's PATH. ` +
        "Tell the user to set FABLE_ADVISOR_CLAUDE_BIN to the claude binary path. " +
        "Continue the task without advice."
      );
    }
    throw error;
  }

  if (result.timedOut) {
    return cliConsultResult(`Advisor error (execution_time_exceeded): the advisor call exceeded ${Math.round(ADVISOR_TIMEOUT_MS / 1000)}s and was killed. Continue the task without advice.`);
  }
  const raw = result.stdout.trim();
  if (result.code !== 0 || !raw) {
    const detail = (result.stderr.trim() || raw || "no output").slice(0, 500);
    return cliConsultResult(
      `Advisor error (claude_cli_exit_${result.code ?? "unknown"}): ${detail}. ` +
      "If this mentions authentication or login, tell the user to run `claude` once " +
      "interactively to sign in. Continue the task without advice."
    );
  }
  // Parse the JSON envelope; fall back to raw text if the format ever changes.
  let advice = raw;
  let telemetry: Partial<ConsultTelemetry> = {};
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (typeof parsed.result === "string" && parsed.result.trim()) {
      advice = parsed.result.trim();
      if (parsed.is_error) {
        return cliConsultResult(`Advisor error (cli_reported_error): ${advice.slice(0, 500)}. Continue the task without advice.`);
      }
    }
    const usage =
      parsed.usage && typeof parsed.usage === "object"
        ? (parsed.usage as Record<string, unknown>)
        : {};
    telemetry = {
      inputTokens: optionalNumber(usage.input_tokens),
      cacheCreationInputTokens: optionalNumber(usage.cache_creation_input_tokens),
      cacheReadInputTokens: optionalNumber(usage.cache_read_input_tokens),
      outputTokens: optionalNumber(usage.output_tokens),
      numTurns: optionalNumber(parsed.num_turns),
      durationMs: optionalNumber(parsed.duration_ms),
      totalCostUsd: optionalNumber(parsed.total_cost_usd),
    };
  } catch {
    // Not JSON — use raw stdout as the advice.
  }
  console.error(
    `[fable-advisor] claude-cli ${ADVISOR_MODEL} ok (${advice.length} chars, in=${telemetry.inputTokens ?? "?"}, cache=${telemetry.cacheReadInputTokens ?? "?"}, out=${telemetry.outputTokens ?? "?"})`,
  );
  return cliConsultResult(advice, telemetry);
}

// ---------------------------------------------------------------------------
// Backend 2 (opt-in): direct Anthropic API with ANTHROPIC_API_KEY.
// ---------------------------------------------------------------------------

function apiErrorText(error: unknown): string {
  if (error instanceof Anthropic.AuthenticationError) {
    return "Advisor error (authentication): the Anthropic API key was rejected. Tell the user to check ANTHROPIC_API_KEY for this MCP server. Continue the task without advice.";
  }
  if (error instanceof Anthropic.RateLimitError) {
    return "Advisor error (too_many_requests): the advisor call was rate-limited. Continue without advice; retry on the next natural checkpoint.";
  }
  if (error instanceof Anthropic.APIError && error.status === 529) {
    return "Advisor error (overloaded): the advisor hit capacity limits. Continue without advice; retry on the next natural checkpoint.";
  }
  if (error instanceof Anthropic.APIError) {
    return `Advisor error (${error.status ?? "unknown"}): ${error.message}. Continue the task without advice.`;
  }
  return `Advisor error (unavailable): ${error instanceof Error ? error.message : String(error)}. Continue the task without advice.`;
}

function apiConsultResult(
  advice: string,
  telemetry: Partial<ConsultTelemetry> = {},
): ConsultResult {
  return { advice, telemetry: { backend: "api", ...telemetry } };
}

async function consultViaApi(
  params: AdvisorInput,
  system: string,
): Promise<ConsultResult> {
  if (!process.env.ANTHROPIC_API_KEY) {
    return apiConsultResult("Advisor error (no_api_key): FABLE_ADVISOR_BACKEND=api requires ANTHROPIC_API_KEY. Continue the task without advice.");
  }
  const client = new Anthropic();
  // Fable 5: thinking is always on — omit the `thinking` param entirely.
  // Streaming keeps long advisor turns inside HTTP timeouts.
  const stream = client.messages.stream({
    model: ADVISOR_MODEL,
    max_tokens: ADVISOR_MAX_TOKENS,
    ...(ADVISOR_EFFORT ? { output_config: { effort: ADVISOR_EFFORT } } : {}),
    system,
    messages: [{ role: "user", content: buildAdvisorPrompt(params) }],
  });
  const message = await stream.finalMessage();

  if (message.stop_reason === "refusal") {
    return apiConsultResult("Advisor error (refusal): the advisor declined this request for safety reasons. Continue the task without advice.");
  }

  let advice = message.content
    .filter((block): block is Anthropic.TextBlock => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();

  if (!advice) {
    return apiConsultResult("Advisor error (empty_response): the advisor returned no text. Continue the task without advice.");
  }
  if (message.stop_reason === "max_tokens") {
    advice += `\n\n[Advisor output truncated at max_tokens=${ADVISOR_MAX_TOKENS}.]`;
  }

  console.error(
    `[fable-advisor] api ${ADVISOR_MODEL} in=${message.usage.input_tokens} out=${message.usage.output_tokens} stop=${message.stop_reason}`,
  );
  const usage = message.usage as unknown as Record<string, unknown>;
  return apiConsultResult(advice, {
    inputTokens: optionalNumber(usage.input_tokens),
    cacheCreationInputTokens: optionalNumber(usage.cache_creation_input_tokens),
    cacheReadInputTokens: optionalNumber(usage.cache_read_input_tokens),
    outputTokens: optionalNumber(usage.output_tokens),
  });
}

// ---------------------------------------------------------------------------
// MCP server
// ---------------------------------------------------------------------------

// Bump on behavior changes: printed at startup so `ps` + codex logs can tell
// a stale long-lived server process from one running current code.
const SERVER_VERSION = "3.1.0";

// Last-resort net: this server is a long-lived dependency of the Codex
// worker's MCP transport (observed running for days across sessions). Any
// exception that reaches here uncaught would otherwise kill the whole
// process and sever that transport out from under the worker. Every known
// throw site is already handled at its source (see runProcess and
// runConsult above) — this only catches something new.
process.on("uncaughtException", (error) => {
  console.error("[fable-advisor] uncaughtException (server staying up):", error);
});
process.on("unhandledRejection", (reason) => {
  console.error("[fable-advisor] unhandledRejection (server staying up):", reason);
});

const server = new McpServer({
  name: "fable-advisor-mcp-server",
  version: SERVER_VERSION,
});

const COMMON_INPUT_SCHEMA = {
  task: z
    .string()
    .min(1, "task is required")
    .max(50_000)
    .describe("The user's task/request, verbatim or faithfully summarized"),
  project_state: z
    .string()
    .max(12_000)
    .optional()
    .describe(
      "Compact current state: goal, relevant decisions, constraints, known failures, and unresolved questions. Keep it stable and under 12,000 characters; omit when the context is self-contained.",
    ),
  context: z
    .string()
    .min(1, "context is required — the advisor sees nothing else")
    .max(400_000)
    .describe(
      "Direct evidence and current interpretation: exact code/diff excerpts, command outputs, evidence for and against, unverified assumptions, and the current plan. Maximum 400,000 characters; prefer a focused packet.",
    ),
  question: z
    .string()
    .max(10_000)
    .optional()
    .describe("Specific decision or question to resolve (optional)"),
  depth: z
    .enum(["quick", "deep"])
    .optional()
    .describe(
      "Optional effort override: 'quick' for cheap sanity checks, 'deep' for go/no-go reviews, negotiations, and anything expensive to get wrong. Omit for the default.",
    ),
};

const COMMON_ANNOTATIONS = {
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: false,
  openWorldHint: true,
} as const;

/** Shared consult path: backend call, then governance bookkeeping. */
async function runConsult(
  tool: string,
  params: AdvisorInput,
  cli: CliConsultOptions,
  projectDir?: string,
): Promise<string> {
  let result: ConsultResult;
  try {
    result =
      ADVISOR_BACKEND === "api"
        ? await consultViaApi(params, withBrief(cli.system))
        : await consultViaClaudeCli(params, { ...cli, system: withBrief(cli.system) });
  } catch (error) {
    const advice =
      ADVISOR_BACKEND === "api"
        ? apiErrorText(error)
        : `Advisor error (unavailable): ${error instanceof Error ? error.message : String(error)}. Continue the task without advice.`;
    result =
      ADVISOR_BACKEND === "api"
        ? apiConsultResult(advice)
        : cliConsultResult(advice);
  }

  const notice = recordConsultAndGetNotice({
    tool,
    task: params.task,
    projectState: params.project_state,
    context: params.context,
    question: params.question,
    projectDir,
    depth: params.depth,
    advice: result.advice,
    telemetry: result.telemetry,
  });
  return notice ? `${notice}\n\n${result.advice}` : result.advice;
}

server.registerTool(
  "advisor",
  {
    title: "Consult Fable Advisor",
    description: `Consult a stronger reviewer model (Claude Fable 5) for strategic guidance. This is the DEFAULT tool for strategy, review, debugging, negotiation, routine completion checks, and go/no-go framing. It has NO tools and every call is FRESH: no transcript is forwarded or resumed. Act as a context compiler—send compact project_state plus a focused context packet containing the task, exact code/diff excerpts, command outputs, evidence for and against your interpretation, unverified assumptions, and your current plan. Evidence you do not send does not exist for the advisor.

Call advisor BEFORE substantive work — before writing, before committing to an interpretation, before building on an assumption. Orientation (finding files, reading, listing) is not substantive work; do that first so you have the excerpts to paste, then call advisor. Also call it when you believe the task is complete (after making the deliverable durable), when stuck, or when considering a change of approach.

Do NOT reach for advisor_verify instead of compiling a better packet. If Fable identifies missing evidence, read it yourself and make another fresh plain call with the new excerpt. advisor_verify costs an order of magnitude more and is reserved for explicit independent-audit requests, a specific load-bearing factual dispute that cannot be packaged credibly, or one genuinely costly-to-undo/irreversible authorization decision.

Returns guidance opening with "VERDICT: proceed|revise|stop" (act on it: revise/stop name the specific change or blocker) and ending with an "UNVERIFIED CLAIMS RELIED ON:" section marking what it took on faith. Routine task/subtask completion can proceed from direct excerpts and outputs; a bare claim gets revise asking for evidence, not a forced audit. Only authorization of a gated milestone, promotion, launch, expensive (>~30 min), costly-to-undo, or irreversible action may require independent evidence or explicit user judgment. May begin with an [advisor-governance] notice; surface that to the user verbatim. If unavailable, returns Advisor error and you continue.`,
    inputSchema: COMMON_INPUT_SCHEMA,
    annotations: COMMON_ANNOTATIONS,
  },
  async (params: AdvisorInput) => {
    const text = await runConsult("advisor", params, {
      system: ADVISOR_SYSTEM,
      tools: "",
      depth: params.depth,
    });
    return { content: [{ type: "text", text }] };
  },
);

server.registerTool(
  "advisor_verify",
  {
    title: "Consult Fable Advisor (grounded, reads the repo)",
    description: `Like advisor, but the reviewer model gets read-only access (Read/Grep/Glob) to the project directory and verifies your load-bearing claims against the actual files before advising — citing file:line for what it checked.

EXPENSIVE — each call starts one fresh repo-reading Fable session on the user's Claude plan. Use it ONLY when (a) the user explicitly asks for independent verification/audit, (b) a plain advisor identifies a specific load-bearing factual claim that Codex cannot package credibly, or (c) one genuinely costly-to-undo or irreversible authorization decision needs independent evidence. Routine completion, ordinary gates/tests, and plan-negotiation rounds use fresh plain advisor calls.

When you do use it, scope one decision: name the exact claims and file:line pointers to inspect. Pass an additional run/output root only when the decision depends on telemetry outside project_dir. Do not ask for broad exploration when a focused audit will decide the claim.

Returns one fresh grounded review opening with "VERDICT: proceed|revise|stop", with file:line citations for what it verified and an "UNVERIFIED CLAIMS RELIED ON:" section for anything it could not check. May begin with an [advisor-governance] notice; surface that to the user verbatim.`,
    inputSchema: {
      ...COMMON_INPUT_SCHEMA,
      project_dir: z
        .string()
        .min(1)
        .max(1_000)
        .describe("Absolute path to the project root the advisor may read"),
      additional_dirs: z
        .array(z.string().min(1).max(1_000))
        .max(8)
        .optional()
        .describe(
          "Other absolute run/output roots the advisor may read when the scoped decision depends on external telemetry. Nonexistent paths are skipped.",
        ),
    },
    annotations: COMMON_ANNOTATIONS,
  },
  async (params: AdvisorInput & { project_dir: string; additional_dirs?: string[] }) => {
    if (!existsSync(params.project_dir)) {
      return {
        content: [
          {
            type: "text",
            text: `Advisor error (bad_project_dir): '${params.project_dir}' does not exist. Pass the absolute path to the project root, or use the plain advisor tool.`,
          },
        ],
      };
    }
    if (ADVISOR_BACKEND === "api") {
      return {
        content: [
          {
            type: "text",
            text: "Advisor error (verify_unsupported_on_api_backend): advisor_verify requires the claude-cli backend (tool use is not wired for the direct API backend). Use the plain advisor tool, or unset FABLE_ADVISOR_BACKEND.",
          },
        ],
      };
    }
    const text = await runConsult(
      "advisor_verify",
      params,
      {
        system: ADVISOR_VERIFY_SYSTEM,
        tools: VERIFY_TOOLS,
        cwd: params.project_dir,
        additionalDirs: params.additional_dirs,
        depth: params.depth,
      },
      params.project_dir,
    );
    return { content: [{ type: "text", text }] };
  },
);

async function main(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(
    `[fable-advisor] MCP server v${SERVER_VERSION} pid=${process.pid} running via stdio (backend: ${ADVISOR_BACKEND}, advisor model: ${ADVISOR_MODEL})`,
  );
}

main().catch((error) => {
  console.error("[fable-advisor] fatal:", error);
  process.exit(1);
});
