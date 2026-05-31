#!/usr/bin/env python3
"""
developer_agent.py
==================
Local "Developer Agent" built on the Strands Agents SDK using a single-agent
ReAct / tool-calling architecture. Cognitive engine: Claude Sonnet 4.5 via
Amazon Bedrock.

Run:
    pip install -r requirements.txt
    python src/developer_agent.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# --- Strands SDK core ---
from strands import Agent
from strands.models import BedrockModel

# --- Strands community tools ---
# These come from the `strands-agents-tools` PyPI package, import root `strands_tools`.
from strands_tools import editor, shell, python_repl, journal

# --- Local modules ---
from .safeguards import IterationLimiter
from .system_prompt import SYSTEM_PROMPT


# ============================================================================
# 1. Logging — gives real-time visibility into which tools the model picks.
# ============================================================================
def configure_logging() -> None:
    """Wire up structured stderr logging.

    Strands SDK uses the standard `logging` module. Setting the `strands`
    logger to DEBUG/INFO exposes:
      - which tool the model selected
      - the tool's input JSON
      - the tool's return payload
      - event-loop lifecycle transitions

    Combined with the default PrintingCallbackHandler (see Agent init below),
    this gives you both streaming token output AND structured trace events.
    """
    level_name = os.getenv("STRANDS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Quiet down boto noise unless we explicitly asked for DEBUG.
    # Keep botocore at INFO to surface auth/permission errors.
    if level > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)


log = logging.getLogger("dev_agent")


# ============================================================================
# 2. Bedrock model setup
# ============================================================================
def verify_aws_credentials(region: str, profile: str | None) -> None:
    """Pre-flight check: verify AWS credentials work before entering REPL.

    Calls sts:GetCallerIdentity to fail-fast if credentials are missing,
    expired, or lack basic permissions. Better to surface auth errors
    immediately than after the user types their first prompt.

    Raises:
        ClientError: if credentials are invalid or STS call fails.
    """
    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    boto_session = boto3.Session(**session_kwargs)
    sts = boto_session.client("sts")

    try:
        identity = sts.get_caller_identity()
        log.info(
            "AWS credentials verified | account=%s arn=%s",
            identity.get("Account"),
            identity.get("Arn"),
        )
    except ClientError as e:
        log.error("AWS credential verification failed: %s", e)
        raise


def build_bedrock_model() -> BedrockModel:
    """Initialise the BedrockModel.

    Credentials are resolved through boto3's standard chain:
      1. Explicit kwargs on boto3.Session  (we pass profile_name + region_name)
      2. Environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / …)
      3. Shared credential file (~/.aws/credentials)
      4. IAM role (irrelevant for local use)

    IAM permissions required on the caller:
      - bedrock:InvokeModelWithResponseStream  (streaming path)
      - bedrock:InvokeModel                    (non-streaming fallback)

    Sonnet 4.5 requires the `us.` inference-profile prefix because it is only
    served via cross-region inference; the bare model ID returns
    "on-demand throughput isn't supported".
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    profile = os.getenv("AWS_PROFILE")
    model_id = os.getenv(
        "BEDROCK_MODEL_ID",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )

    log.info("Bedrock region=%s profile=%s model=%s", region, profile, model_id)

    # Custom boto3 session so we honour AWS_PROFILE explicitly.
    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    boto_session = boto3.Session(**session_kwargs)

    # Sensible defaults for a local dev tool: short connect, generous read,
    # 3 retries for transient throttles.
    boto_config = BotocoreConfig(
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=5,
        read_timeout=120,
    )

    return BedrockModel(
        model_id=model_id,
        boto_session=boto_session,
        boto_client_config=boto_config,
        temperature=0.2,        # Low temp: this is a coding agent, not a poet.
        streaming=True,         # Stream tokens — the PrintingCallbackHandler renders them live.
        cache_tools="default",  # Reuse tool-spec cache between turns to cut input tokens.
    )


# ============================================================================
# 3. Agent assembly
# ============================================================================
def build_agent(limiter: IterationLimiter) -> Agent:
    """Wire model + tools + system prompt + iteration limiter into one Agent.

    `callback_handler` is left at the SDK default (PrintingCallbackHandler),
    which streams assistant tokens, tool calls, and tool results to stdout
    as they arrive. That covers the "traceability" requirement without any
    extra wiring; the structured logger above covers the rest.
    """
    model = build_bedrock_model()

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            editor,       # file system: view / edit / create / undo
            shell,        # terminal commands
            python_repl,  # live Python evaluation
            journal,      # structured to-do list / scratch pad
        ],
        # callback_handler defaults to PrintingCallbackHandler() — leave it.
        # NOTE: Strands wires the iteration limiter via the hook system.
        # See safeguards.py for the implementation.
    )

    # Attach the loop guard. The hook fires before EVERY model call.
    agent.add_hook(limiter.on_before_model_call)
    log.info(
        "Agent ready. Tools=%s | iteration_budget=%d",
        agent.tool_names,
        limiter.max_iterations,
    )
    return agent


# ============================================================================
# 4. REPL — the interactive CLI loop
# ============================================================================
BANNER = r"""
  ____                 _                           _                    _   
 |  _ \  _____   _____| | ___  _ __   ___ _ __    / \   __ _  ___ _ __ | |_ 
 | | | |/ _ \ \ / / _ \ |/ _ \| '_ \ / _ \ '__|  / _ \ / _` |/ _ \ '_ \| __|
 | |_| |  __/\ V /  __/ | (_) | |_) |  __/ |    / ___ \ (_| |  __/ | | | |_ 
 |____/ \___| \_/ \___|_|\___/| .__/ \___|_|   /_/   \_\__, |\___|_| |_|\__|
                              |_|                      |___/                
                       Developer Agent · Strands + Claude Sonnet 4.5
   Type your request, or 'exit' / 'quit' / Ctrl-D to leave.
"""


def repl(agent: Agent, limiter: IterationLimiter) -> None:
    """Read-eval-print loop with per-turn iteration reset and clean termination."""
    print(BANNER)
    while True:
        try:
            user_input = input("\nyou ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            print("bye.")
            return

        # Reset the per-turn iteration counter BEFORE invoking the agent so
        # each user prompt gets a fresh budget.
        limiter.reset()

        try:
            # The Agent's __call__ streams output via the PrintingCallbackHandler.
            # The returned AgentResult contains stop_reason + metrics.
            result = agent(user_input)
        except (ClientError, BotoCoreError) as aws_err:
            # Bedrock-side failures (throttling, validation, model-not-found, …)
            log.error("AWS Bedrock error: %s", aws_err)
            continue
        except RuntimeError as e:
            # SDK-level failures (tool execution errors, hook cancellations, …)
            log.exception("Agent runtime error: %s", e)
            continue

        # Surface stop reason for observability — especially useful when the
        # iteration limiter has tripped (stop_reason == "cancelled").
        stop_reason = getattr(result, "stop_reason", "unknown")
        if stop_reason == "cancelled":
            print(
                f"\n⚠  Agent cancelled at iteration {limiter.count} "
                f"(budget {limiter.max_iterations}). Refine your prompt and retry."
            )
        else:
            log.debug("Turn complete | stop_reason=%s", stop_reason)


# ============================================================================
# 5. main
# ============================================================================
def main() -> int:
    # Load .env (no error if the file is missing — env vars still work).
    load_dotenv()

    configure_logging()

    # Verify AWS credentials before building the agent.
    region = os.getenv("AWS_REGION", "us-east-1")
    profile = os.getenv("AWS_PROFILE")
    try:
        verify_aws_credentials(region, profile)
    except (ClientError, BotoCoreError) as e:
        log.error("Cannot proceed without valid AWS credentials: %s", e)
        return 1

    # Bypass per-tool confirmation prompts if explicitly opted in. The
    # community tools (shell, editor, python_repl) prompt-on-action by default
    # as a safety net. Senior engineers running a trusted local agent usually
    # want this off; new users should leave it on until comfortable.
    if os.getenv("BYPASS_TOOL_CONSENT", "false").lower() == "true":
        os.environ["BYPASS_TOOL_CONSENT"] = "true"
        log.warning("BYPASS_TOOL_CONSENT=true — tools will execute without prompting.")

    try:
        max_iter = max(1, int(os.getenv("DEV_AGENT_MAX_ITERATIONS", "25")))
    except ValueError:
        max_iter = 25
        log.warning("Invalid DEV_AGENT_MAX_ITERATIONS; defaulting to %d", max_iter)

    limiter = IterationLimiter(max_iterations=max_iter)
    agent = build_agent(limiter)

    try:
        repl(agent, limiter)
    finally:
        # Releases MCP clients & any other tool-side resources cleanly.
        agent.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
