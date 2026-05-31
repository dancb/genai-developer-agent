#!/usr/bin/env python3
"""
Local Developer Agent built on Strands SDK + Claude Sonnet 4.5 via Bedrock.
Run: python src/developer_agent.py
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

from strands import Agent
from strands.models import BedrockModel
from strands_tools import editor, shell, python_repl, journal

from safeguards import IterationLimiter
from system_prompt import SYSTEM_PROMPT


# ============================================================================
# Logging
# ============================================================================
def configure_logging() -> None:
    """Configure structured logging for tool visibility and event tracing."""
    level_name = os.getenv("STRANDS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if level > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)


log = logging.getLogger("dev_agent")


# ============================================================================
# Bedrock model setup
# ============================================================================
def verify_aws_credentials(region: str, profile: str | None) -> None:
    """Verify AWS credentials via STS before entering REPL. Fail-fast on auth errors."""
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
    """Initialize BedrockModel with boto3 credential chain and cross-region inference profile."""
    region = os.getenv("AWS_REGION", "us-east-1")
    profile = os.getenv("AWS_PROFILE")
    model_id = os.getenv(
        "BEDROCK_MODEL_ID",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )

    log.info("Bedrock region=%s profile=%s model=%s", region, profile, model_id)

    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    boto_session = boto3.Session(**session_kwargs)

    boto_config = BotocoreConfig(
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=5,
        read_timeout=120,
    )

    return BedrockModel(
        model_id=model_id,
        boto_session=boto_session,
        boto_client_config=boto_config,
        temperature=0.2,
        streaming=True,
        cache_tools="default",
    )


# ============================================================================
# Agent assembly
# ============================================================================
def build_agent(limiter: IterationLimiter) -> Agent:
    """Assemble Agent with model, tools, system prompt, and iteration limiter."""
    model = build_bedrock_model()

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            editor,
            shell,
            python_repl,
            journal,
        ],
    )

    agent.add_hook(limiter.on_before_model_call)
    log.info(
        "Agent ready. Tools=%s | iteration_budget=%d",
        agent.tool_names,
        limiter.max_iterations,
    )
    return agent


# ============================================================================
# REPL
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
    """Interactive loop with per-turn iteration reset."""
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

        limiter.reset()

        try:
            result = agent(user_input)
        except (ClientError, BotoCoreError) as aws_err:
            log.error("AWS Bedrock error: %s", aws_err)
            continue
        except RuntimeError as e:
            log.exception("Agent runtime error: %s", e)
            continue

        stop_reason = getattr(result, "stop_reason", "unknown")
        if stop_reason == "cancelled":
            print(
                f"\n⚠  Agent cancelled at iteration {limiter.count} "
                f"(budget {limiter.max_iterations}). Refine your prompt and retry."
            )
        else:
            log.debug("Turn complete | stop_reason=%s", stop_reason)


# ============================================================================
# main
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
        # Releases resources cleanly.
        agent.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
