# Developer Agent

A local **Single-Agent / ReAct** coding assistant built on the [Strands Agents SDK](https://strandsagents.com), powered by Claude Sonnet 4.5 via Amazon Bedrock.

## Architecture

```
              ┌──────────────────────────────────────────────────────┐
              │                Developer Agent (single)              │
              │                                                      │
   user ───▶  │   ┌────────────┐    ┌────────────────────────────┐   │
              │   │ system     │    │     Strands event loop     │   │
              │   │ prompt     │──▶ │   (ReAct: reason → act)    │   │
              │   └────────────┘    └────────────┬───────────────┘   │
              │                                  │                   │
              │   Hook: IterationLimiter   ──┐   │                   │
              │   (cancels on budget breach) │   │                   │
              │                              ▼   ▼                   │
              │                      ┌──────────────────┐            │
              │                      │   BedrockModel   │            │
              │                      │  (boto3 Converse)│            │
              │                      └────────┬─────────┘            │
              │                               │                      │
              │  ┌────────────────────────────┴──────────────────┐   │
              │  │              Tool registry                    │   │
              │  │   editor · shell · python_repl · journal      │   │
              │  └───────────────────────────────────────────────┘   │
              └──────────────────────────────────────────────────────┘
                               ▲
                               │ bedrock:InvokeModelWithResponseStream
                               ▼
                  ┌──────────────────────────┐
                  │   Amazon Bedrock         │
                  │   Claude Sonnet 4.5      │
                  └──────────────────────────┘
```

**No supervisor, no swarm, no graph.** A single Agent owns the loop.

## Setup

```bash
# 1. Create venv
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify Bedrock access (one-off sanity check)
aws bedrock list-foundation-models --region "$AWS_REGION" \
  --query "modelSummaries[?contains(modelId, 'claude-sonnet-4-5')].modelId"

# 4. Run
python src/developer_agent.py
```

## Required IAM permissions

### User vs Role — which one do I attach this to?

For a **local developer machine** authenticated with `aws configure` (static keys), attach the policy **directly to your IAM user**.

### Policy document

Save this as `bedrock-dev-agent-policy.json` in the project root:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream"
    ],
    "Resource": [
      "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-*",
      "arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-sonnet-4-5-*"
    ]
  }]
}
```

### Create and attach via AWS CLI

```bash
# 1. Find your account ID and the IAM username you're authenticated as
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IAM_USER=$(aws sts get-caller-identity --query Arn --output text | awk -F'/' '{print $NF}')
echo "Account: $ACCOUNT_ID  | User: $IAM_USER"

# 2. Create the customer-managed policy
aws iam create-policy \
  --policy-name BedrockDeveloperAgentAccess \
  --description "Bedrock invoke access for the local Developer Agent" \
  --policy-document file://json/bedrock-dev-agent-policy.json

# 3. Attach the policy to your IAM user
aws iam attach-user-policy \
  --user-name "$IAM_USER" \
  --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/BedrockDeveloperAgentAccess"

# 4. Verify it's attached
aws iam list-attached-user-policies --user-name "$IAM_USER"
```

### Bedrock console: model access

IAM permissions alone are not enough — every Bedrock model is gated by an additional **Model catalog** opt-in per account/region. Once, manually:

1. Open the Bedrock console in the region from your `.env` (e.g. `us-east-1`).
2. Left nav → **Discover** → **Model catalog**.
3. Tick **Claude Sonnet 4.5** → **Open in playground** → and chat.
4. Status changes to *Access granted* within a minute for Anthropic models.

## Loop safety

The current Strands `Agent` does **not** have a `max_iterations` parameter — that
parameter only exists on multi-agent constructs (`Swarm`, `Graph`). For a single
agent the official escape hatch is `agent.cancel()`.

This project enforces an iteration budget cleanly via a hook on
`BeforeModelCallEvent` (see `src/safeguards.py`). When the budget is exceeded
the hook calls `agent.cancel()`, the loop stops at the next checkpoint, and
the user sees a `stop_reason="cancelled"` notice.

Tune the budget via `DEV_AGENT_MAX_ITERATIONS` in `.env`.

## Traceability

Two complementary mechanisms expose what the agent is doing:

1. **Streaming output** — the default `PrintingCallbackHandler` prints assistant
   tokens, tool selections, and tool results to stdout as they arrive.
2. **Structured logging** — set `STRANDS_LOG_LEVEL=DEBUG` to see the full
   internal trace (model calls, tool dispatch, hook events) on stderr.

## File layout

```
genai-developer-agent/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── developer_agent.py   # entry point: BedrockModel + Agent + REPL
    ├── safeguards.py        # IterationLimiter hook (loop cap)
    └── system_prompt.py     # the SRE / DevOps system prompt
```
