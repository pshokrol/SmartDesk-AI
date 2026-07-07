# SmartDesk AI

An IT & HR help desk agent for **BoldAgent** (a fictional company). Employees can ask
IT/HR questions answered from a curated knowledge base, and when the knowledge base
can't confidently answer, the agent collects the details and creates a Jira support
ticket with the employee's explicit confirmation required before anything is created.

## Architecture
```
Employee message
│
▼
LangGraph Agent (langchain.agents.create_agent)
├─ system prompt: routing rules + safety rules
├─ memory: MemorySaver checkpointer (remembers context across turns)
│
├── search_knowledge_base(question) ──► RAG pipeline:
│                                        1. retrieve_with_scores (Chroma vector search)
│                                        2. is_confident_match (distance threshold)
│                                        3. assess_answer (LLM self-assessment,
│                                           structured output via RAGAssessment)
│                                        → grounded answer, or "no confident answer"
│
├── create_support_ticket(...) ──► Jira REST API v3
│                                   → creates ticket, links email→ticket_key locally
│
└── check_ticket_status(email) ──► looks up ticket_key(s) for email (local SQLite),
then fetches live status from Jira
                │
                ▼
          Gradio chat UI (src/app.py)

```

**Why three layers of confidence, not one:** a single distance threshold isn't enough,
retrieval can look confident (good vector-similarity score) while the actual retrieved
content doesn't answer the specific question asked (e.g. a general sick-leave policy
retrieved for a part-time-specific question). The LLM self-assessment step catches this
by explicitly judging whether the retrieved context *actually* answers the question,
not just whether it's topically related.

**Why a local SQLite ticket-link store:** employees are identified by email in
conversation, but they aren't necessarily Jira users/reporters we can query by. The
local `ticket_links` table (`src/ticket_store.py`) records email→ticket_key at creation
time, so status checks can look up "which tickets did this email create" before asking
Jira for live status on each one.

Retrieval in this project is **dense-only** (Chroma vector similarity search over
locally-embedded knowledge base entries), no sparse/keyword (BM25) layer.

## Project structure
```
data/kb/qa_pairs.json      37 Q&A entries (IT Support, HR Policies, Onboarding, Payroll)
                           with deliberate gaps left uncovered (parking, monitor flicker,
                           printer setup, etc.) so escalation has real things to catch
src/
ingest.py             load KB → build Documents → embed into persistent Chroma store
rag_agent.py           retrieve_with_scores, is_confident_match, assess_answer,
answer_question (the unified RAG entry point)
jira_tool.py           create_ticket, get_ticket_status (Jira REST API v3)
ticket_store.py        local SQLite email→ticket_key mapping (duplicate-safe)
tools.py                LangChain @tool wrappers around the above, for agent use
graph.py               build_agent() — the LangGraph agent, tools + system prompt + memory
app.py                 Gradio chat UI
tests/                    manual verification scripts written throughout development
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd smartdesk-ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Notes |
|---|---|
| `OPENAI_API_KEY` | From platform.openai.com/api-keys |
| `JIRA_BASE_URL` | **Must be the API gateway URL, not your site domain** — see note below |
| `JIRA_SITE_URL` | Your actual site domain (e.g. `https://your-site.atlassian.net`), used only for building human-clickable ticket links |
| `JIRA_EMAIL` | Your Atlassian account email |
| `JIRA_API_TOKEN` | Created via "Create API token with scopes" (`read:jira-work`, `write:jira-work`) |
| `JIRA_PROJECT_KEY` | Your Jira project key (e.g. `KAN`) |

**Important Jira gotcha:** newer *scoped* API tokens (the kind Atlassian now recommends)
only work against `https://api.atlassian.com/ex/jira/<cloudId>` — **not** your site's own
domain. Get your Cloud ID with:
```bash
curl https://your-site.atlassian.net/_edge/tenant_info
```
Then set `JIRA_BASE_URL=https://api.atlassian.com/ex/jira/<cloudId>`. This is truely
under-documented by Atlassian and easy to lose an hour to, using the site domain
directly with a scoped token returns a misleading 404, not an auth error.

### 3. Build the knowledge base index

```bash
python -m src.ingest --rebuild
```

Downloads a small local embedding model on first run (needs normal internet access).

### 4. Run

```bash
python -m src.app
```

Open the printed local URL (typically `http://127.0.0.1:7860`).

## Design notes and known limitations

- **Category labels can't contain spaces in Jira.** `create_ticket` sanitizes this
  (`category.upper().replace(" ", "_")`) so `"IT Support"` becomes the valid label
  `IT_SUPPORT` discovered when the agent picked a two-word category on its own and
  Jira rejected the request.
- **The agent must never fabricate required fields.** Early testing showed the LLM
  would invent a placeholder email (`user@example.com`) to satisfy `create_support_ticket`'s
  required `email` parameter rather than pausing to ask the employee. The system prompt
  explicitly forbids this and requires the agent to stop and ask when information is
  missing, rather than proceeding with invented data.
- **Single fixed conversation thread in the Gradio UI.** `chat_fn` uses one fixed
  `thread_id` for memory, meaning this UI is built for single-user local demo use, not
  concurrent multi-user sessions. Acceptable simplification for this project's scope.
- **Dense retrieval only** (no sparse/BM25 hybrid layer). Retrieval quality was verified
  against real KB queries, paraphrased queries, and deliberate knowledge-gap queries
  (see `tests/manual_score_check.py`).

## Testing

The `tests/` folder contains manual verification scripts built incrementally during
development, covering: KB ingestion and dedup, retrieval score thresholds, LLM
self-assessment, Jira ticket creation and status lookup, local ticket-store dedup,
and full multi-turn agent conversations (KB answering, ticket creation with
confirmation-gating, and status checks).