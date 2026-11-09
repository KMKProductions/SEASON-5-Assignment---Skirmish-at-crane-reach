# Using the LLM API

Some assignments may ask your game agent to call an LLM. First, request a development key to authenticate your agent with the course website. The key works only for this environment's submission-open season (the season currently accepting submissions) and stops working when that season closes for submissions.

## Set up development access

1. Sign in to the course website and open **My Agents** in the sidebar. In the row for this environment's submission-open season, select **Get or test key** to open the **LLM Playground**. Find the same season under **Development key** and select **Create development key**. A dialog shows the credential once; select **Copy .env** to copy both lines.

2. Create a file named `.env` in the project root and paste the copied lines. The result looks like this, with your real values:

   ```dotenv
   OPENAI_BASE_URL=https://course.example/api/llm/v1
   OPENAI_API_KEY=the-returned-development-key
   ```

   The template's `.env.example` shows the same two names with empty values to fill in.

3. Send a test request. The command below uses the `small` tier if it is enabled for this environment's submission-open season. Use a tier enabled for that season.

   ```console
   python -m sandbox llm
   ```

   The public tier aliases are `small`, `medium`, and `large`, but this environment's submission-open season may expose only some of them. To test another enabled tier, pass its name as an argument:

   ```console
   python -m sandbox llm medium
   ```

4. Use exactly the same tier name in your agent code, such as `model="medium"`.

A successful test prints the chosen tier, the response, and the token use. It never prints your API key.

You can also test in the browser. On the **LLM Playground**, paste the key and select **Check**. A key you just created there is filled in and checked for you. The page shows the key's season and remaining budget, then lets you try a prompt: the **System prompt** sets who the model is and how it answers, and the **User prompt** is the message it answers. Checking is free, but each prompt counts against your development budget like any other call.

Never commit `.env` or paste a key into source code, a prompt, an issue, or a chat. The template's `.gitignore` excludes `.env`, but still check `git status` before every commit. If you lose a key or think it was exposed, select **Rotate development key** on the **LLM Playground**, which issues a new key and invalidates the old one.

## Use the model in your agent

### Example

This Hearts example uses the template's `openai` and `python-dotenv` packages to read the base URL and key from your `.env` file. It makes one request inside `act`, accepts only a numbered legal-card choice, and otherwise returns the first legal card:

```python
import os

from dotenv import load_dotenv
from openai import OpenAI
from sandbox.cards import card_name, legal_cards, play

class Agent:
    def __init__(self) -> None:
        load_dotenv()
        self.client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=0,
        )

    def reset(self, seed: int, observation) -> None:
        pass

    def act(self, observation) -> int:
        legal = legal_cards(observation)
        fallback = play(legal[0])
        choices = "\n".join(
            f"{number}: {card_name(card)}" for number, card in enumerate(legal)
        )
        prompt = f"Choose a Hearts card. Reply only with its number:\n{choices}"
        try:
            response = self.client.chat.completions.create(
                model="small",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            reply = (response.choices[0].message.content or "").strip()
            choice = int(reply)
            if 0 <= choice < len(legal):
                return play(legal[choice])
        except Exception:
            pass
        return fallback
```

Streaming completions are not supported. Use a normal request with `stream=False`.

### Use a model across ticks

A normal request pauses the current `act`, `chat`, or `learn` method until the model replies. If your agent can carry on without the reply, the template's `BackgroundLLM` helper lets it start the request now and collect the reply in a later method call. The current turn or tick (one game step) does not have to wait.

Do not create background threads yourself. The helper owns its thread and builds its client on the first `request`, then reuses it. It serves plain-text chat completions only, so use the standard synchronous client when you need tools, structured response formats, or another advanced completion shape.

This example answers a message once the reply is ready, often a few ticks after the message that prompted it:

```python
from sandbox.llm import BackgroundLLM

class Agent:
    def __init__(self) -> None:
        self.llm = BackgroundLLM()

    def chat(self, inbox):
        reply = self.llm.response()
        if reply:
            return [{"to": None, "text": reply}]
        if inbox and not self.llm.requesting:
            self.llm.request(
                model="small",
                messages=[
                    {
                        "role": "user",
                        "content": f"Reply in one short sentence to: {inbox[-1]['text']}",
                    }
                ],
            )
        return []
```

Check for a completed reply before requesting another. `response()` returns a completed reply once, and `None` while a request is running or when no unread reply is waiting. The `requesting` check limits this instance to one request at a time: a second `request()` while it is busy returns `False` and leaves the original running. Check `error` if a request fails.

The same pattern works in `act`: start a request for a plan on one tick, continue with a legal fallback, and use the reply on a later tick. A background response still counts against the model budget. If you send it through `chat`, keep it within the environment's message-length limit.

## Model tiers and budget

LLMs cost more than most other techniques used in game AI, so use them efficiently. They read and write text in **tokens**, each a short piece of text, and your usage is measured in the same unit, weighted by the size and capability of the model:

| Tier     | Default budget price |
| -------- | -------------------- |
| `small`  | 1 per token          |
| `medium` | 2 per token          |
| `large`  | 4 per token          |

Your model use is metered by a token budget (how much you can use LLMs) and a request-rate limit (how often you can send LLM requests).

- Your development budget is fixed for the season. Check its usage at **My Agents**.
- Automated leaderboard runs are metered separately. In each run, each agent-controlled player receives a temporary key and its own allowance.

## Troubleshooting

LLM calls can fail even when your code is correct. Every model-assisted path through `act` needs a legal fallback action.

Only successful calls spend the budget. The LLM service handles retries, so you do not need your own retry loop.

| Problem | What to do |
| --- | --- |
| `budget_exceeded` | Use the legal fallback and reduce model use. Your development budget is fixed for the season. |
| `model_not_allowed` | Choose a tier enabled for this environment's submission-open season. |
| A temporary error or `meter_unavailable` | Use the legal fallback and try again later. Tell course staff if the problem continues. |
| A request error or unusable response | Use the legal fallback. Check your request and response parsing, then test again. |

## Tracing your LLM usage

Your prompts and the model's replies are never public: only you and the site operators can read them.

Replays of official games show each successful call, with its tier, token counts, and budget cost, but never the text. Development calls do not appear in replays or leaderboards; review them from **My Agents**.

See [Agent interface](https://vox-deorum.github.io/game-sandbox/students/agent-interface/#llm-calls) for where model calls fit in an agent turn and the [LLM API specification](https://vox-deorum.github.io/game-sandbox/specs/llm/) for the complete platform rules.
