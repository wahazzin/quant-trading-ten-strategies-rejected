"""
llm.py -- LLM providers behind one tiny interface:

    complete(system, user, ctx) -> {"text", "model_id", "tokens_in", "tokens_out", "latency_s"}

  * AnthropicLLM: the real model, called over plain HTTPS (no SDK dependency). The model id the
    API actually RETURNS is logged every call, so a silent model change is detectable
    (PREREGISTRATION.md section 8).
  * MockLLM: deterministic, free, offline. Used for tests and dry runs only. It follows a dumb
    fixed rule so the pipeline can be exercised end to end. Its "decisions" mean nothing and it
    can never be locked in as the experiment (lock.py refuses).

The API key comes from the environment (ANTHROPIC_API_KEY), never from a file in the repo.
"""
import json
import os
import time

import requests

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


class LLMError(Exception):
    pass


class AnthropicLLM:
    def __init__(self, cfg):
        self.c = cfg["llm"]
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise LLMError("ANTHROPIC_API_KEY is not set")

    def complete(self, system, user, ctx=None):
        body = {"model": self.c["model"], "max_tokens": self.c["max_tokens"],
                "temperature": self.c["temperature"], "system": system,
                "messages": [{"role": "user", "content": user}]}
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        t0 = time.time()
        last = None
        for attempt in range(3):
            try:
                r = requests.post(ANTHROPIC_URL, headers=headers, json=body,
                                  timeout=self.c["timeout_seconds"])
                if r.status_code in (429, 500, 502, 503, 529):
                    last = f"HTTP {r.status_code}"
                    time.sleep(5 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                j = r.json()
                text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
                u = j.get("usage", {})
                return {"text": text, "model_id": j.get("model"), "stop_reason": j.get("stop_reason"),
                        "tokens_in": u.get("input_tokens"), "tokens_out": u.get("output_tokens"),
                        "latency_s": round(time.time() - t0, 2)}
            except requests.RequestException as e:
                last = str(e)
                time.sleep(5 * (attempt + 1))
        raise LLMError(f"Anthropic API failed after retries: {last}")


class OpenAICompatLLM:
    """FREE path. Calls any OpenAI-compatible chat API (Groq, OpenRouter, ...).

    WHY an open-weights model (gpt-oss-120b) on free tiers:
      * costs $0 (free-tier quotas are ~50-1,000 requests/day; we need 4)
      * open weights = a fixed file that can't be silently updated, unlike proprietary models
      * if one provider drops its free tier, the SAME weights run elsewhere, so the experiment
        continues without changing the model under test.
    The endpoints list is tried IN ORDER, and every endpoint must serve the same model. We never
    fall back to a different model: that would mix two models inside one arm. If every endpoint
    fails, the cycle is an AI_FAULT (logged, AI does nothing)."""

    def __init__(self, cfg):
        self.c = cfg["llm"]
        self.endpoints = [e for e in self.c["endpoints"] if os.environ.get(e["key_env"])]
        if not self.endpoints:
            raise LLMError("no API key set for any endpoint: " +
                           ", ".join(e["key_env"] for e in self.c["endpoints"]))

    def _call(self, ep, system, user):
        body = {"model": ep["model"], "temperature": self.c["temperature"],
                "max_tokens": self.c["max_tokens"],
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        body.update(ep.get("extra_body", {}))
        headers = {"Authorization": f"Bearer {os.environ[ep['key_env']]}",
                   "Content-Type": "application/json"}
        r = requests.post(ep["url"], headers=headers, json=body, timeout=self.c["timeout_seconds"])
        if r.status_code != 200:
            raise LLMError(f"{ep['name']} HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        ch = (j.get("choices") or [{}])[0]
        u = j.get("usage") or {}
        return {"text": (ch.get("message") or {}).get("content") or "",
                "model_id": j.get("model"), "provider": ep["name"],
                "stop_reason": ch.get("finish_reason"),
                "tokens_in": u.get("prompt_tokens"), "tokens_out": u.get("completion_tokens")}

    def complete(self, system, user, ctx=None):
        t0, errors = time.time(), []
        for ep in self.endpoints:
            for attempt in range(2):
                try:
                    out = self._call(ep, system, user)
                    out["latency_s"] = round(time.time() - t0, 2)
                    out["endpoint_errors"] = errors
                    return out
                except (LLMError, requests.RequestException) as e:
                    errors.append(str(e)[:300])
                    time.sleep(4 * (attempt + 1))
        raise LLMError("all endpoints failed: " + " | ".join(errors))


class MockLLM:
    """Rule: outlook = sign of 24h return; hold a small BTC position with a thesis. Meaningless
    by design -- it exists to exercise the plumbing, not to trade."""

    def complete(self, system, user, ctx=None):
        cycle_id, snap, weights = ctx["cycle_id"], ctx["snapshot"], ctx["weights"]
        outlook = {}
        for a, v in snap["assets"].items():
            r = v["features"]["ret_24h"] or 0.0
            outlook[a] = 1 if r > 0.01 else (-1 if r < -0.01 else 0)
        decisions = []
        if weights.get("BTC-USD", 0.0) < 1e-4:
            decisions.append({"asset": "BTC-USD", "action": "BUY", "target_weight": 0.10,
                              "stop_loss_pct": 12, "confidence": 50,
                              "reasons": ["mock: plumbing test position"],
                              "invalidation": ["mock: none, this is a test"]})
        else:
            decisions.append({"asset": "BTC-USD", "action": "HOLD", "confidence": 50,
                              "reasons": ["mock: hold"], "invalidation": ["mock: none"]})
        out = {"cycle_id": cycle_id, "outlook": outlook, "decisions": decisions,
               "thesis_updates": [{"asset": "BTC-USD", "status": "neutral",
                                   "summary": "mock thesis", "reasons": ["mock"],
                                   "invalidation": ["mock"]}],
               "portfolio_note": "mock provider -- not a real decision"}
        return {"text": json.dumps(out), "model_id": "mock", "stop_reason": "end_turn",
                "tokens_in": len(system + user) // 4, "tokens_out": 300, "latency_s": 0.0}


def make_llm(cfg, provider=None):
    p = provider or cfg["llm"]["provider"]
    if p == "openai_compat":
        return OpenAICompatLLM(cfg)
    if p == "anthropic":
        return AnthropicLLM(cfg)
    if p == "mock":
        return MockLLM()
    raise LLMError(f"unknown provider {p}")
