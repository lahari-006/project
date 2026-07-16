"""
Langfuse observability for the RAG pipeline.
 
Wraps every query in a Langfuse trace with:
    - retrieval span (latency, #chunks, top distances)
    - guardrails spans (input/output pass-fail)
    - generation span (Bedrock call: latency, input/output tokens, cost, model)
    - an error event if any stage raises / guardrails block the request
 
Also exposes a shareable Langfuse dashboard URL per trace (QueryTrace.trace_url),
so a UI (e.g. Streamlit) can link straight to the trace for debugging.
 
NOTE: written against Langfuse Python SDK v4 (tested against 4.7.1). v4
introduced the "observation-centric" data model: correlating attributes
(user_id, session_id, tags) must now live on every observation instead of
just the trace, so `update_current_trace(user_id=..., tags=...)` (v3) was
removed in favor of `propagate_attributes()`.
 
IMPORTANT: `propagate_attributes()` is a top-level function imported from
the `langfuse` package — it is NOT a method on the `Langfuse` client
instance. Import it directly:
    from langfuse import propagate_attributes
 
Free-form `metadata` (not user_id/session_id/tags) is still set per-span via
`.update(metadata=...)` as before — that part didn't change.
"""
import time
from contextlib import contextmanager
 
from app import config
 
_langfuse_client = None
 
# USD per 1K tokens (input, output) — update if you change GENERATION_MODEL.
PRICING_PER_1K = {
    "amazon.nova-micro-v1:0": (0.000035, 0.00014),
    "amazon.nova-lite-v1:0": (0.00006, 0.00024),
    "amazon.nova-pro-v1:0": (0.0008, 0.0032),
}
 
 
def get_client():
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        return None
    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
    except Exception as e:  # pragma: no cover - observability must never break the app
        print(f"[langfuse] disabled: {e}")
        _langfuse_client = None
    return _langfuse_client
 
 
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICING_PER_1K.get(model, (0.003, 0.015))
    return round((input_tokens / 1000) * in_rate + (output_tokens / 1000) * out_rate, 6)
 
 
class QueryTrace:
    """One trace per user query. Usable even when Langfuse isn't configured
    (in which case it just measures latency locally and `trace_url` stays
    None), so the app never hard-depends on Langfuse being set up.
 
    Public interface: `.span()`, `.log_generation()`, `.finish()`,
    plus `.trace_id` / `.trace_url` for linking out to the Langfuse UI.
    """
 
    def __init__(self, query: str, user_id: str = "demo-user"):
        self.query = query
        self.user_id = user_id
        self.client = get_client()
        self.t_start = time.time()
        self.spans = {}
        self.error = None
        self.trace_id = None
        self.trace_url = None
 
        # Both context managers below are manually entered/exited (instead
        # of a single `with` block) because QueryTrace's lifetime spans
        # multiple method calls (init -> several .span() calls -> finish).
        #
        # propagate_attributes() must be entered BEFORE the root span is
        # created so the root span (and everything nested under it) picks
        # up user_id/tags — only spans started after entering the context
        # inherit the propagated attributes.
        self._attr_cm = None
        self._root_cm = None
        self._root_span = None
 
        if self.client:
            from langfuse import propagate_attributes
 
            self._attr_cm = propagate_attributes(
                user_id=user_id,
                tags=["rag", "internal-knowledge-navigator"],
            )
            self._attr_cm.__enter__()
 
            self._root_cm = self.client.start_as_current_observation(
                as_type="span",
                name="knowledge-navigator-query",
                input={"query": query},
            )
            self._root_span = self._root_cm.__enter__()
 
            # Grab the trace id + a shareable dashboard link right away so
            # callers can show/log it even before the query finishes.
            try:
                self.trace_id = self._root_span.trace_id
                self.trace_url = self.client.get_trace_url(trace_id=self.trace_id)
            except Exception as e:  # pragma: no cover
                print(f"[langfuse] could not resolve trace_url: {e}")
 
    @contextmanager
    def span(self, name: str, **metadata):
        t0 = time.time()
        span_cm = None
        span_obj = None
        if self.client:
            span_cm = self.client.start_as_current_observation(
                as_type="span", name=name, input=metadata
            )
            span_obj = span_cm.__enter__()
        try:
            yield span_obj
        except Exception as e:
            self.error = str(e)
            if span_obj:
                span_obj.update(level="ERROR", status_message=str(e))
            raise
        else:
            elapsed = time.time() - t0
            self.spans[name] = elapsed
            if span_obj:
                span_obj.update(output={"latency_seconds": round(elapsed, 3)})
        finally:
            if span_cm:
                span_cm.__exit__(None, None, None)
 
    def log_generation(self, model, prompt, completion, input_tokens, output_tokens):
        cost = estimate_cost(model, input_tokens, output_tokens)
        if self.client:
            with self.client.start_as_current_observation(
                as_type="generation",
                name="answer-generation",
                model=model,
                input=prompt,
            ) as gen:
                gen.update(
                    output=completion,
                    usage_details={
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens,
                    },
                    metadata={"estimated_cost_usd": cost},
                )
        return cost
 
    def finish(self, answer: str, guard_blocked: bool = False):
        total_latency = time.time() - self.t_start
        if self._root_span:
            # Root span's output doubles as the trace output (same as v3).
            # Free-form metadata (not user_id/tags) still goes via .update().
            self._root_span.update(
                output={"answer": answer},
                metadata={
                    "total_latency_seconds": round(total_latency, 3),
                    "span_latencies": {k: round(v, 3) for k, v in self.spans.items()},
                    "guardrail_blocked": guard_blocked,
                    "error": self.error,
                },
            )
        if self._root_cm:
            self._root_cm.__exit__(None, None, None)
        if self._attr_cm:
            self._attr_cm.__exit__(None, None, None)
        if self.client:
            self.client.flush()
        return total_latency
 
