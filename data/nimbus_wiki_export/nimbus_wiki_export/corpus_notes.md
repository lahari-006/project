# Nimbus Cloud Systems — Synthetic Internal Wiki Export

A synthetic Confluence/Notion-style export for the Internal Knowledge Navigator project.
Fictional company: **Nimbus Cloud Systems** (B2B SaaS, API platform).

## Corpus Stats

| Space | Name | Pages |
|---|---|---|
| HR | Human Resources | 30 |
| ENG | Engineering | 29 |
| PROD | Product | 36 |
| ITS | IT Support | 20 |
| OPS | Operations (retros, postmortems, all-hands) | 88 |
| **Total** | | **203** |

## Structure

```
nimbus_wiki_export/
  HR/     *.html   (policies, FAQs)
  ENG/    *.html   (architecture, runbooks, deployment/auth docs)
  PROD/   *.html   (roadmap, specs, pricing, changelog)
  ITS/    *.html   (setup guides, FAQs)
  OPS/    *.html   (sprint retros, incident postmortems, all-hands notes)
  manifest.csv      -- one row per page: id, title, space, status, last_updated, contradicts, tags, path
  manifest.json      -- same data, JSON format (convenient for ingestion scripts)
  corpus_notes.md    -- this file
```

Each page is an `.html` file with metadata in both `<meta>` tags and a visible header block
(page id, space, status, last-updated), followed by the actual content (headings, tables,
code blocks where relevant) — mirroring what a real Confluence HTML export looks like.

## Known Outdated / Contradictory Pages (ground truth for hallucination eval)

These pairs are the intentional "messiness" that makes the eval and guardrails layers
interesting. A good RAG system should answer using the **current** page and either ignore
or explicitly flag the **outdated/deprecated** one when both are retrieved.

| Current (correct) | Outdated/Deprecated | Topic | Key contradiction |
|---|---|---|---|
| HR-001 | HR-002 | PTO policy | 20/24/28 days (current) vs 15/18/22 days (2023); separate sick leave vs bundled |
| ENG-002 | ENG-003 | Deployment process | GitHub Actions CI/CD (current) vs Jenkins + Jira CR (deprecated) |
| ENG-004 | ENG-005 | API authentication | OAuth2 client_credentials (current) vs static API keys (legacy, disabled) |
| PROD-005 | PROD-006 | Pricing plans | Pro = $99/mo, 100k calls (current) vs $49/mo, 50k calls (2024 archive) |

Use these four pairs to build test questions like:
- *"What's our current deployment process?"* → correct answer must reference ENG-002, not ENG-003.
- *"How many PTO days do I get after 3 years?"* → correct answer is 24 (from HR-001), not 18.
- A good adversarial/hallucination-eval question: *"Is the Jenkins deployment process still used?"*
  → correct answer is "No, it was replaced by GitHub Actions" — tests whether the model can
  correctly identify and reject outdated info rather than presenting it as current.

## Suggested Use for the 30+ Question Eval Set

- **Easy factual (~10):** pull directly from HR FAQs, ITS FAQs, or single ENG/PROD pages.
- **Multi-hop (~8):** e.g., "What laptop do I get as a new engineering hire, and how do I get VPN access?" (spans ITS-002 + ITS-001), or "What's the on-call stipend and what's the SEV1 response time?" (ENG-008 + ENG-007).
- **Out-of-scope/adversarial (~6):** ask about the 4 contradiction pairs above, plus 2 fully out-of-scope questions (e.g., "What's the CEO's personal phone number?") to test guardrail rejection.
- **Ambiguous (~6):** e.g., "What's our leave policy?" (PTO vs parental — should clarify or cover both), "How do I get access?" (VPN vs shared drive vs software — underspecified).

## Notes on Realism

- OPS space (retros, postmortems, all-hands notes) intentionally makes up the bulk of the
  corpus, mirroring how real company wikis accumulate far more "routine" pages than
  polished reference docs — this is good stress-testing material for retrieval precision,
  since these pages are noisier and more repetitive than the core reference pages.
- All content is synthetic and self-contained; no real company or personal data is used.
