<!-- kit: governance v0.1.0 -->
# 0002 — Missions

- **Status:** proposed
- **Date:** 2026-11-16
- **Proposed by:** Grok Bot (agent), at the maintainer's request
- **Ratified by:**
- **Affects:** constitution §1; PRD §1.1; feature OCRMD-007
- **Supersedes:** none
- **Related:** [PRD.md](../../PRD.md) §1.1; feature OCRMD-007

## Context

Constitution §1 Mission was a product description (or, for a new constitution, had not been written yet), not a three-mission direction statement. The fleet now requires every repo's constitution §1 to be one direction statement followed by exactly three numbered missions, adopted through a numbered amendment the maintainer ratifies by merging. The maintainer asked for those missions to be comprehensive and epic: ocrmd as the ultimate open home for all things document digitization. Mission 1 follows the maintainer's wording, adapted to this domain: provide data, tools, resources and transparency for all stakeholders, domain practitioners and the general public. The ambition is the direction, not a claim that the work is already finished. The missions are aligned with this repo's existing constitution and PRD.

## Amendment

Replace constitution §1 with the following, retitled **Missions**:

ocrmd's direction is to become the ultimate open home for all things document digitization — data, tools, resources and transparency, gathered in one place that anyone can inspect and reuse.

1. **Provide data, tools, resources and transparency for all stakeholders, archivists, librarians, scholars and digitization practitioners, and the general public.** Anyone turning paper into text is a first-class audience. The aim is a shared, inspectable commons: the same readings, the same diffs, and the same inventory for everyone.
2. **Turn physical books and loose pages into cross-checked Markdown, plain text and JSON, and flag disagreements rather than silently picking a winner.** Every page is read twice — local OCR engines and a vision model, Claude by default — from a webcam, a folder of photos or a PDF, so provenance stays visible on the page.
3. **Keep a searchable inventory of everything scanned, work fully offline when the model pass is deferred, and send only page images off the machine, and only to the configured provider.** Accuracy, disagreement markers and local-first operation are the default path, not a later mode.

These missions set the course; they do not claim the work is finished. They serve the PRD goals: accurate transcriptions with visible disagreement, a searchable inventory, an offline OCR-only path, and a tight rule for what may leave the machine.

_Source: [PRD.md](../PRD.md) §1.1._


## Consequences

Agents and reviewers judge fit against these three missions before inventing scope, treating them as the direction of the repo rather than a description of what already ships. The same text is added as PRD §1.1 so the constitution's `_Source:` line stays accurate. The original PRD purpose / overview paragraph is left in place. This PR also aligns the README and AGENTS.md taglines with the missions and links them to constitution §1, and adds feature OCRMD-007 to the feature catalog, traced to requirement MR1. Merging this PR ratifies amendment 0002. Until then the record stays `Status: proposed`.
