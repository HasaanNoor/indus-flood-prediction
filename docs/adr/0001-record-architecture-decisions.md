# ADR-0001: Use Architecture Decision Records

- Status: Accepted
- Date: 2026-08-04
- Decision owners: Project maintainers
- Related phases: Phase 15.1
- Related files: [README.md](../../README.md), [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md), [docs/research/spatial-model-training.md](../research/spatial-model-training.md), [docs/research/multi-event-sentinel-labels.md](../research/multi-event-sentinel-labels.md), [docs/research/multi-event-processing.md](../research/multi-event-processing.md)

## Context

The repository has accumulated decisions across temporal forecasting, spatial grid construction, Sentinel-1 label ingestion, mapping safeguards, and model interpretation. The research notes explain phase-specific reasoning, but there was no stable index of decisions that reviewers or future contributors could scan.

## Decision

Use Architecture Decision Records under `docs/adr/` for decisions that materially shape architecture, data semantics, modeling methodology, validation, or interpretation. ADRs are sequentially numbered, dated, statused, and linked from an ADR index and the main README.

## Alternatives Considered

- Keep decisions only in phase research notes. This preserves chronology but makes durable decisions harder to find.
- Put decision summaries only in the README. This would overload the README and duplicate research detail.
- Use issue tracker discussions. That would not travel with the repository snapshot.

## Consequences

### Positive

- Reviewers can see why important tradeoffs were accepted.
- Future phases have a place to record changes without rewriting history.
- Technical interview discussion can point to concise, evidence-backed decisions.

### Negative

- Documentation must be maintained when decisions change.
- ADRs can become misleading if new work supersedes them without status updates.

### Risks

- Recording too many trivial details would reduce signal.
- Treating ADRs as promotional summaries would obscure real limitations.

## Validation

This phase adds an ADR index and records only decisions already reflected in code, tests, README content, or research notes.

## Revisit Conditions

Revisit this process if ADRs become stale, if the project adopts another formal decision log, or if a major redesign requires decision records to be reorganized.

## References

- [README.md](../../README.md)
- [docs/research/spatial-feature-grid.md](../research/spatial-feature-grid.md)
- [docs/research/spatial-model-training.md](../research/spatial-model-training.md)
- [docs/research/multi-event-sentinel-labels.md](../research/multi-event-sentinel-labels.md)
- [docs/research/multi-event-processing.md](../research/multi-event-processing.md)

