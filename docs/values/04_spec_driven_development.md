# Spec-Driven Development

## The Principle

Specifications are contracts. Write the spec, get it approved, then build to spec.

The spec defines: what the component does (overview), what it depends on (dependencies), how it is called (public interface with full method signatures, type hints, and docstrings), how it handles errors (exception classes), and how to verify it works (testing strategy and validation criteria).

Deviations from the spec require explicit user approval. If something is ambiguous in the spec, ASK — don't assume.

## Why It Matters

**Specs prevent scope creep.** Without a spec, implementation tends to drift. "While I'm here, I'll also add..." leads to bloated, poorly focused code. The spec draws a boundary: this is what we're building, and nothing more.

**Specs enable parallel work.** When the spec is written, the reviewer can evaluate the design without waiting for implementation. The test writer can write tests against the spec's method signatures. The documentation writer can write user-facing docs. The spec is the handshake between all parties.

**Specs are reviewable.** Code is hard to review in the abstract — you need to run it, trace its logic, and understand its context. A spec is a plain-English description of what the code will do. It is much easier to catch design mistakes in a spec than in a completed implementation.

**From the Financial Topic Modeling project:** The `wrds_connector_spec.md` defined the complete WRDSConnector interface — constructor signature, all public and private methods with full Args/Returns/Raises docstrings, SQL query templates, data structures, exception classes, and validation criteria — before a single line of implementation was written. This spec was reviewed by both the user and the Codex reviewer. When Sprint 2 began, the implementing instance had an unambiguous contract to build against.

The sentiment-ready schema spec (`sentiment_ready_schema_spec.md`) defined the exact PyArrow schema that the pipeline's output must conform to. This meant the downstream sentiment analysis team could begin writing their consumer code against the spec while the pipeline was still being built.

## How to Apply It

1. **Before implementation**: Write an ADR for architectural decisions. Write an interface spec for each component.
2. **Spec structure**: Overview, Module Location, Dependencies, Class Definition, Constructor, Public Methods (full signatures), Private Methods, Data Structures, Exception Classes, Configuration, Testing Strategy, Validation Criteria.
3. **Get approval**: The spec is reviewed before implementation begins. The Codex reviewer and/or user approves the spec.
4. **Build to spec**: Implementation follows the spec exactly. Method signatures, return types, error handling — all as specified.
5. **Change management**: If during implementation you discover the spec needs to change, update the spec and get re-approval. Don't silently deviate.

## When to Invoke This Value

- Before writing any implementation code: "Is there a spec for this?"
- When the implementation diverges from the spec: "Stop. Update the spec first."
- When planning a sprint: "What specs need to be written?"
- When reviewing completed work: "Does this match the spec?"
