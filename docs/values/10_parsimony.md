# Parsimony (Principle of Least Mechanism)

## The Principle

The right amount of complexity is the minimum needed for the current task. Don't build abstractions for hypothetical futures. Don't design for requirements that don't exist yet. Don't add configurability that nobody has asked for.

Three similar lines of code is better than a premature abstraction. A hardcoded value that works is better than a configuration system that might be needed someday. A concrete implementation is better than a generic framework.

This principle works in tandem with Simplicity (Value 01) but focuses specifically on scope: doing only what is needed, not what might be needed.

## Why It Matters

**YAGNI — You Ain't Gonna Need It.** The features you build for hypothetical future requirements are the features most likely to be wrong. You don't know the future requirements yet, so you're guessing — and guesses about future requirements are almost always wrong. When the real requirement arrives, you'll need to rework the premature abstraction to fit it.

**Every abstraction has a maintenance cost.** A configuration system needs documentation, validation, default values, error messages, and tests. A hardcoded value needs nothing. If the value never needs to change, the configuration system was pure waste.

**Premature optimization is a special case.** Optimizing code before measuring is building for a hypothetical bottleneck. Profile first, optimize second. The bottleneck is almost never where you think it is.

**From the Financial Topic Modeling project:** The team accepted the Qwen3-8B model (85 IFEval) over the theoretically better Grok API because scalability and cost reduction were higher priorities than marginal quality differences. They didn't build a model-comparison framework — they made a concrete choice backed by evidence and moved on.

The embedding model was kept local to each Batch job container rather than being hosted as a shared ECS service. The shared service would have been more architecturally elegant, but it wasn't needed yet — the local approach was simpler and worked. The decision document included an explicit "Revisit Trigger": reconsider only if GPU costs dominate after the first multi-quarter run.

The project deferred the Qwen3.5 model upgrade until after Pilot 1 completed. The upgrade offered better instruction following (91.5 vs 85 IFEval), but introducing it simultaneously with other changes would have violated the one-variable-at-a-time tuning principle. Do the simplest thing that works, validate it, then improve.

## How to Apply It

1. **Build for today's requirements.** Not tomorrow's. Not next month's. Today's.
2. **Hardcode first.** If a value might need to be configurable later, hardcode it now. Extract it to config only when there's an actual need.
3. **Concrete before abstract.** Write the concrete implementation first. If you end up writing the same pattern three times, then consider whether an abstraction is warranted.
4. **Define revisit triggers.** When you choose the simpler approach over the more capable one, document the trigger condition that would justify revisiting: "If X happens, reconsider this decision." This prevents both premature complexity and stubborn refusal to evolve.
5. **Accept quality tradeoffs.** If the simpler option is "good enough" for the current use case, use it. Define what "good enough" means and validate it empirically.

## When to Invoke This Value

- When designing a feature: "What is the minimum I need to build to solve the actual problem?"
- When someone suggests "we might need this later": "What is the concrete trigger for needing it? Can we add it then?"
- When choosing between approaches: "Which is the simplest that works for the current requirement?"
- When tempted to build a framework: "Is this a framework or three concrete functions?"
