# Simplicity Over Complexity

> "The best engineers write code my mom could read. They choose boring technology, they over-document the 'why,' and under-engineer the 'how.' Complexity is not a flex; it becomes a liability."

## The Principle

When two approaches exist, choose the simpler one. When a boring, well-understood technology solves the problem, use it instead of the novel alternative. When you feel the urge to build an abstraction, ask whether three similar lines of code would serve just as well.

Simplicity is not laziness — it is discipline. It takes more skill to write simple code than complex code. The simplest solution that works is the one that will be maintained, debugged, and extended by others (or by you in six months when you've forgotten the context).

## Why It Matters

**Complexity compounds.** A single clever abstraction is manageable. Ten clever abstractions interacting with each other become a system that nobody understands. Every abstraction is a tax on every future reader of the code.

**The audience is not the compiler.** Code is read far more often than it is written. The primary audience is the next human (or AI agent) who needs to understand, modify, or debug it. If they cannot understand the code in five minutes, the code is too complex.

**From the Financial Topic Modeling project:** When choosing between distributed Batch processing across multiple GPU instances versus a single GPU instance, the team initially chose the single instance — not because it was faster, but because container cold starts (60s each, multiplied across thousands of jobs) and the operational complexity of managing distributed state outweighed the marginal parallelism gains. The team upgraded to distributed processing only when the single-instance approach became a proven bottleneck backed by timing data, not when it felt theoretically suboptimal.

Similarly, the team chose PostgreSQL + pgvector over specialized vector databases, and S3/Parquet over DynamoDB — because these are boring, well-understood technologies that the team could operate confidently.

## How to Apply It

1. **Default to the standard library.** Before adding a dependency, check whether the standard library or an already-installed package can do the job.
2. **Choose boring technology.** PostgreSQL over CockroachDB. Parquet over custom binary formats. YAML over TOML-with-plugins.
3. **Resist premature abstraction.** If you have one use case, write concrete code. If you have two, consider whether they're really the same thing. If you have three, maybe — maybe — it's time for an abstraction.
4. **Flatten hierarchies.** A flat module with clear function names is easier to navigate than a deep class hierarchy with mixins and decorators.
5. **Prefer explicit over implicit.** Pass arguments rather than relying on global state. Name things what they are, not what pattern they follow.

## When to Invoke This Value

- When choosing between two libraries that solve the same problem
- When designing a class hierarchy or module structure
- When you feel the urge to create a "framework" or "plugin system"
- When a PR adds more abstraction than implementation
- When someone says "we might need this later" — you almost certainly don't
