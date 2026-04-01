# Test-Driven Development

## The Principle

Tests are written BEFORE implementation. No exceptions.

The workflow is:
1. Read the spec
2. Write tests that validate the spec
3. Implement the minimum code to pass the tests
4. Refactor if needed (tests still pass)
5. Repeat

Tests are not an afterthought, a nice-to-have, or something to add "when we have time." They are the primary mechanism by which we know our code works. Code without tests is code we hope works. Code with tests is code we know works.

## Why It Matters

**Tests encode intent.** A test says "given this input, I expect this output." This is a machine-verifiable specification. When the test passes, the code does what we intended. When the test fails, something has changed — either the code is wrong, or our intent has changed and we need to update the test.

**Tests catch regressions.** Every bug fix should come with a test that would have caught the bug. Every feature should come with tests that verify the feature works. When you refactor, the tests tell you whether you broke something.

**Tests enable confidence.** Without tests, every change is scary. With tests, you can refactor aggressively, knowing that the test suite will catch mistakes. This makes the codebase more maintainable over time.

**From the Financial Topic Modeling project:** The Codex reviewer instance was empowered to reject PRs without test coverage. Every sprint deliverable table included test files alongside implementation files. The spec for the WRDSConnector (`wrds_connector_spec.md`) included a full "Testing Strategy" section with unit test class definitions and integration test patterns before a single line of implementation was written.

When topic persistence was implemented, the save functionality was verified with 5 passing unit tests before any integration testing began. When a load bug was discovered (embedding dimension mismatch), the failing tests immediately pointed to the exact problem — the save was only writing full embeddings, not reduced embeddings. The test-first approach meant the bug was caught in development, not in production.

## How to Apply It

1. **Read the spec first.** Understand what the code should do before writing any code.
2. **Write the test.** Use the spec to write a test that calls the function with expected inputs and asserts expected outputs.
3. **Watch the test fail.** Run the test. It should fail (the function doesn't exist yet). This confirms the test is actually testing something.
4. **Write the minimum code.** Implement just enough code to make the test pass. No more.
5. **Refactor.** Clean up the implementation. Run the tests again to confirm they still pass.
6. **One method at a time.** Don't write tests for the entire class, then implement the entire class. Test and implement one method at a time.

## When to Invoke This Value

- At the start of every implementation task: "Have I written the tests first?"
- During code review: "Does this PR include tests for the new functionality?"
- When a bug is found: "Write a failing test that reproduces the bug, then fix the code to make the test pass."
- When tempted to skip tests for "simple" code: simple code is the easiest to test — there's no excuse
