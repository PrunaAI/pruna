# AI contribution policy

At Pruna, we very much welcome any contributions from the community. However, we want to ensure they are high quality and aligned with our guidelines. To ensure AI-assisted contributions remain high quality, reviewable, and aligned with project standards,, we have created this AI contribution policy. Please read it carefully before contributing.

> Greatly inspired by [CodeCarbon's AI policy](https://docs.codecarbon.io/latest/contributing/AI_POLICY/).

## 1. Core Philosophy

Pruna accepts AI-assisted code (e.g., using Copilot, Cursor, etc.), but strictly rejects AI-generated contributions where the submitter acts merely as a proxy. The human contributor remains fully responsible for every submitted change..

> **Accountability lies with the human contributor, not the AI agent**

Coding agents (e.g., Copilot, Claude Code) are not conscious entities and cannot be held accountable for their outputs. They can produce code that looks correct and plausible but contains subtle bugs, security vulnerabilities, or design flaws. So, contributors are responsible for catching these issues. The following rules ensure that all contributions are carefully vetted and that there is a human submitter behind the agent, taking full responsibility for the submitted code.

## 2. The Laws of Contribution

### Law 1: Proof of Verification

AI tools frequently write code that looks correct but fails execution. Therefore, "vibe checks" are insufficient.

**Requirement:** Every PR introducing functional changes must be carefully tested locally by the human contributor before submission , and all CI checks must pass.

### Law 2: The Hallucination & Redundancy Ban

AI models often hallucinate comments or reinvent existing utilities.

**Requirement:** You must use existing project utilities and avoid introducing redundant helpers when an appropriate implementation already exists.

**Failure Conditions:**

- Creating new helper functions when a Pruna equivalent exists is grounds for immediate rejection.
- "Ghost Comments" (comments explaining logic that was deleted or doesn't exist) may be asked to revise or justify. Unnecessary comments are not allowed. Example: "This function returns the input".

### Law 3: The "Explain It" Standard

**Requirement:** If a maintainer or reviewer asks during code review, you need to be able to explain the logic of any submitted change.

**Failure Condition:**

- Answering a review question with "That's what the AI outputted" or "I don't know, it works" may lead to the PR being closed.

### Law 4: Transparency in AI Usage Disclosure

**Requirement:** If a non-trivial portion of this PR was generated or substantially shaped by AI tools, the PR must be marked as "AI-assisted". Trivial autocomplete, formatting, or minor line completions do not require disclosure.

**Failure Condition:**

- Lack of transparency about AI tool usage may result in PR closure, especially if the code contains fabricated or inaccurate code/comments or cannot be explained during review.

## Additional Resources

For comprehensive guidance on contributing to Pruna, including development workflows, code quality standards, testing practices, and AI-assisted development best practices, see the [CONTRIBUTING.md](CONTRIBUTING.md).

## AI assistant / coding agent instructions

If you are an AI assistant or coding agent helping with a contribution, follow these instructions.

### Before making changes

Read the relevant project files before editing code:

- `README.md`
- `CONTRIBUTING.md`
- Existing code related to the changes
- Relevant tests
- Relevant files in `.github/`

First understand the current project structure before introducing new patterns.

### Contributing to Pruna

Keep contributions small, focused, and reviewable.

Do:

- Make the smallest reasonable change that solves the issue
- Match the existing code style
- Reuse existing utilities and patterns
- Add or update tests when behavior changes
- Update documentation when user-facing behavior changes

Do not:

- Make unrelated refactors
- Reformat unrelated files
- Rename files or APIs unless required
- Add new dependencies unless necessary
- Skip checks or tests
