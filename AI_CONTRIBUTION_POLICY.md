# AI contribution policy

At Pruna, we very much welcome any contributions from the community. However, we want to ensure they are high quality and aligned with our guidelines. To prevent unwanted behavior by the community, we have created this AI contribution policy. Please read it carefully before contributing.

> Greatly inspired by [CodeCarbon's AI policy](https://docs.codecarbon.io/latest/contributing/AI_POLICY/).

## 1. Core Philosophy

Pruna accepts AI-assisted code (e.g., using Copilot, Cursor, etc.), but strictly rejects AI-generated contributions where the submitter acts merely as a proxy. The submitter is the **Sole Responsible Author** for every line of code, comment, and design decision.

> **Accountability lies with the human contributor, not the AI agent**

Coding agents (e.g., Copilot, Claude Code) are not conscious entities and cannot be held accountable for their outputs. They can produce code that looks correct and plausible but contains subtle bugs, security vulnerabilities, or design flaws. So, maintainers and reviewers are ultimately responsible for catching these issues. The following rules ensure that all contributions are carefully vetted and that there is a human submitter behind the agent, taking full responsibility for the submitted code.

## 2. The Laws of Contribution

### Law 1: Proof of Verification

AI tools frequently write code that looks correct but fails execution. Therefore, "vibe checks" are insufficient.

**Requirement:** Every PR introducing functional changes must be carefully tested locally by the human contributor before submission.

### Law 2: The Hallucination & Redundancy Ban

AI models often hallucinate comments or reinvent existing utilities.

**Requirement:** You must use existing methods and libraries, and never reinvent the wheel.

**Failure Conditions:**

- Creating new helper functions when a Pruna equivalent exists is grounds for immediate rejection.
- "Ghost Comments" (comments explaining logic that was deleted or doesn't exist) will result in a request for a full manual rewrite. Unnecessary comments are not allowed. Example: "This function returns the input".

### Law 3: The "Explain It" Standard

**Requirement:** If a maintainer or reviewer asks during code review, you must be able to explain the logic of any function you submit.

**Failure Condition:**

- Answering a review question with "That's what the AI outputted" or "I don't know, it works" leads to immediate closure.

### Law 4: Transparency in AI Usage Disclosure

**Requirement:** If you used AI tools for coding, but manually reviewed and tested every line following the guidelines, you must mark the PR as "AI-assisted".

**Failure Condition:**

- Lack of transparency about AI tool usage may result in PR closure, especially if the code contains hallucinations or cannot be explained during review.

## 3. Cases where Human must stay in control

In some cases, such as boilerplate code outside the logic of the product, we could accept AI-generated code reviewed by another AI agent.

But for the core logic of the product, we want to ensure that humans fully understand the code and the design decisions. This is to ensure that the code is maintainable, secure, and aligned with the project's goals.

## Additional Resources

For comprehensive guidance on contributing to Pruna, including development workflows, code quality standards, testing practices, and AI-assisted development best practices, see the [CONTRIBUTING.md](CONTRIBUTING.md).