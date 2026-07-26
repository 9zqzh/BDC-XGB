---
name: "research-paper-writing"
description: "Improve academic paper writing quality for ML/CV/NLP-style papers with clear section structure, paragraph flow, and reviewer-facing presentation. Use when drafting or revising Abstract, Introduction, Related Work, Method, Experiments, or Conclusion; polishing figures/tables; checking claim-support alignment; or performing self-review before submission."
---

---
name: research-paper-writing
description: Improve academic paper writing quality for ML/CV/NLP-style papers with clear section structure, paragraph flow, and reviewer-facing presentation. Use when drafting or revising Abstract, Introduction, Related Work, Method, Experiments, or Conclusion; polishing figures/tables; checking claim-support alignment; or performing self-review before submission.
---
# Research Paper Writing

## Overview

Use this skill to rewrite a research paper into a reviewer-friendly, high-clarity draft.
Prioritize first-impression quality (figures/tables/layout), logical flow, and evidence-backed claims.

## Core Workflow

1. Clarify the paper story before sentence-level edits.
2. Use section-specific guidance in `references/`.
3. Rewrite paragraph-by-paragraph with one message per paragraph.
4. Run reverse outlining after writing each section.
5. Check every major claim in Abstract/Introduction against experimental evidence.
6. Run final-paper adversarial review with `references/paper-review.md`.

## Global Principles

1. Keep one paragraph for one message only.
2. State the paragraph message in the first sentence.
3. Make nouns self-contained; define new terms before reusing them.
4. Maintain sentence-to-sentence flow (cause, contrast, consequence, or refinement).
5. Iterate with adversarial self-review: read as a skeptical reviewer.
6. Treat visual quality as core content, not decoration.
7. Use a clean teaser and pipeline figure.
8. Use readable, minimal-ink tables.
9. Keep formatting consistent and tidy.

## Paragraph Clarity Check (Important)

Use this quick test whenever the user asks whether a paragraph "flows" or is clear.

1. Read as an external reader:
   - Does this paragraph have one explicit message?
   - Does the first sentence state what this paragraph will do?
   - Are all key nouns/terms readable without hidden context?
   - Does each sentence connect to the previous one with a clear relation (cause, contrast, consequence, refinement, example)?
2. Run reverse outlining for the current section:
   - Write down thesis/main claim.
   - Write down each paragraph topic sentence.
   - Write down the evidence/explanation points under each paragraph.
   - Check mapping: topic sentence -> thesis, and evidence -> topic sentence.
   - Revise or remove any paragraph that cannot be mapped cleanly.
3. If flow is still weak, add temporary section headers and explicit transition phrases during revision, then remove unnecessary headers before finalizing.

Source reference for this check:

- `references/does-my-writing-flow-source.md`

## Section Guides

Load only the needed section file:

- Introduction: `references/introduction.md`
- Abstract: `references/abstract.md`
- Related Work: `references/related-work.md`
- Method: `references/method.md`
- Experiments: `references/experiments.md`
- Conclusion: `references/conclusion.md`
- Paper review (Paper Rview): `references/paper-review.md`
- Paragraph clarity source: `references/does-my-writing-flow-source.md`
- Example bank index: `references/examples/index.md`

## Paper Review Core Points

Use `references/paper-review.md` for the full checklist and workflow.

1. Add an end-of-draft self-review question list in five dimensions:
   - contribution,
   - writing clarity,
   - experimental strength,
   - evaluation completeness,
   - method design soundness.
2. Treat claim-evidence alignment as a hard constraint, especially for Abstract and Introduction.
3. Perform adversarial writing: review as a skeptical reviewer and resolve every high-risk question.
4. Revise until major rejection risks are explicitly addressed.

## Execution Rules

1. Build a mini-outline before drafting prose.
2. For each subsection, explicitly include motivation, design, and technical advantage when applicable.
3. Avoid writing style that looks like incremental patching of a naive baseline.
4. Keep terminology stable across the full paper.
5. If a claim cannot be supported by results, weaken or remove the claim.
6. Before finalizing, append and answer a five-dimension self-review question list, then revise the paper based on unresolved items.
7. Do not load all section references (Introduction/Abstract/Related Work/Method/Experiments/Conclusion) at once; load only the specific section guide needed for the current edit target.

## Output Contract

When asked to rewrite or draft sections, return:

1. A compact section outline (3-7 bullets).
2. Revised paragraphs with explicit paragraph roles (opening/challenge/method/advantage/evidence/limitation).
3. A short self-review checklist covering clarity, flow, terminology consistency, unsupported claims, and missing evidence.
4. A claim-evidence map for each major claim in the revised text using `Claim: ... | Evidence: ... | Status: supported/needs evidence`.

## Reference: Paragraph Flow Source

### Summary
Good academic writing flows when each sentence follows naturally from the previous one. The reader should never wonder "why is this sentence here?" or "how does this connect?"

### Core Principles

1. **One message per paragraph.** Each paragraph should communicate exactly one idea. If a paragraph contains two distinct messages, split it.

2. **Topic sentence first.** The first sentence of each paragraph should state what the paragraph will argue or explain. Readers use topic sentences to decide whether to read the paragraph.

3. **Known-to-new contract.** Each sentence should begin with something the reader already knows (from earlier in the paragraph or paper), then introduce new information at the end. This creates a chain: the new information at the end of sentence N becomes the known information at the start of sentence N+1.

4. **Explicit connectors.** Use explicit transition words and phrases when the logical relationship between sentences might be unclear: cause (therefore, as a result), contrast (however, in contrast), consequence (consequently, thus), refinement (specifically, in particular), example (for instance, to illustrate).

5. **Parallel structure.** When presenting a list of related points, use parallel grammatical structure. This signals to the reader that these items belong to the same category.

### Reverse Outlining

Reverse outlining is the most reliable test of paragraph flow:

1. Write down the thesis or main claim of the entire section.
2. For each paragraph, write down only its topic sentence (first sentence).
3. Read just the topic sentences in order. They should form a logical argument chain that supports the thesis.
4. Under each topic sentence, list the evidence or explanation points.
5. Check that each evidence point actually supports its paragraph's topic sentence.
6. If a paragraph cannot be cleanly mapped, revise or remove it.

### Common Problems and Fixes

- **Buried topic sentence:** The paragraph's main point appears in sentence 3 or 4. Fix: move it to the first sentence.
- **Missing connector:** Two sentences sit next to each other with no clear relationship. Fix: add an explicit transition word or phrase.
- **Known-new violation:** A sentence starts with brand-new information. Fix: reorder so the sentence begins with something established.
- **Multiple messages:** A paragraph tries to do too many things. Fix: split into separate paragraphs, each with its own topic sentence.

## Reference: Paper Review Checklist

### Five-Dimension Self-Review

Before submitting, append this question list to your draft and answer each question:

#### 1. Contribution
- What is the single most important claim this paper makes?
- Would a reviewer from a related but different subfield understand the contribution?
- Is the contribution incremental or significant? If incremental, is the delta clearly justified?

#### 2. Writing Clarity
- Does the abstract contain all key results (numbers)?
- Can a reader understand Figure 1 and Table 1 without reading the paper?
- Are all acronyms defined on first use?
- Are all symbols in equations defined?
- Is terminology consistent throughout?

#### 3. Experimental Strength
- Does every claim in the abstract and introduction have corresponding experimental evidence?
- Are ablation studies complete (each proposed component tested)?
- Are error bars or statistical tests reported?
- Are the strongest baselines included?

#### 4. Evaluation Completeness
- Are multiple datasets used?
- Are multiple metrics reported?
- Are failure cases analyzed?
- Are computational costs compared?

#### 5. Method Design Soundness
- Is each design choice motivated?
- Are there simpler alternatives that could achieve similar results?
- Are limitations honestly stated?

### Claim-Evidence Map Template

For each major claim, fill in:
- Claim: [the specific assertion]
- Evidence: [table/figure/result that supports it]
- Status: supported / weak / needs evidence

### Adversarial Review Workflow

1. Read the paper as if you are a skeptical reviewer who wants to find reasons to reject.
2. Write down every question or objection a reviewer might raise.
3. For each high-risk question, either add evidence, weaken the claim, or add a limitation note.
4. Repeat until no high-risk questions remain unanswered.

## Reference: Introduction Writing

### Structure

The introduction should follow a funnel structure: broad context → specific problem → gap in existing work → proposed solution → contributions.

### Opening Strategies

Choose one of these patterns based on your paper type:

1. **Task-then-application:** Define the task, then explain why it matters for real applications.
2. **Application-first:** Start with a compelling application, then introduce the task needed to enable it.
3. **General-to-specific setting:** Start broad, progressively narrow to your specific setting.
4. **Open with challenge:** Start by stating a major unsolved challenge, then position your work as addressing it.

### Contribution Paragraph

The final paragraph of the introduction should explicitly list contributions. Use bullet points or numbered items. Each contribution should be a concrete, verifiable statement, not a vague aspiration.

### Common Pitfalls

- Starting too broad ("In recent years, deep learning has achieved great success...")
- Claiming contributions not supported by experiments
- Missing explicit comparison to closest prior work
- Failing to state the problem before proposing the solution

## Reference: Abstract Writing

### Structure

A strong abstract follows this sequence: context (1 sentence), problem (1 sentence), insight/approach (1-2 sentences), key results (2-3 sentences with numbers), significance (1 sentence).

### Key Rules

- Always include quantitative results.
- No citations (usually).
- No undefined acronyms.
- Self-contained: a reader should understand the contribution from the abstract alone.

## Reference: Method Writing

### Module Description Triad

For each module or component, follow this pattern:
1. **Motivation:** Why is this module needed? What problem does it solve?
2. **Design:** How does it work? What is the key mechanism?
3. **Technical advantage:** Why is this design better than alternatives?

### Overview Figure

The method section should open with an overview figure showing the full pipeline. The figure should be referenced in the text before detailed module descriptions begin.

## Reference: Experiments Writing

### Structure

1. **Setup:** Datasets, metrics, baselines, implementation details.
2. **Main results:** Comparison to baselines on primary metrics.
3. **Ablation studies:** Contribution of each component.
4. **Analysis:** Qualitative results, failure cases, efficiency comparisons.

### Table Design

- Use minimal-ink tables: remove unnecessary gridlines.
- Bold the best result in each column.
- Align numbers on decimal points.
- Include the metric name and direction (↑ for higher-is-better, ↓ for lower-is-better) in column headers.

## Reference: Related Work

### Structure

Organize related work by topic, not chronologically. For each topic area:
1. Summarize the key idea of existing work (2-3 sentences).
2. Explain the limitation or gap (1-2 sentences).
3. State how your work differs or improves (1 sentence).

### Important Rules

- Never just list papers. Group and synthesize.
- Be fair and accurate about prior work. Reviewers who wrote those papers will read your related work section.
- End with a paragraph that summarizes how your work collectively differs from all prior work.

## Reference: Conclusion

### Structure

1. Restate the problem and main contribution (1-2 sentences).
2. Summarize key findings (2-3 sentences).
3. State limitations (1-2 sentences).
4. Suggest future directions (1-2 sentences).

### Important Rules

- Do not introduce new results or claims not already supported.
- Do not overclaim significance.
- Be honest about limitations.
