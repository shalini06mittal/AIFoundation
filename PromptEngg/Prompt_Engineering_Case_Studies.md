# Prompt Engineering: Complete Hands-On Guide

> A comprehensive guide covering foundational to advanced prompt engineering techniques, illustrated through practical exercises and real-world business case studies.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Techniques at a Glance](#techniques-at-a-glance)
3. [Basic Exercises](#basic-exercises)
   - [Exercise 1: Improve a Vague Prompt](#exercise-1-improve-a-vague-prompt)
   - [Exercise 2: Role-Based Prompting](#exercise-2-role-based-prompting)
   - [Exercise 3: Few-Shot Prompting](#exercise-3-few-shot-prompting)
   - [Exercise 4: Chain-of-Thought Prompting](#exercise-4-chain-of-thought-prompting)
   - [Exercise 5: Output Formatting](#exercise-5-output-formatting)
4. [Intermediate Exercises](#intermediate-exercises)
   - [Exercise 6: Hallucination Prevention](#exercise-6-hallucination-prevention)
   - [Exercise 7: Prompt Debugging](#exercise-7-prompt-debugging)
   - [Exercise 8: Prompt Chaining](#exercise-8-prompt-chaining)
5. [Business Case Studies — Core Techniques](#business-case-studies--core-techniques)
   - [Case Study 1: The Direct Ask](#case-study-1-the-direct-ask)
   - [Case Study 2: Role Assignment](#case-study-2-role-assignment)
   - [Case Study 3: Step-by-Step Breakdown](#case-study-3-step-by-step-breakdown)
   - [Case Study 4: Providing Examples](#case-study-4-providing-examples)
   - [Case Study 5: Setting Constraints](#case-study-5-setting-constraints)
   - [Case Study 6: Iterative Refinement](#case-study-6-iterative-refinement)
   - [Case Study 7: Format Specification](#case-study-7-format-specification)
6. [Advanced Case Studies — Professional Techniques](#advanced-case-studies--professional-techniques)
   - [Advanced Case 1: Role Prompting — Coaching a New Sales Rep](#advanced-case-1-role-prompting--coaching-a-new-sales-rep)
   - [Advanced Case 2: Context Setting — Writing a Proposal Email](#advanced-case-2-context-setting--writing-a-proposal-email)
   - [Advanced Case 3: Specify the Output Format — Competitive Landscape Summary](#advanced-case-3-specify-the-output-format--competitive-landscape-summary)
   - [Advanced Case 4: Chain-of-Thought — Choosing the Right Sales Channel](#advanced-case-4-chain-of-thought--choosing-the-right-sales-channel)
   - [Advanced Case 5: Few-Shot Prompting — Social Media Posts at Scale](#advanced-case-5-few-shot-prompting--social-media-posts-at-scale)
   - [Advanced Case 6: Tone & Audience Targeting — Announcing an Incentive Programme](#advanced-case-6-tone--audience-targeting--announcing-an-incentive-programme)
   - [Advanced Case 7: Constraint Setting — Pharmaceutical Ad Copy](#advanced-case-7-constraint-setting--pharmaceutical-ad-copy)
   - [Advanced Case 8: Iterative Prompting — Crafting a Sales Email Sequence](#advanced-case-8-iterative-prompting--crafting-a-sales-email-sequence)
7. [Advanced Bonus Exercises](#advanced-bonus-exercises)
   - [Exercise 9: Resume Screening Case Study](#exercise-9-resume-screening-case-study)
   - [Exercise 10: Customer Support AI Case Study](#exercise-10-customer-support-ai-case-study)
   - [Exercise 11: AI Tutor Prompt](#exercise-11-ai-tutor-prompt)
   - [Exercise 12: Creative Prompt Engineering](#exercise-12-creative-prompt-engineering)
   - [Bonus: ReAct Prompting](#bonus-react-prompting)
   - [Bonus: Persona Switching](#bonus-persona-switching)
8. [Group Activity: Prompt Competition](#group-activity-prompt-competition)
9. [Quick Reference Guides](#quick-reference-guides)
10. [Final Reflection Questions](#final-reflection-questions)
11. [Solutions](#solutions)
    - [Solutions: Basic & Intermediate Exercises](#solutions-basic--intermediate-exercises)
    - [Solutions: Business Case Studies](#solutions-business-case-studies)
    - [Solutions: Advanced Case Studies — Practice Exercises](#solutions-advanced-case-studies--practice-exercises)

---

## Introduction

Prompt engineering is the practice of crafting inputs to AI models in ways that produce accurate, useful, and well-structured outputs. This guide progresses from foundational concepts to advanced professional techniques, combining hands-on exercises with realistic business scenarios.

**How to use this guide:**

For each case study or exercise: read the scenario, study the weak vs. strong prompt comparison, attempt the practice exercise yourself, then check the solutions at the end. The goal is to build instincts, not to memorise prompts.

---

## Techniques at a Glance

| # | Technique | One-Line Summary |
|---|-----------|-----------------|
| #1 | **Role Prompting** | Tell the AI who to be |
| #2 | **Context Setting** | Explain the situation fully |
| #3 | **Specify the Output Format** | Define exactly what you want back |
| #4 | **Step-by-Step Thinking (Chain-of-Thought)** | Ask the AI to reason through problems |
| #5 | **Use Examples (Few-Shot Prompting)** | Show, don't just tell |
| #6 | **Tone & Audience Targeting** | Tailor the voice and language |
| #7 | **Constraint Setting** | Set rules and boundaries |
| #8 | **Iterative Prompting** | Refine until perfect |

---

## Basic Exercises

### Exercise 1: Improve a Vague Prompt

**Objective:** Understand how specificity improves AI outputs.

**Bad Prompt**

> "Write about Artificial Intelligence."

**Task for Participants**

Improve the prompt by adding:
- Audience
- Tone
- Length
- Format
- Purpose

> *(Try writing your improved prompt before looking at the solution at the end.)*

---

### Exercise 2: Role-Based Prompting

**Objective:** Learn how assigning roles changes AI behavior.

**Task:** Generate an email declining a project deadline extension.

**Prompt A**

> "Write an email declining a deadline extension."

**Prompt B**

> "You are a professional project manager. Write a polite but firm email declining a client's request for a two-week project extension due to contractual timelines."

**Discussion Points**
- Tone difference
- Professionalism
- Clarity
- Context awareness

> *(Try writing your own role-based prompt variation before looking at the solution.)*

---

### Exercise 3: Few-Shot Prompting

**Objective:** Teach AI using examples.

**Task:** Classify customer reviews as Positive, Negative, or Neutral.

**Prompt**

```
Classify the sentiment.

Review: "The food was amazing and service was quick."
Sentiment: Positive

Review: "The product arrived late and damaged."
Sentiment: Negative

Review: "The movie was okay, nothing special."
Sentiment:
```

> *(What output do you expect? Write it down before checking the solution.)*

---

### Exercise 4: Chain-of-Thought Prompting

**Objective:** Improve reasoning ability.

**Basic Prompt**

> "A shopkeeper buys a pen for ₹20 and sells it for ₹25. What is the profit percentage?"

**Improved Prompt**

> "Solve step-by-step and explain how you calculated the profit percentage."

> *(Try the improved prompt and compare your output against the solution at the end.)*

---

### Exercise 5: Output Formatting

**Objective:** Control AI response structure.

**Task:** Generate interview questions.

**Weak Prompt**

> "Give interview questions for Java."

**Strong Prompt**

> "Generate 5 Java interview questions for freshers in table format with columns: Question | Difficulty Level | Expected Answer Summary"

> *(Write your own strongly-formatted prompt for a topic of your choice before checking the solution.)*

---

## Intermediate Exercises

### Exercise 6: Hallucination Prevention

**Objective:** Reduce incorrect AI-generated facts.

**Weak Prompt**

> "Tell me about future AI laws in India."

**Improved Prompt**

> "Based only on officially announced policies till 2026, summarize current AI governance discussions in India. Do not speculate."

> *(Rewrite the weak prompt for a topic you care about, adding clear constraints against speculation.)*

---

### Exercise 7: Prompt Debugging

**Scenario:** AI gives inconsistent outputs.

**Original Prompt**

> "Create social media content for my company."

**Problems with this prompt:**
- Too generic
- No audience defined
- No platform specified
- No tone provided

**Improved Prompt**

> "Create 3 LinkedIn posts for a software training company targeting fresh graduates. Tone should be motivational and professional. Include hashtags."

> *(Debug one of your own recent prompts using the same analysis before checking the solution.)*

---

### Exercise 8: Prompt Chaining

**Objective:** Break complex tasks into smaller prompts.

**Task:** Create a business presentation.

**Prompt Chain**

**Prompt 1:** "Generate key points for a presentation on AI in Healthcare."

**Prompt 2:** "Convert these key points into slide titles."

**Prompt 3:** "Generate speaker notes for each slide."

> *(Design your own 3-prompt chain for a complex task you need to complete.)*

---

## Business Case Studies — Core Techniques

### Case Study 1: The Direct Ask

**Technique:** Be specific and state exactly what you need.

The most common prompting mistake is being vague. Instead of hoping the AI reads your mind, tell it precisely what you want — format, length, audience, and purpose.

**The Scenario**

Role: Regional Sales Manager  
Problem: You need a follow-up email for a prospect who attended your product demo last week but hasn't responded.

| ❌ Weak Prompt | ✅ Strong Prompt |
|----------------|-----------------|
| "Write a follow-up email for a sales prospect." | "Write a follow-up email to a prospect named David Chen, VP of Operations at a mid-size logistics company. He attended our warehouse management software demo last Tuesday. He mentioned concerns about implementation time during the demo. The email should be professional but warm, under 150 words, and include a soft call-to-action to schedule a 15-minute call this week." |
| **Why it fails:** Zero details about recipient, context, tone, or purpose. | **Why it works:** Includes who, context, tone, word limit, and a clear desired outcome. |

**Practice Exercise**

You're a marketing coordinator who needs social media captions for a new sustainable water bottle launch. Write a prompt that would generate effective Instagram captions.

*Consider: audience, tone, length, and specific features to highlight.*

> *(Write your prompt before checking the solution.)*

---

### Case Study 2: Role Assignment

**Technique:** Tell the AI to adopt a specific perspective or expertise.

When you assign the AI a role, it draws on knowledge and communication patterns associated with that expertise, shaping both content and style.

**The Scenario**

Role: Marketing Director  
Problem: Preparing for a difficult budget meeting — you need to anticipate tough CFO questions about a proposed ad spend increase.

| ❌ Basic Prompt | ✅ Role-Based Prompt |
|----------------|---------------------|
| "What questions might someone ask about increasing marketing spend?" | "Act as a skeptical CFO who prioritizes ROI and cash flow management. I'm proposing a 30% increase in digital advertising budget for Q3. What hard questions would you ask me, and what data would you need to see before approving this spend?" |
| **Why it fails:** No role is assigned; the AI gives a generic article-style response. | **Why it works:** CFO role produces financially-focused questions from a cost-justification angle. |

**Practice Exercise**

You're launching a customer loyalty program and want to identify potential flaws before the executive presentation. Create a prompt using role assignment to stress-test your idea.

*Consider: A competitor? A dissatisfied customer? A retail operations manager?*

> *(Write your prompt before checking the solution.)*

---

### Case Study 3: Step-by-Step Breakdown

**Technique:** Ask the AI to work through a problem methodically.

For complex tasks, asking the AI to break down its thinking produces more thorough and accurate results — especially for analysis, planning, and decision-making.

**The Scenario**

Role: Sales Operations Analyst  
Problem: Your team's close rate dropped from 32% to 24% over the past quarter.

| ❌ Surface-Level Prompt | ✅ Step-by-Step Prompt |
|------------------------|----------------------|
| "Why might our sales close rate have dropped?" | "Our B2B software sales team's close rate dropped from 32% to 24% over the past quarter. Walk me through a systematic analysis: First, identify the major categories of factors that typically influence close rates. For each category, list specific potential causes. Then suggest what data I should examine to diagnose each possible cause. Finally, recommend three questions I should ask my sales reps to gather qualitative insights." |
| **Why it fails:** AI jumps to one obvious answer without reasoning. | **Why it works:** Creates a comprehensive investigative framework with actionable next steps. |

**Practice Exercise**

Your company's email newsletter open rate has declined for three consecutive months. Create a prompt that asks the AI to analyze this problem step-by-step.

> *(Write your prompt before checking the solution.)*

---

### Case Study 4: Providing Examples

**Technique:** Show the AI what "good" looks like by including samples.

When you need output in a specific style, format, or voice, providing examples dramatically improves results.

**The Scenario**

Role: Content Marketing Manager  
Problem: You need product descriptions that match your established brand voice, but new writers keep missing the tone.

| ❌ Vague Prompt | ✅ Example-Based Prompt |
|----------------|------------------------|
| "Write a product description for our new hiking backpack." | "Write a product description for our new 45L hiking backpack. Match the voice and style of these existing descriptions from our site: Example 1 (Rain jacket): 'Caught in a downpour? Bring it on. The TrailShield Jacket laughs at sudden storms...' Example 2 (Hiking boots): 'Your feet called. They said they're done with blisters and trail fatigue...' The new backpack features: ventilated back panel, integrated rain cover, multiple access points, and a lifetime warranty. Keep the same conversational, slightly cheeky tone with short punchy sentences." |
| **Why it fails:** "Write a description" gives the AI no style reference. | **Why it works:** Examples establish voice, tone, structure, and sentence patterns concretely. |

**Practice Exercise**

You need to create customer testimonial requests that match your company's friendly, non-pushy style. Write a prompt that includes examples of your preferred communication style.

> *(Write your prompt before checking the solution.)*

---

### Case Study 5: Setting Constraints

**Technique:** Define boundaries, limits, and requirements.

Constraints prevent the AI from going off-track — they are guardrails that keep output focused, appropriate, and usable.

**The Scenario**

Role: HR Communications Specialist  
Problem: Announce a policy change (remote work reduced from 3 days to 2 days per week) clearly and empathetically without being apologetic or inviting debate.

| ❌ Open-Ended Prompt | ✅ Constrained Prompt |
|---------------------|----------------------|
| "Help me write an announcement about reducing remote work days." | "Write an internal announcement about reducing our remote work policy from 3 days to 2 days per week, effective June 1st. Must include: clear statement of change and effective date; brief rationale focused on collaboration; what remains unchanged (flexible scheduling of the 2 days); contact person for questions (HR Director, Sarah Martinez). Must avoid: apologetic language; phrases that invite negotiation; negative framing ('Unfortunately...' or 'We regret...'). Format: Under 200 words, professional but human tone, three short paragraphs maximum." |
| **Why it fails:** No boundaries; AI might produce apologetic or debate-inviting language. | **Why it works:** Must-include and must-avoid lists guarantee complete, on-brand output. |

**Practice Exercise**

You're writing a response template for handling customer complaints about shipping delays. Create a prompt with clear constraints about what the response must and must not include.

> *(Write your prompt before checking the solution.)*

---

### Case Study 6: Iterative Refinement

**Technique:** Start broad, then narrow down through follow-up prompts.

You don't have to get everything right in one prompt. Treat it as a conversation where you refine, redirect, and build on previous responses.

**The Scenario**

Role: Event Marketing Manager  
Problem: Planning a customer appreciation event and need creative theme ideas.

**Iterative Approach**

**Prompt 1 (Explore):** "Give me 8 creative theme ideas for a customer appreciation event. Our company is a B2B accounting software provider. Attendees are finance professionals — CFOs, controllers, and accountants. Budget is moderate, venue is a rooftop space in Chicago, timing is September."

**Prompt 2 (Narrow):** "I like options 3 (Harvest/Autumn Elegance) and 6 (Future of Finance). Expand on both — give me the visual aesthetic, suggested activities, and how each connects to showing customer appreciation."

**Prompt 3 (Decide and Detail):** "Let's go with Future of Finance. Now help me develop: a tagline for the invitations; 3 interactive elements that tie into the theme; a brief welcome speech opening (2–3 sentences) that thanks customers and introduces the theme."

**Practice Exercise**

You need to develop a sales incentive program but aren't sure what structure would work best. Map out an iterative prompting sequence (3–4 prompts) that would help you explore options and then develop the chosen approach.

> *(Design your sequence before checking the solution.)*

---

### Case Study 7: Format Specification

**Technique:** Tell the AI exactly how to structure the output.

Specifying format — tables, bullet points, specific sections — makes output immediately usable and saves reformatting.

**The Scenario**

Role: Competitive Intelligence Analyst  
Problem: Present a competitor comparison to the sales team in a format they can quickly reference during calls.

| ❌ Format-Less Prompt | ✅ Format-Specified Prompt |
|----------------------|--------------------------|
| "Compare our product to CompetitorX and CompetitorY." | "Create a competitive comparison between our CRM (AcmeCRM) and two competitors (SalesForce and HubSpot) for our sales team to use during prospect calls. Format as a table with these columns: Feature/Aspect | AcmeCRM | SalesForce | HubSpot | Our Advantage (one-sentence talking point). Include rows for: Pricing model, Implementation time, Mobile app capabilities, Reporting/analytics, Customer support, Integrations, and Ease of use. After the table, include a 'Quick Objection Handlers' section with 3 bullet points for the most common competitive objections and suggested responses." |
| **Why it fails:** AI produces paragraphs of prose that can't be used directly. | **Why it works:** Table format, columns, rows, and follow-on sections all specified — output is ready to use. |

**Practice Exercise**

You need a quarterly marketing performance report that your CEO can review in under 5 minutes. Create a prompt that specifies both content and format to produce a scannable, executive-friendly document.

> *(Write your prompt before checking the solution.)*

---

## Advanced Case Studies — Professional Techniques

### Advanced Case 1: Role Prompting — Coaching a New Sales Rep

**Industry:** B2B SaaS Sales | **Role:** Sales Manager

**Background**

Priya manages a team of sales representatives at a software company. One of her new hires, Rajan, is struggling to handle objections when prospects say "Your price is too high." Priya wants AI-generated coaching scripts she can share with Rajan.

| ❌ Weak Prompt | ✅ Strong Prompt |
|----------------|-----------------|
| "How should a salesperson respond when a customer says the price is too high?" | "You are a senior B2B sales trainer with 15 years of experience coaching SaaS sales teams. A new sales rep on my team is struggling when prospects say 'Your price is too high.' Write a practical coaching script with 3 specific response techniques she can practice word-for-word, including how to acknowledge the objection before countering it." |
| **Why it fails:** No role assigned; answer will be generic bulleted tips. | **Why it works:** Role sets expertise level and domain; output request is precise. |

**Practice Exercise**

You are an HR manager preparing to tell a long-serving employee that their role is being restructured. Write a role prompt that assigns the AI the most useful expert identity, specifies the deliverable, and includes important constraints.

> *(Write your prompt before checking the solution.)*

---

### Advanced Case 2: Context Setting — Writing a Proposal Email

**Industry:** Marketing Agency | **Role:** Account Executive

**Background**

Arjun is an account executive at a digital marketing agency in Mumbai. He has just met with a potential client — a mid-sized premium home décor e-commerce brand whose current agency is failing to deliver ROI on paid social.

| ❌ Weak Prompt | ✅ Strong Prompt |
|----------------|-----------------|
| "Write a proposal email to a potential client for my marketing agency." | "Write a follow-up proposal email for me. Context: I'm an account executive at a digital marketing agency based in Mumbai. The recipient is the marketing director of a premium home décor e-commerce brand. In our meeting yesterday, she mentioned she's frustrated that her current agency isn't delivering ROI from Meta and Instagram ads. My agency specialises in performance marketing for lifestyle and home brands. The email should: reference our conversation, acknowledge her frustration, briefly explain our approach, and propose a 30-minute call to walk through a custom strategy. Keep it professional but warm, under 200 words." |
| **Why it fails:** Zero context produces a generic template with placeholders. | **Why it works:** Full situational context makes the output feel personally written. |

**Practice Exercise**

You work in HR at a 200-person manufacturing company introducing mandatory 2-hour cybersecurity training for all employees within 30 days. Many employees are not tech-savvy and may resist. Write an internal announcement email prompt loaded with the right context.

> *(Write your prompt before checking the solution.)*

---

### Advanced Case 3: Specify the Output Format — Competitive Landscape Summary

**Industry:** Product Marketing | **Role:** Product Marketing Manager

**Background**

Sonal is a product marketing manager preparing materials for a leadership meeting. Her VP needs a quick competitive landscape summary comparing her company's product to three key competitors across five dimensions. The VP has exactly 10 minutes to read before the meeting.

| ❌ Weak Prompt | ✅ Strong Prompt |
|----------------|-----------------|
| "Compare our product management software to Asana, Monday, and Trello." | "Create a competitive landscape comparison table for a leadership meeting. Compare our product management software against Asana, Monday.com, and Trello. The table must have exactly 5 rows: Price Range, Key Features, Best For (target customer), Main Weakness, and Why Customers Choose It. Format it as a clear table with one column per competitor and our product in the first column. Keep each cell to 1–2 concise sentences maximum. After the table, add a 3-bullet 'Key Takeaways' section." |
| **Why it fails:** No format specified; AI produces 600+ words of narrative. | **Why it works:** Table structure, rows, column order, cell length, and follow-on section all defined. |

**Practice Exercise**

You are preparing a weekly team meeting update on three active projects: a website redesign, a new product launch campaign, and a customer satisfaction survey. Each project needs its status, completions this week, upcoming items, and any blockers. Write a format-specific prompt.

> *(Write your prompt before checking the solution.)*

---

### Advanced Case 4: Chain-of-Thought — Choosing the Right Sales Channel

**Industry:** Consumer Goods | **Role:** Head of Sales Strategy

**Background**

Deepak heads sales strategy for a mid-sized company that makes premium kitchenware. The company has been selling exclusively D2C through its own website. Leadership is debating whether to expand to Amazon Marketplace, independent retail stores, or both.

| ❌ Weak Prompt | ✅ Strong Prompt |
|----------------|-----------------|
| "Should we sell our premium kitchenware on Amazon or in retail stores?" | "I need a reasoned recommendation on expanding our sales channels. We currently sell premium kitchenware exclusively D2C through our website. We're evaluating three options: Amazon Marketplace, independent retail stores, or both. Think through this step by step: (1) What are the advantages and risks of each channel for a premium brand? (2) How does each option affect brand perception and pricing control? (3) What are the operational demands of each? (4) Which option do you recommend for a premium brand that wants to grow revenue while protecting brand equity, and why? Present your reasoning before your recommendation." |
| **Why it fails:** Asks for a conclusion without reasoning; produces shallow justifications. | **Why it works:** Four explicit reasoning steps guide the AI; recommendation comes after structured analysis. |

**Practice Exercise**

You manage operations for a 50-person company. Your lease is up in 6 months and leadership is choosing between renewing the office lease, moving to co-working, or going fully remote. Write a chain-of-thought prompt asking the AI to reason through cost, employee wellbeing, collaboration, and talent attraction before giving a recommendation.

> *(Write your prompt before checking the solution.)*

---

### Advanced Case 5: Few-Shot Prompting — Social Media Posts at Scale

**Industry:** Marketing / E-commerce | **Role:** Social Media Manager

**Background**

Kavita manages social media for a fast-growing organic food brand with a very specific voice: enthusiastic, health-conscious, slightly witty, always ending with a call to action or question. She needs 5 Instagram captions for new product launches that match the existing brand voice exactly.

| ❌ Weak Prompt | ✅ Strong Prompt |
|----------------|-----------------|
| "Write 5 Instagram captions for our new range of organic snack bars. Keep them fun and healthy-sounding." | "Write 5 Instagram captions for our new range of organic snack bars. Match the tone and style of these existing brand captions exactly: Example 1: 'Your 3pm slump called. We told it to take a hike. 🌿 Our Spirulina Energy Bites are here to keep you powered — no crash, no compromise. What's your go-to afternoon pick-me-up?' Example 2: 'Real food. Real energy. Zero guilt. Our Cacao Almond Bars are what happens when a nutritionist and a chocolatier become best friends. Grab yours before they're gone. 🍫' Example 3: 'We read the label so you don't have to. (Spoiler: it's just 6 ingredients you can actually pronounce.) Tag a friend who deserves a clean snack upgrade! ✨' Now write 5 captions for our new Mango Coconut Snack Bar using the same voice, emoji style, ending question or CTA, and character of these examples." |
| **Why it fails:** 'Fun and healthy-sounding' is too vague; generic wellness copy results. | **Why it works:** Three concrete examples give the AI sentence rhythm, emoji placement, CTA patterns, and humor style. |

**Practice Exercise**

You work in the customer success team of a SaaS company. Your team sends handwritten-style thank-you emails to customers who renew their annual subscription. You have one great example email. Write a prompt that provides the example and asks the AI to write 3 more in the same voice, for different customer tenure scenarios.

> *(Write your prompt before checking the solution.)*

---

### Advanced Case 6: Tone & Audience Targeting — Announcing an Incentive Programme

**Industry:** Retail / HR | **Role:** HR Director

**Background**

Meena is HR Director at a large retail chain with 3,000 employees across 80 stores. She needs to announce a new performance-based bonus scheme to two very different audiences on the same day: store-level sales associates and regional managers.

| ❌ Weak Prompt | ✅ Strong Prompt |
|----------------|-----------------|
| "Announce our new performance bonus scheme to all employees." | "Write two versions of an internal announcement about a new performance bonus scheme. Version 1 — for store sales associates: Use simple, energetic, motivating language. Avoid jargon. Focus on 'what's in it for me' — how much can they earn and how do they qualify? Use short sentences, a conversational tone, and one or two relatable examples. Under 200 words. Version 2 — for regional managers: Use professional language. Cover the scheme structure, how to explain it to their teams, implementation timeline, FAQ they may receive from staff, and their role in tracking performance. This version can be up to 400 words and use subheadings." |
| **Why it fails:** One generic version lands in the middle — too complex for associates, too simple for managers. | **Why it works:** Two audiences defined with different vocabulary levels, emotional goals, structural complexity, and length. |

**Practice Exercise**

Your company is switching from email threads to a new project management platform. Write a single prompt asking for two separate communications — one for the IT team who will set it up, and one for non-technical employees who will use it daily.

> *(Write your prompt before checking the solution.)*

---

### Advanced Case 7: Constraint Setting — Pharmaceutical Ad Copy

**Industry:** Pharmaceutical / Healthcare Marketing | **Role:** Marketing Copywriter

**Background**

Rahul writes marketing copy for a consumer health brand. He needs a product description for a new vitamin supplement. The brand's compliance team has strict rules: no comparative claims, no disease-cure claims, no superlatives, and word count must be under 80 words.

| ❌ Weak Prompt | ✅ Strong Prompt |
|----------------|-----------------|
| "Write a product description for our new Vitamin D3 + K2 supplement for a digital ad." | "Write a product description for our new Vitamin D3 + K2 supplement for a digital advertisement. Constraints you must follow: (1) Maximum 80 words — do not exceed this. (2) Do not use superlatives like 'best', 'most effective', 'number one', or similar. (3) Do not make any disease-treatment or cure claims — focus on general wellness and lifestyle support only. (4) Do not reference or compare to any other brands or products. (5) Tone must be warm, trustworthy, and grounded — not salesy or hyped. The description should explain what D3 and K2 do together and why someone would want to take them daily." |
| **Why it fails:** AI naturally reaches for superlatives and cure-adjacent claims. | **Why it works:** Five numbered constraints are unambiguous and cannot be accidentally missed. |

**Practice Exercise**

Your publicly listed company's CEO wants to post a LinkedIn update about strong quarterly results. Legal constraints: no specific financial figures before the earnings call; no forward-looking statements; must acknowledge the team, not just the CEO; must be under 150 words. Write the constrained prompt.

> *(Write your prompt before checking the solution.)*

---

### Advanced Case 8: Iterative Prompting — Crafting a Sales Email Sequence

**Industry:** Professional Services / Sales | **Role:** Business Development Manager

**Background**

Sunita is a BDM at a management consulting firm. She has used AI to generate a 3-email cold outreach sequence. The first draft is too generic, the subject lines are weak, and the second email doesn't acknowledge the prospect hasn't responded to the first.

| ❌ Weak Follow-Up | ✅ Strong Follow-Up |
|------------------|---------------------|
| "Can you make it better?" | "Please revise the email sequence with these specific changes: (1) Email 1 — The subject line is too generic. Replace it with something that references a specific supply chain pain point, such as inventory delays or supplier lead times. Keep the subject under 8 words. (2) Email 1 — The opening line starts with 'I' — rewrite it to open with a statement about the prospect's industry challenge instead. (3) Email 2 — This email doesn't acknowledge that the prospect hasn't replied. Add a brief, non-pushy acknowledgement that this is a follow-up, and make it feel warmer — not like a chase email. (4) Email 3 — The call to action is vague ('let me know if you're interested'). Replace it with a specific, low-commitment CTA like proposing a 20-minute call on two specific days. Keep all other elements the same." |
| **Why it fails:** 'Make it better' gives the AI no direction; it may change things you liked. | **Why it works:** Four numbered, specific pieces of feedback — each says what's wrong and what to replace it with. |

**Practice Exercise**

You asked an AI to write a job posting for a Senior Account Manager role. The first draft has 12 bullet points in the responsibilities section (too long), generic corporate language, no salary range included, and a tone that doesn't reflect your startup culture. Write a specific follow-up iterative prompt.

> *(Write your prompt before checking the solution.)*

---

## Advanced Bonus Exercises

### Exercise 9: Resume Screening Case Study

**Scenario:** HR team wants to shortlist candidates.

**Prompt**

> "You are an HR recruiter. Analyze the resume below for a Java Developer role. Evaluate:
> - Technical skills
> - Relevant experience
> - Communication indicators
> - Final recommendation out of 10"

**Extension**

Ask AI to:
- Compare multiple resumes
- Generate interview questions
- Identify skill gaps

---

### Exercise 10: Customer Support AI Case Study

**Scenario:** A telecom company receives repetitive complaints.

**Participant Task:** Create prompts for angry customer handling, refund explanation, and network issue troubleshooting.

**Example Prompt**

> "You are a calm and empathetic telecom support executive. Respond to an angry customer complaining about poor internet connectivity. Keep the response under 120 words and offer troubleshooting steps."

---

### Exercise 11: AI Tutor Prompt

**Objective:** Create personalized learning experiences.

**Prompt**

> "You are a Python trainer teaching absolute beginners. Explain loops using real-life examples and include 3 practice exercises with solutions."

---

### Exercise 12: Creative Prompt Engineering

**Task:** Generate marketing slogans.

**Weak Prompt**

> "Give slogans for coffee."

**Strong Prompt**

> "Generate 10 modern and catchy slogans for a premium organic coffee brand targeting Gen Z customers."

---

### Bonus: ReAct Prompting

Ask AI to reason, search, act, and verify.

**Example**

> "Think step-by-step, identify missing information, then provide the final answer."

---

### Bonus: Persona Switching

Ask AI to respond as:
- A CEO
- A teacher
- A lawyer
- A stand-up comedian

Observe differences in tone, vocabulary, and structure.

---

## Group Activity: Prompt Competition

**Challenge:** Each team receives a vague prompt such as: *"Explain blockchain."*

**Goal:** Transform it into the best engineered prompt.

**Evaluation Criteria:**
- Clarity
- Context
- Constraints
- Creativity
- Output quality

---

## Quick Reference Guides

### Choosing Your Technique

| Situation | Best Technique |
|-----------|----------------|
| You know exactly what you need | Direct Ask |
| You need a specific perspective or expertise | Role Assignment |
| The problem is complex or multi-layered | Step-by-Step Breakdown |
| You need output to match an existing style | Providing Examples |
| The output must meet specific requirements | Setting Constraints |
| You're exploring options or unsure what you want | Iterative Refinement |
| You need output in a specific structure | Format Specification |

### Prompting Techniques Cheat Sheet

| Technique | Prompt Template | Quick Example |
|-----------|----------------|---------------|
| **Role Prompting** | *You are a [role] with [experience]. Help me...* | You are a senior sales trainer... |
| **Context Setting** | *Background: [situation]. The audience is [who]. My goal is [what].* | We are a 50-person firm. Our lease expires in... |
| **Format Specification** | *Format as [table/bullets/script]. Include [sections]. Max [words].* | Format as a 5-row comparison table. Each cell max 2 sentences. |
| **Chain-of-Thought** | *Think step by step: (1)... (2)... (3)... Then give your recommendation.* | Analyse advantages, then risks, then recommend. |
| **Few-Shot Examples** | *Here are [N] examples of the style I want: [examples]. Now write...* | Here are 3 captions. Write 5 more in the same voice. |
| **Tone & Audience** | *Audience: [who they are]. Tone: [specific qualities]. Avoid: [what not to do].* | Audience: non-tech staff. Tone: warm and simple. No jargon. |
| **Constraint Setting** | *You must not: (1)... (2)... (3)... You must: (1)... (2)...* | No figures before earnings call. Max 150 words. |
| **Iterative Prompting** | *Please revise with these changes: (1)... (2)... Keep everything else.* | Change subject line. Fix opening. Make CTA specific. |

### The Golden Rules of Prompting

| Rule | Explanation |
|------|-------------|
| **Be Specific, Not Vague** | The more precise your prompt, the more precise the output. Vague prompts produce generic results. |
| **Specify the Format** | Always say what you want back — a table, a list, a script, an email. Don't leave it to chance. |
| **Include Context** | The AI knows nothing about your company, your audience, or your goals unless you tell it. |
| **Assign a Role** | An expert identity improves almost every prompt. Who is the best person to answer this question? |
| **Set Constraints** | Tell the AI what NOT to do as much as what TO do. Rules prevent wasted output. |
| **Use Examples** | One good example is worth a paragraph of description. Show the AI what great looks like. |
| **Iterate, Don't Restart** | If the first answer is close but not right, give specific feedback. Don't start from scratch. |
| **Read What You Get** | AI output needs review. It's a first draft, not a final answer. Check facts, tone, and fit. |

---

## Final Reflection Questions

- What makes a prompt effective?
- How does context improve output quality?
- When should examples be used?
- How can hallucinations be reduced?
- Which prompting technique worked best for your task?

---

## Solutions

> **Note:** There are many valid approaches to each exercise. The solutions below are examples of effective prompts — not the only right answers.

---

### Solutions: Basic & Intermediate Exercises

---

**Exercise 1 — Improve a Vague Prompt**

**Sample Improved Prompt:**

> "Write a 200-word beginner-friendly explanation of Artificial Intelligence for college students. Include 3 real-life examples and explain the difference between AI and Machine Learning in simple language."

**What makes it work:** Adds audience (college students), length (200 words), format (explanation + examples), and purpose (beginner understanding). The output will be better structured, more focused, and more readable.

**Key learning:** Participants observe better structure, more focused output, and improved readability.

---

**Exercise 2 — Role-Based Prompting**

**Solution Insight:** Role prompting improves tone alignment, domain-specific language, and audience targeting.

Prompt A produces a generic email. Prompt B — with the role of "professional project manager" assigned — produces language that reflects contractual awareness, professional tone, and domain vocabulary.

**Key learning:** Assigning a role shapes both content and delivery style.

---

**Exercise 3 — Few-Shot Prompting**

**Expected Output:**

```
Neutral
```

**Key learning:** Examples guide AI behavior and improve consistency. The pattern established by the two examples (Positive and Negative) teaches the AI the classification format, and the third review ("okay, nothing special") maps logically to Neutral.

---

**Exercise 4 — Chain-of-Thought Prompting**

**Expected Step-by-Step Solution:**

```
Profit = 25 - 20 = ₹5
Profit Percentage = (5 / 20) × 100
                 = 25%

Answer: 25%
```

**Key learning:** Step-by-step prompting improves logical reasoning accuracy. The improved prompt forces the AI to show its work, making it easier to catch errors.

---

**Exercise 5 — Output Formatting**

The strong prompt produces a table like:

| Question | Difficulty Level | Expected Answer Summary |
|----------|-----------------|------------------------|
| What is the difference between == and .equals() in Java? | Beginner | == checks reference equality; .equals() checks value equality |
| ... | ... | ... |

**Key learning:** Formatting constraints improve usability. The output is immediately presentation-ready.

---

**Exercise 6 — Hallucination Prevention**

**Key learning:** Constraining AI to official, announced policies and explicitly prohibiting speculation dramatically reduces hallucinated facts. Adding "Do not speculate" or "Based only on [source/date]" is a powerful guard.

---

**Exercise 7 — Prompt Debugging**

**Identified Problems:**
- Too generic ("my company" gives the AI nothing)
- No audience specified
- No platform named
- No tone direction

**Improved Prompt:**

> "Create 3 LinkedIn posts for a software training company targeting fresh graduates. Tone should be motivational and professional. Include hashtags."

**Key learning:** Debug prompts by systematically identifying what information is missing.

---

**Exercise 8 — Prompt Chaining**

**Sample Chain:**

**Prompt 1:** "Generate key points for a presentation on AI in Healthcare."

**Prompt 2:** "Convert these key points into slide titles."

**Prompt 3:** "Generate speaker notes for each slide."

**Key learning:** Complex workflows become manageable through chaining. Each prompt builds on the previous output, allowing you to guide the AI step by step.

---

### Solutions: Business Case Studies

---

**Case Study 1 Solution — Direct Ask (Instagram Captions)**

**Sample Prompt:**

> "Write 5 Instagram captions for our new sustainable water bottle launch.
>
> Product details: 24oz insulated stainless steel bottle, keeps drinks cold 24 hours / hot 12 hours, made from 90% recycled materials, comes in 6 colors, $35 price point.
>
> Audience: Environmentally-conscious millennials and Gen Z, fitness-oriented, urban professionals.
>
> Tone: Upbeat, slightly playful, eco-proud without being preachy.
>
> Requirements: Each caption under 150 characters (for full display); include 1 call-to-action caption, 1 feature-focused caption, 1 lifestyle caption, 1 sustainability-focused caption, and 1 playful/humorous caption. Suggest 3–5 relevant hashtags for each."

**What makes it effective:** Specifies product details, audience, tone, variety of angles, practical constraints (character count), and additional useful elements (hashtags).

---

**Case Study 2 Solution — Role Assignment (Loyalty Program Critique)**

**Sample Prompt:**

> "Act as three different perspectives to stress-test our new customer loyalty program:
>
> Perspective 1 — Skeptical Operations Manager: Focus on implementation complexity, staff training burden, and point-of-sale integration challenges.
>
> Perspective 2 — Bargain-Hunting Customer: Evaluate whether the rewards are actually compelling compared to competitor programs, and identify ways customers might game the system.
>
> Perspective 3 — CFO Concerned About Costs: Question the revenue impact, breakage assumptions, and liability we're creating with outstanding points.
>
> For each perspective, give me your top 3 concerns and the questions you'd ask before approving this program."

**What makes it effective:** Multiple roles provide diverse angles of criticism; specific focus areas for each role; actionable output (concerns and questions).

---

**Case Study 3 Solution — Step-by-Step (Email Open Rate Decline)**

**Sample Prompt:**

> "Our company newsletter open rate has dropped from 28% to 19% over the past three months. Analyze this systematically:
>
> - Categorize the factors: What are the main categories of things that affect email open rates? (e.g., deliverability, content, timing, list quality)
> - Diagnose within each category: For each category, what specific changes or problems might explain a decline?
> - Identify what to investigate: What data should I pull from our email platform to check each potential cause?
> - Consider external factors: What industry-wide changes (email client updates, privacy features, seasonal patterns) might be contributing?
> - Recommend quick tests: Suggest 3 A/B tests I could run next month to isolate the problem."

**What makes it effective:** Clear numbered structure; moves from diagnosis to investigation to action; considers both internal and external factors; ends with testable next steps.

---

**Case Study 4 Solution — Providing Examples (Testimonial Requests)**

**Sample Prompt:**

> "Write an email requesting a customer testimonial. Match the voice of these examples from our customer communications:
>
> Example 1 (Support follow-up): 'Hey Jordan — just checking in after last week's setup session. Everything running smoothly? If you hit any snags, reply to this email and I'll jump in. We're rooting for your success.'
>
> Example 2 (Feature announcement): 'Good news: the dashboard upgrade you've been asking about just went live. No learning curve required — it's right where you expect it, just faster and prettier. Take it for a spin and let us know what you think.'
>
> The ask: Request a brief testimonial (2–3 sentences) we can feature on our website. Make it easy to say yes — offer to draft something based on their feedback that they can approve. Keep it under 100 words, warm and human, not corporate. The customer's name is Priya, and she's been using our project management tool for 8 months."

**What makes it effective:** Clear examples establish brand voice; explains the specific ask; offers a friction-reducing approach (we'll draft, they approve); includes personalization details.

---

**Case Study 5 Solution — Setting Constraints (Shipping Delay Response)**

**Sample Prompt:**

> "Create a customer service response template for shipping delay complaints.
>
> Must include:
> - Acknowledgment of the frustration (without over-apologizing)
> - Clear statement that we're looking into their specific order
> - Realistic timeline for update (24–48 hours)
> - Direct contact info for follow-up (not a general support queue)
> - Offer of [X]% discount code for their next order
>
> Must avoid:
> - Blaming carriers, warehouses, or external factors
> - Phrases like 'unprecedented volume' or 'supply chain issues'
> - Promises we can't guarantee ('it will definitely arrive by...')
> - Defensive language or asking the customer to check tracking themselves
> - More than one apology
>
> Format: Under 100 words, professional but empathetic, include [PLACEHOLDER] brackets for order number and discount code, structure as 3 short paragraphs."

**What makes it effective:** Comprehensive inclusion/exclusion lists address specific pain points; practical format requirements with placeholders; word limit keeps it concise.

---

**Case Study 6 Solution — Iterative Refinement (Sales Incentive Program)**

**Sample Prompting Sequence:**

**Prompt 1 (Explore options):** "What are 6 different structures for a sales incentive program? Our team is 12 inside sales reps selling B2B software. Mix of individual and team-based options, both monetary and non-monetary rewards. Just give me a one-paragraph overview of each approach."

**Prompt 2 (Narrow and compare):** "Let's compare options 2 (tiered commission accelerator) and 5 (team competition with experiential prizes). For each, explain: how the mechanics work, what behaviors it incentivizes, potential downsides or gaming risks, and what type of sales culture it works best for."

**Prompt 3 (Develop chosen approach):** "Let's develop the tiered commission accelerator. Design a specific structure where: base quota is $50K/month, there are 3 performance tiers, and top performers can earn up to 2x their base commission rate. Show me the tier breakpoints, commission multipliers at each tier, and a sample calculation for a rep who closes $75K in a month."

**Prompt 4 (Finalize details):** "Now help me communicate this program. Create: a one-page program summary for the sales team, an FAQ section with 5 likely questions, and a tracking spreadsheet structure reps can use to monitor their tier progress."

**What makes it effective:** Starts broad for discovery; uses comparison to make an informed choice; builds increasing detail; ends with practical implementation materials.

---

**Case Study 7 Solution — Format Specification (Executive Marketing Report)**

**Sample Prompt:**

> "Create a Q2 marketing performance report for CEO review. Must be scannable in under 5 minutes.
>
> Format structure:
> - Executive Summary (3 bullet points max): Most important headline, biggest win, one area needing attention
> - Key Metrics Dashboard (table format): Metric | Q2 Result | Q1 Result | Target | Status (✓ or ✗). Include: Website traffic, Lead generation, Marketing qualified leads, Cost per lead, Email subscribers, Social engagement
> - Channel Performance (2 sentences per channel): Paid digital, Organic/SEO, Email, Social media
> - Wins & Challenges (two columns side by side): Left: Top 3 wins | Right: Top 3 challenges
> - Next Quarter Focus (3 numbered priorities only)
>
> Use bold for any metric that beat target by more than 10%. Use [PLACEHOLDER] for specific numbers I'll insert. Total length: one page maximum."

**What makes it effective:** Explicit section structure; table format for metrics; visual elements (checkmarks, bold); clear length constraints; placeholder notation for real data insertion.

---

### Solutions: Advanced Case Studies — Practice Exercises

---

**Advanced Case 1 — Role Prompting (HR Difficult Conversation)**

**Sample Solution:**

> "You are an experienced HR business partner who specialises in change management and difficult employee conversations. I need to inform a 12-year employee that their role is being restructured — their current responsibilities will change significantly, though they are not being let go. Write me a step-by-step conversation framework I can follow in our 30-minute meeting, including how to open the conversation, deliver the news clearly, handle emotional reactions, and close on a constructive note."

**Key principles:** Role specificity (change management expertise); context embedded (12-year tenure, not a termination); output defined (conversation framework with specific sections); constraint included (30-minute meeting).

---

**Advanced Case 2 — Context Setting (Cybersecurity Training Announcement)**

**Sample Solution:**

> "Write an internal company-wide email announcing a new mandatory cybersecurity training. Context: The company is a 200-person manufacturing firm. Many employees work on the floor and are not very tech-savvy. Some may be resistant to online training. The training is 2 hours, fully online, and must be completed within 30 days. The tone should be friendly and encouraging — not authoritative or threatening. The email should explain why cybersecurity matters in simple, non-technical language, give clear instructions on how to access the training, address likely concerns (e.g., 'When do I do this?' and 'Is this difficult?'), and mention that managers will be available to help. Keep it under 300 words."

**Key principles:** Audience description shapes tone; specifics (2 hours, 30-day deadline, online) appear naturally in a good response; tone direction prevents a corporate-sounding draft; anticipated objections prompt the AI to address them proactively.

---

**Advanced Case 3 — Format Specification (Weekly Project Update)**

**Sample Solution:**

> "Create a weekly project status update document. Format it as three separate sections, one per project: (1) Website Redesign, (2) New Product Launch Campaign, (3) Customer Satisfaction Survey. For each project, use four labelled sub-points: Status (one word: On Track / At Risk / Delayed), Completed This Week (2–3 bullet points), Coming Up Next (2–3 bullet points), and Blockers (one sentence or 'None'). Keep the whole document under one page. Use a clean, professional style suitable for sharing with a senior manager."

**Key principles:** Format named explicitly; status options defined for consistent language; length bounded; use case stated to calibrate formality.

---

**Advanced Case 4 — Chain-of-Thought (Office Lease Decision)**

**Sample Solution:**

> "I need a structured analysis to inform a workplace decision. Our 50-person company's office lease expires in 6 months. We are considering: (a) renewing the current office lease, (b) moving to a smaller co-working space, or (c) going fully remote. Please think through this in steps: (1) Analyse the cost implications of each option (assume current lease is mid-range for our city). (2) Consider the impact on employee wellbeing and day-to-day collaboration for a team that works across 3 departments. (3) Assess how each option affects our ability to attract talent. (4) Identify the key risks of each option. After working through all four points, give your recommendation with a clear rationale. Present the analysis first, then the recommendation."

**Key principles:** Numbered reasoning steps; decision criteria explicit; sequence enforced (analysis before recommendation); assumptions stated to allow reasoning without real data.

---

**Advanced Case 5 — Few-Shot Prompting (SaaS Renewal Thank-You Emails)**

**Sample Solution:**

> "Write 3 customer renewal thank-you emails for our SaaS platform. Match the voice, warmth, and structure of this example exactly: 'Hi Meera, we just saw your renewal come through and honestly — it made our day. You've been with us through two major product updates and we're so glad the platform has grown with you. We'd love to hear how things are going. Reply anytime — we're always here. Warmly, the [Company] team.'
>
> Preserve: the conversational warmth, the personal 'we just saw' framing, a specific reference to something real about the customer's journey, the open invitation to reply, and the sign-off style.
>
> Write three versions for: (1) a customer renewing for the first time after their first year, (2) a customer renewing for the 5th consecutive year, and (3) a customer who recently reached out with a complaint but chose to renew anyway. Use different names: Rohan, Aisha, and Vikram."

**Key principles:** Example given verbatim; named qualities to preserve; variation defined; specific recipients and scenarios prevent identical emails.

---

**Advanced Case 6 — Tone & Audience Targeting (Project Management Platform Switch)**

**Sample Solution:**

> "Write two internal communications announcing the switch from email-based project coordination to a new project management platform (Asana).
>
> Communication 1 — for the IT/Technical Team: Professional and direct. Include: the implementation timeline, their responsibilities for setup and user provisioning, integration points with existing tools, and the escalation process for technical issues. Assume technical literacy — use appropriate platform terminology. Under 300 words with bullet points for tasks.
>
> Communication 2 — for general non-technical staff: Warm, reassuring, and simple. Many will feel nervous about change. Focus on: why this makes their work easier (not why IT needs it), what will change day-to-day, that training will be provided, and who to contact with questions. Use plain language, short sentences, and avoid technical terms. Under 200 words.
>
> Both should come from 'The Operations Team'."

**Key principles:** Audience emotional state considered; information hierarchy set; vocabulary calibrated; sender consistent for organisational coherence.

---

**Advanced Case 7 — Constraint Setting (CEO LinkedIn Post)**

**Sample Solution:**

> "Write a LinkedIn post from our CEO celebrating a strong quarter for the company. The post must follow these strict constraints: (1) No specific financial figures, revenue numbers, or percentages — results can be described positively but not quantified. (2) No forward-looking statements or predictions about future quarters or growth. (3) The post must acknowledge the full team's contribution prominently — this is not a personal achievement post. (4) Maximum 150 words — do not exceed this. (5) Professional but human tone — not corporate-speak. The post should convey genuine pride in the team's work, signal confidence in the company's direction without making any projections, and end with a note of appreciation for employees and stakeholders."

**Key principles:** Numbered constraints (discrete and impossible to skip); positive and negative framing; hard word limit as firm instruction; tone specified both ways.

---

**Advanced Case 8 — Iterative Prompting (Job Posting Revision)**

**Sample Solution:**

> "Please revise the job posting with the following specific changes:
>
> (1) Responsibilities section — cut from 12 bullets to a maximum of 6. Keep only the most essential and distinctive responsibilities. Remove anything generic (e.g., 'communicate effectively with stakeholders').
>
> (2) Language and tone — remove all corporate clichés like 'dynamic team player', 'fast-paced environment', 'go-getter', and 'results-driven'. Replace with direct, honest language that reflects how we actually work. Our culture is informal, direct, and collaborative — the posting should sound like a real person wrote it.
>
> (3) Salary — add a salary range of ₹18–24 LPA in the compensation section. If there is no compensation section, add one after the requirements.
>
> (4) Culture section — add a short 2–3 sentence section describing our startup culture: small team, flat structure, lots of autonomy, and fast decision-making. Place it before the requirements.
>
> Keep all other sections unchanged."

**Key principles:** Numbered feedback (each issue isolated); what to remove AND what to replace with; specific examples quoted ('dynamic team player'); explicit instruction to preserve what was already good.

---

*Reference: [Prompting Guide — Basics](https://www.promptingguide.ai/introduction/basics)*
