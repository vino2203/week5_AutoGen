# Dialogue flow

```mermaid
sequenceDiagram
    participant User
    participant Manager as Manager (AI, tools)
    participant Agents as Framework agents (AI)
    participant Rec as Recommender (AI)
    User->>Manager: Description (+ optional fields)
    Manager->>Manager: set_project_brief (assumptions listed)
    par one step, in parallel
        Manager->>Agents: analyze_framework(rag)
        Manager->>Agents: analyze_framework(n8n)
        Manager->>Agents: analyze_framework(crewai)
        Manager->>Agents: analyze_framework(autogen)
    end
    Agents-->>Manager: one-line result each (full reports are stored)
    Manager->>Rec: recommend_stack
    Rec-->>User: Recommendation, comparison, design, plan
    User->>Rec: approve / reject / new limits (re-score asks the recommender only)
```

Only the options the user chose are analyzed. The manager decides the order and what to call; the tools refuse
calls that break the rules and the code reminds the manager once if it stops early.
