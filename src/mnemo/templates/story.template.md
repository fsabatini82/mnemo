---
id: {{id}}
title: {{title}}
kind: story
lifecycle: proposed
epic:
priority: medium
related_files: []
related_adrs: []
related_bugs: []
key_symbols: []
---

# {{id}} — {{title}}

## User Story

Come **<ruolo>**, voglio **<azione>** così da **<beneficio>**.

## Acceptance Criteria

- AC1: ...
- AC2: ...

## Test Scenarios

### Happy path

```gherkin
Given <precondizione>
When <azione>
Then <risultato atteso>
```

### Error path

```gherkin
Given <precondizione che induce errore>
When <azione>
Then <comportamento di errore atteso>
```

### Edge case

```gherkin
Given <boundary condition>
When <azione>
Then <comportamento atteso al limite>
```

## Acceptance Summary

<1-3 frasi distillate: cosa significa "fatto" per questa story. Usato
da drift detection e retrieval contestuale, deve riflettere lo stato
canonico del comportamento atteso.>
