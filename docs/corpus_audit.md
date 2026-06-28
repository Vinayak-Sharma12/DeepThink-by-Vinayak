# LOGOS Corpus Audit

Generated: 2026-06-28T12:15:09.582840+00:00

## Summary

- Train rows: 6568
- Holdout rows: 341
- Train tokens (est.): 3,147,732

## Metadata coverage (train)

- `stance`: 100.0%
- `tradition`: 100.0%
- `topic`: 100.0%
- `source_type`: 100.0%
- `license`: 100.0%

## Topic × tradition heatmap

| Topic | Tradition | Count |
|-------|-----------|-------|
| authority | critique_of_religion | 326 |
| compassion | buddhist | 251 |
| critique_of_eternalism | buddhist | 1 |
| critique_of_religion | atheism | 139 |
| critique_of_religion | critique_of_religion | 604 |
| critique_of_religion | hindu | 11 |
| deontology | deontology | 57 |
| design_argument | atheism | 220 |
| design_argument | christian | 229 |
| design_argument | skepticism | 97 |
| devotion | hindu | 377 |
| devotion | sikh | 537 |
| divine_command | christian | 75 |
| divine_command | platonism | 17 |
| emptiness | buddhist | 221 |
| epistemology | atheism | 165 |
| epistemology | jain | 5 |
| epistemology | moral_psychology | 33 |
| epistemology | skepticism | 128 |
| ethics | buddhist | 81 |
| evolution | atheism | 220 |
| existence_of_god | atheism | 679 |
| existence_of_god | buddhist | 222 |
| existence_of_god | christian | 1553 |
| existence_of_god | critique_of_religion | 130 |
| existence_of_god | hindu | 332 |
| existence_of_god | islamic | 305 |
| existence_of_god | jewish | 7 |
| existence_of_god | sikh | 537 |
| existence_of_god | skepticism | 97 |
| faith_vs_evidence | atheism | 987 |
| faith_vs_evidence | critique_of_religion | 456 |
| faith_vs_evidence | hindu | 83 |
| faith_vs_evidence | islamic | 230 |
| faith_vs_evidence | skepticism | 225 |
| fallacy | logic | 53 |
| free_will | christian | 80 |
| logic | atheism | 136 |
| logic | buddhist | 222 |
| logic | logic | 53 |
| meaning | atheism | 34 |
| meaning | buddhist | 82 |
| meaning | critique_of_religion | 148 |
| meaning | hindu | 27 |
| meditation | buddhist | 81 |
| mind | buddhist | 23 |
| miracles | skepticism | 128 |
| morality | buddhist | 355 |
| morality | christian | 1169 |
| morality | deontology | 57 |
| morality | hindu | 377 |
| morality | islamic | 296 |
| morality | secular_humanism | 46 |
| morality | sikh | 537 |
| morality | virtue_ethics | 200 |
| morality_without_god | atheism | 479 |
| morality_without_god | buddhist | 82 |
| morality_without_god | islamic | 9 |
| morality_without_god | platonism | 17 |
| naturalism | naturalism | 244 |
| nihilism | atheism | 34 |
| no_self | buddhist | 647 |
| no_self | hindu | 11 |
| problem_of_evil | atheism | 49 |
| problem_of_evil | christian | 80 |
| problem_of_evil | islamic | 9 |
| rebirth | jain | 5 |
| religion_and_politics | atheism | 139 |
| religion_and_science | atheism | 431 |
| religion_and_science | naturalism | 244 |
| religion_and_violence | atheism | 177 |
| religion_as_phenomenon | naturalism | 244 |
| revelation | christian | 1169 |
| revelation | islamic | 296 |
| revelation_vs_reason | christian | 75 |
| revelation_vs_reason | islamic | 221 |
| revelation_vs_reason | jewish | 7 |
| soul | buddhist | 1 |
| soul | christian | 1169 |
| soul | hindu | 332 |
| soul | islamic | 296 |
| soul | jain | 5 |
| soul | sikh | 537 |
| suffering | buddhist | 669 |
| survey | atheism | 254 |
| survey | buddhist | 93 |
| utilitarianism | secular_humanism | 46 |
| virtue_ethics | virtue_ethics | 200 |

## Mix buckets (train tokens)

- `scripture`: 1,144,827 tokens (36.4%) — target 22%
- `theist_philosophy`: 306,540 tokens (9.7%) — target 18%
- `atheist_secular`: 866,224 tokens (27.5%) — target 18%
- `buddhism`: 307,308 tokens (9.8%) — target 14%
- `ethics_logic_psych`: 137,841 tokens (4.4%) — target 12%
- `indian_multi`: 59,174 tokens (1.9%) — target 8%
- `other`: 325,818 tokens (10.4%) — target 8%

**Mix note:** Corpus ≥500k tokens but mix buckets outside ±5%: `scripture` at 36.4% (target 22%); `theist_philosophy` at 9.7% (target 18%); `atheist_secular` at 27.5% (target 18%); `ethics_logic_psych` at 4.4% (target 12%); `indian_multi` at 1.9% (target 8%). Scripture-heavy skew expected until licensed canon is rebalanced.

## Gaps flagged

- Missing (pending owner file): C.S. Lewis — Mere Christianity
- Missing (pending owner file): Debiprasad Chattopadhyaya — Charvaka/Lokāyata
- Missing (pending owner file): Ambedkar — Annihilation of Caste
- Missing (pending owner file): Ambedkar — The Buddha and His Dhamma

## Pillar checklist

- Atheist/secular: **OK**
- Four scriptures: **OK**
- Buddhism (3 schools): **OK**
- Theist philosophy: **OK**
- Ethics/logic/psych: **OK**
- Multi-tradition: **OK**
- Debates: **OK**
