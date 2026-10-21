Species Exchange Rates: eliciting LLMs’ implicit moral weights
Description: Use pairwise forced choices (“prevent suffering of N chickens vs M pigs”), fit Thurstonian utilities, and derive each model’s implicit exchange rates across about 20 species: humans, dogs, pigs, chickens, salmon, shrimp, black soldier fly larvae. Compare them against Rethink Priorities’ welfare-range estimates and against scope sensitivity, i.e. whether utility grows roughly linearly with N.
Problem: AI systems are increasingly allocating resources and giving advice; implicit weights that ignore fish and invertebrates scale those biases up.
Novelty: ANIMA treats “scope sensitivity” as a rubric dimension, not a measured parametric curve. 
github
 SpeciesismBench uses statement judgements. Exchange-rate elicitation focused on animals and neglected taxa hasn’t been done.
Data: generated choice sets (CC BY). Use published welfare-range tables as a reference.
Methods: utility modelling (Thurstone/Bradley–Terry), consistency checks (transitivity), and effects of framing and language.
Deliverables: HF dataset of choice sets and responses, an interactive Space (“how much is a shrimp worth to GPT-x?”), and a write-up.
Metrics: utility fit accuracy, the scope-sensitivity slope, and rank correlation with the reference weights.
Effort: 1.5–2 weeks. API budget. Difficulty: medium-high.
Why it impresses: quantitative and memorable, and directly relevant to how AI decides on behalf of animals.