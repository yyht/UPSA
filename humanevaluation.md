# Human evaluation 
Suggested by the reviewer, we’ve now conducted double-blind human evaluation following[Li+EMNLP’18]. The results are consistent with automatic metric. Here we elaborate our experiment with more details.

We randomly select 500 sentences from the test data as input and generate paraphrases using different models (including UPSA, CGMH, and VAE). The pairs of paraphrases are then aggregated and partitioned into three random buckets for three human assessors to evaluate. The assessors are asked to rate each sentence pair according to the following two criteria: relevance (the paraphrase sentence is semantically
close to the reference sentence) and fluency (the paraphrase sentence is fluent as a natural language sentence, and the grammar is correct). Hence each assessor gives two scores to each paraphrase, both ranging from 1 to 5. Each paraphrase is rated by two assessors, and then averaged as the final judgement.

# Results
| Model |Relevance|Fluency|  
|---|---|---|
| VAE   |  2.70 ± 0.18  | 3.23 ± 0.12  |  
| CGMH  |  3.02 ± 0.13   | 3.51 ± 0.08   |  
| UPSA  |  3.64 ± 0.15   | 3.76 ± 0.08   |  

The above results show that our UPSA achieves significant better performance than VAE and CGMH in terms of both sentence relevace and fluency ($P<10^{-100}$ by two-sided Wilcoxon signedrank test), further demonstrating the superiority of our method.
