Hi Jan,

I hope you are doing well! As we discussed on Wednesday, the metric we think can be used to compare the probability distributions of the firm's patent portfolios is the Bhattacharyya Coefficient (BC). It is closely related to a distance metric, the Bhattacharyya Distance. I have attached the original paper (Bhattacharyya, 1946 -- attached) as well as a follow-up paper (Kailath, 1967 -- attached) that shows how to calculate the BC for multivariate gaussian distributions (page 58). I believe we can apply something similar in our methodology. The paper is quite old, but the metric is used in some modern feature selection. It looks like there might be some modern alternatives that researchers are beginning to explore, but I need to do more research on those.

For reference, here is our methodology as discussed on Wednesday.

methodology

\*\*\* for a single point in time before a period of high m&a activity (so we can run an event study confirming our methodology)

1. vectorize patents

run PatenSBERTa on title + abstract
run PatentSBERTa on citations (authors + titles?)
append vectors (768 x 2 dimension vector)
reduce dimensionality to 50 dimensions (UMAP)

=> 50D vector for each patent

2. generate firm patent portfolio representations

each firm will have a set of 50D vectors representing their patent portfolio
aggregate 50D vectors into a 50D gaussian mixture model (GMM)
involves determining K (clusters)—also the number of 50D gaussian models. then aggregating them together with weights (gaussian mixture model)

\*\*\* still need method for determining optimal clusters for GMM given patent vectors

=> each firm’s patent portfolio will be represented by a 50D GMM probability distribution

3. compare probability distributions
   measure degree of overlap

bhattacharyya coefficient (BC): calculate BC comparing each GM cluster across firms, then aggregate using GMM weights
\*\*\* still need additional, economically relevant ways to compare high dimensional probability distributions

=> we will have a new metric that compares two firms’ technological overlap

we can use this as a way to compare firms’ patent portfolios, a feature in a predictive model, etc.

4. extensions
   given firms A, B, and C, can we synthetically combine the patent portfolios of A and B to create the patent portfolio of C? if so, B is an acquisition target for A to mimic C’s portfolio (also implies A and C are in competition in the same space)
   we can synthetically craft AB’s GMM as the total patents in A and B turned into a singular GMM. then we can compare AB’s GMM to C’s GMM.

given firms A and C, can we derive a representative probability distribution that when added to firm A’s distribution, create firm C’s distribution? In other words, if A + x = C, can we determine x? if so, we can solve the aforementioned problem comparing x to the set of all available firms to determine candidate target firms

Please let me know if you have any questions or if you would like to discuss more before next week’s presentations!

Thank you,
Arthur Khamkhosy
