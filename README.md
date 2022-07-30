# Reviewer p9bJ
Weakness:

> The convergence analysis is in-complete without the discussion on convergence rates

> Theoretical assumptions need to be clarified further and it is not clear whether the NN adopted in practice indeed satisify the assumption, e.g. the Lipschitz assumption.

> The motivation is not clear. As the paper clearly aims at deep learning, it is not clear to me at all why the regularization induced benefit the application. A simple prune can do the same job.

> The discussion on experimental evaluation lacks important details. It is very challenging to reproduce the result from this paper.

> How to tune-up the hyper-parameter and how sensitive it is?

> How the proposed method compared to some recent works and how it compared to simple prune technique？


# Reviwer PdWX  

> The new method HSPG avoids manual tunning of K_switch in HSPG while introducing additional parameters m_p and m_h. 

A: $m_p$ and $m_h$ are number of iterations of the ProxSVRG step /Half-space step is performed. In practice, we never tune them. We just set both of them to `#data samples / batch size`.


> Why is the last quantity on the RHS of the inequality in Lemma 2.6 missing when deriving inequality (19) for the proof of Theorem 2.7?

> The authors mention that stronger convergence results are obtained for HSPG+ under looser assumptions in the abstract but not discussed in detail in Section 2.

> I wonder if the parameter epsilon is adapted in a similar manner in HSPG as in HSPG+, the performance of HSPG would also be improved or not in terms of solution sparsity.

# Reviewer AAJf

> One key step in the proposed algorithm (i.e., Algorithm 1) is the Prox-SVRG algorithm [21]. Thus, the novelty of this paper is limited.

> The main difference of existing half-space methods and the enhanced half-space algorithm (i.e., Algorithm 2) is not clear.


> The authors should compare the proposed algorithm with sophisticated algorithms such as Katyusha [R1]. 

> In Fig. 2, why not show ResNet18 on CIFAR10 to make a more intuitive comparison?

> In Section 1.1, the definitions of some parameters such as N are not given.

> From the results in Table 2 and Fig. 2, the accuracy results of the proposed algorithm are not better than those of existing methods.

# Reviewer nhzx 

> This paper works under the assumption that group sparse regularization is important or necessary and only compares to methods that hold this assumption. But, for all the nuanced benefits, how does this optimization algorithm compare in performance to the most commonly used algorithms? This should be added to Section 3.3 as a baseline to emphasize any tradeoff between performance and group sparsity ratio. For example, does the Adam or Nesterov-SGD optimizer converge 10x faster with 10% better performance? It might be very well worth the cost of worse group sparsity. I don't see why you would leave this out.

> Convex Experiment: Showing the raw objective value in Table 1 doesn't tell us anything without knowing the unique global solution; please show the error or relative error ||x-x*||/||x*||. 


> The group sparse ratio for kdda is 0%... so 0% of the groups were under the sparsification threshold. Does the inclusion of this dataset add anything? 

> It seems crucial to show plots of the loss curves vs. iteration for each method since you are talking about convergence rates. 
