# Reviewer p9bJ
We would like to thank the reviewers for their comments. We try to address as many of them as possible below.

1. The convergence analysis is in-complete without the discussion on convergence rates

> The main challenge that prevents us from establishing such a convergence result is the diminishing stepsize used in the EHS step. We conjecture that with additional variance reduction integrated into the EHS step one might be able to establish such a result.Here, we avoid variance reduction technique in the EHS step because the practical performance on  deep learning tasks is significantly better.

2. Theoretical assumptions need to be clarified further and it is not clear whether the NN adopted in practice indeed satisfy the assumption, e.g. the Lipschitz assumption.

>  We kindly point out that when performing the convergence analysis for non-convex non-smooth problem $f(x)+r(x)$, assume $\nabla f$ to be Lipschtiz continuous is standard in the optimization community, for example [1,2,3]. In the deep learning community, the same assumption is also popular, for example [4,5]. It's is known that the ReLu function will break the Lipschtiz continuous gradient assumption. But one can replace ReLu with SiLU[6] to address such a concern.
    
>  [1] Lin Xiao and Tong Zhang. A proximal stochastic gradient method with progressive variance
  reduction. SIAM Journal on Optimization, 24(4):2057–2075, 2014.

>  [2] Yang, M., Milzarek, A., Wen, Z., & Zhang, T. (2022). A stochastic extra-step quasi-newton method for nonsmooth nonconvex optimization. Mathematical Programming, 194(1), 257-303.

>  [3] Reddi, S. J., Hefny, A., Sra, S., Póczos, B., & Smola, A. (2016, June). Stochastic variance reduction for nonconvex optimization. In International conference on machine learning (pp. 314-323). PMLR.

>  [4] Deleu, T., & Bengio, Y. (2021). Structured sparsity inducing adaptive optimizers for deep learning. arXiv preprint arXiv:2102.03869.

>  [5] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. Proceedings of Machine Learning and Systems, 2, 429-450.

>  [6] Elfwing, S., Uchibe, E., & Doya, K. (2018). Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107, 3-11.

3. The motivation is not clear. As the paper clearly aims at deep learning, it is not clear to me at all why the regularization induced benefit the application. A simple prune can do the same job.

> As shown in literature [7, 8], simple pruning that projects groups of variables based on magnitude are easily causing model performance regression, thereby a costly re-training is typically required after simple pruning to recap the regressed accuracy. In sharp contrast, regularization based sparsity inducing method does not suffer from such significant accuracy regression carried by simple pruning, and can get a model with high sparsity as well as good performance simultaneously. Therefore, the regularization sparsity induction benefits the applications a lot.
     
> [7] Learning Pruning-Friendly Networks via Frank-Wolfe: One-Shot, Any-Sparsity, And No Retraining. 

> [8] Only Train Once: A One-Shot Neural Network Training And Pruning Framework.

4. The discussion on experimental evaluation lacks important details. It is very challenging to reproduce the result from this paper. How to tune-up the hyper-parameter and how sensitive it is?

>  The choices of hyper-parameter and the sensitivity is discussed in Section 3.1. We repeat them here for the reviewer's convenience. For most of the hyper-parameters in experiments, such as learning rate, mini-batch size, etc., we followed the settings in existing literature. Compared with HSPG, our AdaHSPG enjoys benefits that requires much fewer efforts on fine-tuning hyper-parameters via equipping with adaptive strategy. For the algorithm specific hyper-parameter, i.e., $\mu$ in adaptive switching mechanism, we set $\mu=1$ to give equal preference to the Enhanced half space step and the ProxSVRG step. Prior testing showed that our numerical results are rather insensitive to $\mu$ except when $\mu\gg 1$, which would significantly favor ProxSVRG step. 


5. How the proposed method compared to some recent works and how it compared to simple prune technique？

> The simple pruning methods sometimes set all entries below certain threshold, denoted as $\mathcal{T}$, to zero [9, 10]. However, such simple truncation mechanism is empirical, hence may hurt convergence and model performance. To illustrate this, we project the groups of variables onto zeros in the solutions of Prox-SG and Prox-SVRG (which are not effective to generate zeros in the view of optimization) if the magnitudes of the group variables are less than some $\mathcal{T}$, and denote the corresponding solutions as Prox-SG* and Prox-SVRG*. 

> As shown in the [Figure](https://github.com/DoubleBlindReview2022/NeurIPS2022_Paper7106/blob/main/versus_simple_pruning.png) (i), under the   $\mathcal{T}$ with no accuracy regression, Prox-SG* and Prox-SVRG* equipped with simple pruning still significantly perform worse than AdaHSPG+ and HSPG without simple pruning method on the group sparsity ratio. Under the $\mathcal{T}$ to reach the same group sparsity ratio as AdaHSPG+, the testing accuracy of Prox-SG* and Prox-SVRG* regresses drastically to 37% and 32%
in [Figure](https://github.com/DoubleBlindReview2022/NeurIPS2022_Paper7106/blob/main/versus_simple_pruning.png)  (ii) respectively. Remark here that although the regressed accuracy can be recapped via further fine-tuning, it
requires additional engineering efforts and training cost, which is less convenient than AdaHPSG+. 
    
> [9] Structured sparse principal component analysis.

> [10] Combinatorial penalties: Which structures are preserved by convex relaxations?

> * Links bring you to an Anonymized github account, where we host all plots.


# Reviwer PdWX  

We would like to thank the reviewers for their comments. We try to address as many of them as possible below.

1. The new method HSPG avoids manual tunning of K_switch in HSPG while introducing additional parameters m_p and m_h. 

> $m_p$ and $m_h$ are number of iterations of the ProxSVRG step and Half-space step are performed, respectively. In practice, we never tune them. We just set both of them to `#data samples / batch size`.


2. Why is the last quantity on the RHS of the inequality in Lemma 2.6 missing when deriving inequality (19) for the proof of Theorem 2.7?

> We thank the reviewer for point out this. It is indeed a typo. The term $\frac{I(|B_{k,t}| < N)\eta\sigma^2}{|B_{k,t}|}$ should be $\frac{I(|B_{k}| < N)\eta\sigma^2}{|B_{k}|}$. And in Theorem 2.7, since we assume the full batch is used, i.e.,  $|B_k|=N$, then the term $\frac{I(|B_{k}| < N)\eta\sigma^2}{|B_{k}|}$ indeed would be $0$. The modification is made to the paper. 

3. The authors mention that stronger convergence results are obtained for HSPG+ under looser assumptions in the abstract but not discussed in detail in Section 2.

> We assume that HSPG+ refers to our AdaHSPG algorithm. The differences are stated the Section 2.5. But for the reviewer's convenience, we repeat the differences. The assumptions are looser in the following aspects.

> 1) HSPG is a **two-stage** algorithm and the second stage only operates on groups of variables that are **non-zero**. If there are groups of variables that are wrongly set to zero during the first stage, then HSPG can not set those groups of variables to zero in the second stage. To avoid such a situation, HSPG requires the starting point of the second stage be “close enough" to the solution, which is unknown. On the contrary, our AdaHSPG+ is an **adaptive** algorithm that dynamically choose between two steps, hence no need to make the stringent assumption.

> 2) Analysis for HSPG requires the regularizer to be differentiable with Lipschitz continuous gradient, a property that does not hold at the origin for the group regularizer considered in this paper; thus, their analysis does not apply in our setting.
    
4. I wonder if the parameter epsilon is adapted in a similar manner in HSPG as in HSPG+, the performance of HSPG would also be improved or not in terms of solution sparsity.

> Yes, if HSPG is equipped with adaptive epsilon strategy, its performance on group sparsity exploration would be improved as well. In fact, we delicately design this strategy in AdaHSPG+ to avoid the time-costly epsilon fine-tuning efforts in HSPG to achieve high group sparse solution via one-time training. Besides the adaptive epsilon mechanism, AdaHSPG+ enjoys adaptive switching mechanism to achieve better practical performance and equips with stronger convergence theory.

# Reviewer AAJf

We would like to thank the reviewers for their comments. We try to address as many of them as possible below.

1. One key step in the proposed algorithm (i.e., Algorithm 1) is the Prox-SVRG algorithm [21]. Thus, the novelty of this paper is limited.

> As mentioned in Section 1.3, our contributions focus on exploring group sparsity and pruning parameters, on which Prox-SVRG has no superiority than rest existing methods (see Table 2 and Figure 2 - Group sparsity).

2. The main difference of existing half-space methods and the enhanced half-space algorithm (i.e., Algorithm 2) is not clear.

> The difference between the existing half-space method and the enhanced half-space algorithm is stated Section 2.5. We repeat it here for reviewer's convinces.

> In simple words, the groups of variables that are optimized by the our half-space step (Algorithm 2) is different from those are optimized by HSPG. Specifically, our method judiciously selects groups of variables are both **non-zero** and **sufficiently** far from the origin (see Equation 3). This is different from HSPG, which selects all groups of  non-zero variables to optimize. This difference, though look subtle, allows us to establish stronger convergence result under weaker assumption.


3. In Fig. 2, why not show ResNet18 on CIFAR10 to make a more intuitive comparison?

> In the main body of the paper, we want to cover a variety of neural network architectures and datasets. Therefore, besides only CIFAR10, we picked up a FashinMNIST experiment for increasing the show-case diversity. We've taken your adavice and updated the plot for ResNet18 on CIFAR10 in the main paper.

4. In Section 1.1, the definitions of some parameters such as N are not given.

>  N is the total number of data samples.

5. From the results in Table 2 and Fig. 2, the accuracy results of the proposed algorithm are not better than those of existing methods.

> As stated in the paper, our goal is to get sparser models and establish stronger convergence results. Improving the accuracy is not our major focus. Indeed, our method can produce sparser models with comparable accuracy. This brings additional benefits like fast inference and easy to deploy our models to edge devices.

<!-- 6. The authors should compare the proposed algorithm with sophisticated algorithms such as Katyusha [R1].  -->




# Reviewer nhzx 

We would like to thank the reviewers for their comments. We try to address as many of them as possible below.

1. This paper works under the assumption that group sparse regularization is important or necessary and only compares to methods that hold this assumption. But, for all the nuanced benefits, how does this optimization algorithm compare in performance to the most commonly used algorithms? This should be added to Section 3.3 as a baseline to emphasize any tradeoff between performance and group sparsity ratio. For example, does the Adam or Nesterov-SGD optimizer converge 10x faster with 10% better performance? It might be very well worth the cost of worse group sparsity. I don't see why you would leave this out.

> The reason why we do not compare with Adam or Nesterov-SGD optimizer is that they are not designed for solving problems with that with sparsity promoting regularizes. To address the about the accuracy, we collect some results that use SGD with momentum to optimize different neural networks on different datasets. As one can see from the table, sparsity promoting regularizers can achieve the comparable accuracy compared with the SGD with momentum. Meanwhile, our method AdaHSPG+ can significantly reduce the model size. This brings benefits like faster inference and easy to deploy the model on edge devices.


| Model | Dataset | Accuracy |
| ----------- | ----------- | ----------- |
**VGG16** | CIFAR10 | 92.64% |
**ResNet18** | CIFAR10 | 93.02% |
**MobileNetV2** | CIFAR10 | 94.43% |
**VGG16** | Fashion-MINST | 93.50 % |
**ResNet18** | Fashion-MINST| 94.90% |
**MobileNet** | Fashion-MINST | 95.00% |

> Accuracy baseline for existing results. (CIFAR10: https://github.com/kuangliu/pytorch-cifar; Fashion-MINST: see https://github.com/zalandoresearch/fashion-mnist)

2. Convex Experiment: Showing the raw objective value in Table 1 doesn't tell us anything without knowing the unique global solution; please show the error or relative error ||x-x*||/||x*||. 

> For the convex problem, the reason why we do not plot the relative error ||x-x*||/||x*|| is that the optimal solution x* is not known at prior. Meanwhile, since the tested problem is not strongly convex x* is not unique. Lastly, since we are solving the minimization problem, a smaller final objective value means the algorithm performs better. That's reason why we report the final objective value.
    
> However, we still use W8a dataset as an example to show the relative error for a demonstration purpose. Please click the [link](https://github.com/DoubleBlindReview2022/NeurIPS2022_Paper7106/blob/main/w8a_lambda_0.01.pdf) to see the plot.
>  Note that, the way to derive x* is by running four algorithms for 200 epochs and collecting the four final iterates. Such a process is repeated for for 5 independent run, so a total of 20 iterates are collected. We take the average of these 20 iterates as the x*.  
>  (We are aware of that the way of generating x* is not optimal.) One can see that the ProxSVRG is not performing well as one might expect. This is indeed due to the fact that the optimal solution is not unique. Such a plot might be misleading. Therefore, we do not include them in the paper.

3. The group sparse ratio for kdda is 0%... so 0% of the groups were under the sparsification threshold. Does the inclusion of this dataset add anything? 

> We thank the reviewer for pointing out this. Indeed, for a given problem, there are two thresholds $\lambda_{\max}$ and $\lambda_{\min}$. Once the $\lambda>\lambda_{\max}$, the solution becomes $0$, i.e., group sparse ratio is 100\%; on the other hand, $\lambda<\lambda_{\min}$, the solution becomes fully dense, i.e., group sparse ratio is 0\%. $\lambda_{\min}$ and $\lambda_{\max}$ need to be determined case by case. In the main paper, we just show results for different datasets under the same $\lambda$ for consistency. And in Table 4, the group sparsity of Kdda is now ~98\%. So to get a smoother transition from a fully dense solution to a fully sparse solution, we need to choose $\lambda\in (1e-3, 1e-2)$.  We will refined the choice of $\lambda$ in the revision.

4. It seems crucial to show plots of the loss curves vs. iteration for each method since you are talking about convergence rates. 

> To be clear, our contribution is not on establishing the convergence rate. But, we thank for the suggestion. We will add these plots in the final revision. For demonstration purpose, we add a plot for [W8a](https://github.com/DoubleBlindReview2022/NeurIPS2022_Paper7106/blob/main/w8a_lambda_0.01_Fval.pdf). To better show the difference, we plot $F-F^*$ instead of $F$. Again, the result is based on the average of 5 runs.
