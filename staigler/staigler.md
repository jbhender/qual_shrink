## Staigler et al Adapation


### Preliminaries and notation
The basic algorithm suggested by Staigler et al consists of the following
steps for creating composites of quality measures using multivariate shrinkage. 

In the basic setup, $O_{ijk}$ is the outcome on a binary quality measure $k$ 
for person $j$ in cluster $i$. In our adaptation, clusters will be defined
as subgroups within health care plans. When needed we will denote the subgroup
and plan corresponding to cluster $i$ by $g(i)$ and $p(i)$, respectively.  

Let $K$ be the number of measures and $n_c$ the number of clusters. Let $n_i$
be the number of unique individuals in cluster $i$ and $n_i(k)$ the number in 
the subset contributing to measure $k$. Finally, let $N = \sum_i n_i$ be the
total number of unique individuals and $N(k) = \sum_i n_i$(k)$ the total for
measure $k$. 

### Overview

1. **Estimate risk adjusted cluster-level means for each measure.**  
  a. Let $p_{ijk}(x_{ij}) = \mathbb{P}[O_{ijk} = 1 | X_{ij} = x_{ij}]$ and
     estimate $p_{ijk}$ with
     $\hat p_{ijk}(x_{ij})$ using logistic regression or another method.  
  a. Let $E_{ik} = \frac{1}{n_{ik}} \sum_j \hat p_{ijk}$,
     $O_{ik} = \frac{1}{n_{ik}} \sum_j \hat p_{ijk}$ and
     $Y_{ik} = O_{ik} / E_{ik}$.  
1. Estimate the within-cluster sampling variance,
  $V_i \in \mathbb{R}^{k \times k}$.  
1. Estimate cluster level predictions (update this for adapated model),
   $\mu_{i\dot}(z_i) := \mathbb{E}[Y_{i\dot} | Z_i = z_i] = z_i\Beta$.
1. Estimate the variance-covariance of the cluster-level means,
   $\Sigma := \mathbb{V}[\mu_{i\dot}] \in \mathbb{R}^{k \times k}$. 
1. Compute weights, $W_i = \left(\hat \Sigma + \hat V_{i}\right)$.
1. Compute risk-adjusted composites,
   $\Theta_i = Y_i W_i + \hat \mu_i (I - W_i)$. 

### Estimating $V_i$

For each cluster (plan + subgroup) $i$, $V_i$ is the sampling 
variance-covariance of $Y_i$. Follow the following steps to estimate $V_i$.

1. For each measure, compute residuals $r_{ijk} = y_{ijk} - \hat p_{ijk}$.  
1. Mean center $r_{ijk}$ within each cluster,
$\tilde r_{ijk} = r_{ijk} - \frac{1}{n_i(k)}\sum_j r_{ijk}$.
1. Let $R_i \in \mathbb{R}^{n_i \times K}$ be a matrix/dataframe where,
$R_i(j, k)$ is missing if person $j$ in cluster $i$ isn't measured on item $k$ 
and equals $\tilde r_{ijk}$ otherwise. Stack the $R_i$ to form
$R \in \mathbb{R}^{N \times K}$. $R$ is a residual matrix mean-centered for 
each measure within each cluster. 
1. Use pairwise complete observations to estimate a variance-covariance matrix
$U \in \mathbb{R}^{k \times k}$.  
1. Adjust elements of $U$ to account for the extra degrees of freedom used
to mean center $\tilde r_{ijk}$ within hospital. Let $S$ be a scaling matrix. 
Along the diagonal, the  variance $U_{kk}$ is estimated by $N(k) \sum_i n_i(k)$ 
individuals. The corresponding scaling factor is $S_{kk} = \frac{N(k) - 1}{N{k} 
- n_c + 1}. For $S_{kk'}$ count
the number of pairwise complete observations and call it $N(k, k')$ and then
let $S_{kk'} = S_{k'k} = \frac{N(k, k') - 1}{N(k) - n_c + 1}.  Note, if some 
clusters don't have data on measure $k$ then replace $n_c$ to $n_c(k)$ above. 
This should also include any clusters with only a single person ($n_i(k) \le 1$)
in which case the residuals should be changed to missing.
1. Let $E_i$ be a $K$-vector of the one over the expected outcomes used to 
normalize $O_i$ to $Y_i$ and let $E_i$ be the $K \times K$ matrix formed by the 
outer product $E_iE_i'$.  
1. Let $N_i$ be a $K \times K$ matrix with
$N_i(k, k') = \frac{1}{\sqrt{n_i(k)n_i(k')}}$.
1. For each $i$, computed $V_i = U_i * S_i * E_i * N_i$ where $*$ is 
element-wise multiplication. 

### Estimating $\mu_i$

### Estimating $\Sigma$
