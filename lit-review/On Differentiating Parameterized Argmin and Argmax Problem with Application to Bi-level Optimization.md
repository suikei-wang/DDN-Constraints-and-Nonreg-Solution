# On Differentiating Parameterized Argmin and Argmax Problem with Application to Bi-level Optimization

### TL;DR: This paper proposed a solution for parameterized lower-level problem (differentiating argmin & argmax optimization problems), which binds variables. These variables appear in the objective of an upper-level problem. It collects results on differentiating parameterized argmin and argmax problems in the context of first-order gradient procedures for solving bi-level optimization problems. 

### I add some related work from my research project, the Deep Declarative Network in this article. 



## ==MATH WARNING==

### Bi-level Optimization Problem - Overview

#### Definition

A bi-level optimization problem consists of an upper problem and a lower problem. Upper problem defines an objetive over two sets of variables $\boldsymbol{x}$ and $\boldsymbol{y}$. Lower problem binds $$\boldsymbol{y}$$ is a function of $$\boldsymbol{x}$$, typically by solving a minimization problem: 
$$
\begin{array}{ll}\operatorname{minimize}_{\boldsymbol{x}} & f^{U}(\boldsymbol{x}, \boldsymbol{y}) \\ \text { subject to } & \boldsymbol{y} \in \operatorname{argmin}_{\boldsymbol{y}^{\prime}} \boldsymbol{f}^{L}\left(\boldsymbol{x}, \boldsymbol{y}^{\prime}\right)\end{array}
$$
where $f^U$ and $f^L$ are the upper- and lower-level objectives, respectively. 

Lower-level: optimizes its objectives subject to the value of the upper-level variable $\boldsymbol{x}$.

Upper-level: optimizes the value $\boldsymbol{x}$ accoring to its own objective let the lower-level following optimally.

Two views of bi-level optimization problem:

- argmin in lower-level problem is just a mathematical function, therefore bi-level optimization $ \approx$ constrained optimization;
- argmin cannot be computed in closed-form, it is still a structure of the problem.

#### Previous methods & Background

- Explicit function method:

  Find an analytic solution for the lower-level problem, which is also known as the "constraint". The explicit function $\boldsymbol{y^*({x})}$ can return an element of $\text{argmin}_\boldsymbol{y}f^L(\boldsymbol{x}, \boldsymbol{y})$. Therefore, the bi-level problem can be simplifed as a single-level problem: $$\text{minimize}_\boldsymbol{x} \quad f^U (\boldsymbol{x}, \boldsymbol{y^*}(\boldsymbol{x}))$$ .

  Disadvantage: the explicit function can be difficult to find and no analytic solution exists. 

- Constrained problem method:

  Replace the lower-level problem with a set of sufficient conditions for optimality. For example using KKT conditions to solve convex lower-level problem (Lagrange multipliers method). Transfer as solving a constrained optimization problem.

  Disadvantage: sufficient conditions can be difficult to express and hard to solve. The constrained problem may not be convex.

- Gradient descent method:

  Apply gradient descent on the upper-level objective. Compute the gradient of the solution to the lower-level problem w.r.t. the variables in the upper-level problem and then update it. It is a popular and preferred method for large-scale and end-to-end machien learning.

  Disadvantage: there should exist a method to find the gradient at the current solution. (Non-regular solution has no gradient)



### Unconstrained Optimization Problem

$$
g(x) = \text{argmin}_{y \in \mathbb{R}} f(x, y)
$$

Considering such a scalar function above, its results is under the assumption that the minimum or maximum over $y$ of the function $f(x,y)$ exists over the domain of $x$. And the close-form representation of $g(x)$ is unnecessary. 

In this case, let $f: \mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}$ be a continuous function with first and second derivatives, then:
$$
\frac{d g(x)}{d x}=-\frac{f_{X Y}(x, g(x))}{f_{Y Y}(x, g(x))}
$$
where $f_{X Y} \doteq \frac{\partial^{2} f}{\partial x \partial y}$ and $f_{Y Y} \doteq \frac{\partial^{2} f}{\partial y^{2}}$. 

----

$$
\boldsymbol{g}(x)=\operatorname{argmin}_{\boldsymbol{y} \in \mathbb{R}^{n}} f(x, \boldsymbol{y})
$$

Considering such a vector-valued function above. In this case, let $f: \mathbb{R} \times \mathbb{R}^{n} \rightarrow \mathbb{R}$ be a continuous function with first and second derivatives, then:
$$
\boldsymbol{g}^{\prime}(x)=-f_{Y Y}(x, \boldsymbol{g}(x))^{-1} f_{X Y}(x, \boldsymbol{g}(x))
$$
where $f_{Y Y} \doteq \nabla_{\boldsymbol{y} \boldsymbol{y}}^{2} f(x, \boldsymbol{y}) \in \mathbb{R}^{n \times n}$ (matrix) and $f_{X Y} \doteq \frac{\partial}{\partial x} \nabla_{\boldsymbol{y}} f(x, \boldsymbol{y}) \in \mathbb{R}^{n}$ (vector).

----

$$
\boldsymbol{g}(\boldsymbol{x})=\operatorname{argmin}_{\boldsymbol{y} \in \mathbb{R}^{n}} f(\boldsymbol{x}, \boldsymbol{y})
$$

It is trivial to extend theresult to multiple parameters $\boldsymbol{x} = (x_1, \dots, x_m)$ by performing the derivative calculation for each parameter separately. Then:
$$
\nabla_{\boldsymbol{x}} \boldsymbol{g}\left(x_{1}, \ldots, x_{m}\right)=-f_{Y Y}(\boldsymbol{x}, \boldsymbol{g}(\boldsymbol{x}))^{-1}\left[f_{X_{1} Y}(\boldsymbol{x}, \boldsymbol{g}(\boldsymbol{x})) \quad \cdots \quad f_{X_{m} Y}(\boldsymbol{x}, \boldsymbol{g}(\boldsymbol{x}))\right]
$$

----

Argmax is similar to the argmin case:
$$
\boldsymbol{g}(x)=\operatorname{argmax}_{\boldsymbol{y} \in \mathbb{R}^{n}} f(x, \boldsymbol{y})
$$
Considering such a vector-valued function above. In this case, let $f: \mathbb{R} \times \mathbb{R}^{n} \rightarrow \mathbb{R}$ be a continuous function with first and second derivatives, then:
$$
\boldsymbol{g}^{\prime}(x)=-f_{Y Y}(x, \boldsymbol{g}(x))^{-1} f_{X Y}(x, \boldsymbol{g}(x))
$$
where $f_{Y Y} \doteq \nabla_{\boldsymbol{y} \boldsymbol{y}}^{2} f(x, \boldsymbol{y}) \in \mathbb{R}^{n \times n}$ (matrix) and $f_{X Y} \doteq \frac{\partial}{\partial x} \nabla_{\boldsymbol{y}} f(x, \boldsymbol{y}) \in \mathbb{R}^{n}$ (vector).

----

In the paper, it gives 4 examples. The first example is the vanilla one, which is a scalar mean function with only one unique optimal point. The second eample is scalar function with three local minima, which calculates the gradient w.r.t. $x$ at each stationary point and the gradient describes how each stationary point $g(x)$ moves locally with an infintisimal change in $x$. The third example is below, which is based on soft-max classifier and maximizes the log likelihood. 

**Example : Maximum Likelihood of Soft-max Classifier**

According to the definition of Softmax function, assume $$m$$ classes and let classifier be parameterized by $\Theta = \{(\boldsymbol{a}_i , b_i\}^m_{i=1}$. Then the likelihood of feature vector $\boldsymbol{x}$ for the $i$-th class of a soft-max distribution as
$$
\begin{aligned} \ell_{i}(\boldsymbol{x}) &=P(Y=i \mid \boldsymbol{X}=\boldsymbol{x} ; \Theta) \\ &=\frac{1}{Z(\boldsymbol{x} ; \Theta)} \exp \left(\boldsymbol{a}_{i}^{T} \boldsymbol{x}+b_{i}\right) \end{aligned}
$$
where $Z(\boldsymbol{x} ; \Theta)=\sum_{j=1}^{m} \exp \left(\boldsymbol{a}_{j}^{T} \boldsymbol{x}+b_{j}\right)$. (retrieved from the original softmax function)



 Therefore, the maximum log likelihood feature vector for class $i$ can be found as: (find the maximum log likelihood of function $\ell_{i}(\boldsymbol{x})$)
$$
\begin{aligned} \boldsymbol{g}_{i}(\Theta) &=\underset{\boldsymbol{x} \in \mathbb{R}^{n}}{\operatorname{argmax}} \log \ell_{i}(\boldsymbol{x}) \\ &=\underset{\boldsymbol{x} \in \mathbb{R}^{n}}{\operatorname{argmax}}\left\{\boldsymbol{a}_{i}^{T} \boldsymbol{x}+b_{i}-\log \left(\sum_{j=1}^{m} \exp \left(\boldsymbol{a}_{j}^{T} \boldsymbol{x}+b_{j}\right)\right)\right\} \end{aligned}
$$
Computing the derivative of the maximum likelihood feature vector $\boldsymbol{g}_{i}(\Theta)$ w.r.t. any of the model parameters, $\nabla_{\boldsymbol{x}} \log \ell_{i}\left(\boldsymbol{x}^{\star}\right)=0$ where $\boldsymbol{x}^{\star}=\boldsymbol{g}_{i}(\Theta)$ .



The last example is under monotonic transformations. It is based on the function $f(x,y)$ as a exponential function, which is smooth and monotonically increasing:

Let $g(x) = \text{argmin}_yf(x,y)$. Now let $\tilde{f}(x,y) = e^{f(x,y)}$ and $\tilde{g}(x) = \text{argmin}_y \tilde{f}(x,y)$ . Clearly, $\tilde{g}(x) = g{(x)}$ since the exponential function is smooth and onotonically increasing. Computing the gradients we have $\tilde{g}^{\prime}(x)= g^{\prime}(x)$ . 

Similarly, now assume $f(x,y)>0$ for all $x$ and $y$, let $\tilde{f}(x,y) = \log f(x,y)$ and $\tilde{g}(x) = \text{argmin}_y \tilde{f}(x,y)$. Again we have $\tilde{g}(x) = g{(x)}$ since the logarithmic function is smooth and monotonically increasing on the positive reals. From this stage we can still get the same result that $\tilde{g}^{\prime}(x)= g^{\prime}(x)$ and it follows the derivatives of unconstrained problem defined above.



### Unconstrained Problem in Deep Declarative Network

Here's some extension of deep declarative nodes with unconstrained optimization problem.  

Consider a function $f: \mathbb{R} \times \mathbb{R} \rarr \mathbb{R} $. Let:
$$
y(x) = \text{argmin}_{u \in \mathbb{R}^m}f(x,u)
$$
Assume $y(x)$ exists and that $f$ is second-order differentiable in the neighborhood of the point $(x,y(x))$. Set $H = D_{YY}^2f(x, y(x)) \in \mathbb{R}^{m \times m}$ and $B = D_{XY}^2f(x,y(x)) \in \mathbb{R}^{m \times n}$. Then for $H$ non-singular the derivative of $y$ with respect to $x$ is 
$$
\text{D}y(x) = -H^{-1}B
$$

----

### Equality Constrained Optimization Problems

Introduce a linear equality constraints $A\boldsymbol{y} = \boldsymbol{b}$ into the vector version of the minimization problem. 

Now we have $\boldsymbol{g}(x) = \text{argmin}_{\boldsymbol{y}:A\boldsymbol{y} = \boldsymbol{b}} f(x, \boldsymbol{y})$ and wish to find $\boldsymbol{g}^{\prime}(x)$. 

Let $f: \mathbb{R} \times \mathbb{R}^n \rarr \mathbb{R}$ be a continuous function with first and second derivatives.

$A \in \mathbb{R}^{m \times n}$ , $\boldsymbol{b} \in \mathbb{R}^m$, and $\boldsymbol{y}_0 \in \mathbb{R}^n$ be any vector satisfying $A\boldsymbol{y}_0 = \boldsymbol{b}$.

Let the columns of $F$ span the null-space of $A$. Let $\boldsymbol{z}^\star(x) \in \text{argmin}_\boldsymbol{z} f(x, \boldsymbol{y}_0 + F \boldsymbol{z})$ so that $\boldsymbol{g}(x) = \boldsymbol{y}_0 + F\boldsymbol{z}^\star(x)$ 

Then
$$
\boldsymbol{g}^{\prime}(x)=-F\left(F^{T} f_{Y Y}(x, \boldsymbol{g}(x)) F\right)^{-1} F^{T} f_{X Y}(x, \boldsymbol{g}(x))
$$
where $f_{Y Y} \doteq \nabla_{\boldsymbol{y} \boldsymbol{y}}^{2} f(x, \boldsymbol{y}) \in \mathbb{R}^{n \times n}$ and $f_{X Y} \doteq \frac{\partial}{\partial x} \nabla_{\boldsymbol{y}} f(x, \boldsymbol{y}) \in \mathbb{R}^{n}$ .

Alternatively, compute it directly using Lagrange multipliers:

For 
$$
\begin{array}{ll}\underset{\boldsymbol{y} \in \mathbb{R}^{n}}{\operatorname{minimize}} & f(x, \boldsymbol{y}) \\ \text { subject to } & A \boldsymbol{y}=\boldsymbol{b}\end{array}
$$
With Lagrange multipliers, we have 
$$
\mathcal{L}(x, \boldsymbol{y}, \boldsymbol{\lambda})=f(x, \boldsymbol{y})+\boldsymbol{\lambda}^{T}(A \boldsymbol{y}-\boldsymbol{b})
$$
Assume $\tilde{\boldsymbol{g}}(x)=\left(\boldsymbol{y}^{\star}(x), \boldsymbol{\lambda}^{\star}(x)\right)$ is a optimal primal-dual pair we can know that the derivatives on this two points are zero:
$$
\left[\begin{array}{c}\nabla_{\boldsymbol{y}} \mathcal{L}\left(x, \boldsymbol{y}^{\star}, \boldsymbol{\lambda}^{\star}\right) \\ \nabla_{\boldsymbol{\lambda}} \mathcal{L}\left(x, \boldsymbol{y}^{\star}, \boldsymbol{\lambda}^{\star}\right)\end{array}\right]=\left[\begin{array}{c}f_{Y}\left(x, \boldsymbol{y}^{\star}\right)+A^{T} \boldsymbol{\lambda}^{\star} \\ A \boldsymbol{y}^{\star}-\boldsymbol{b}\end{array}\right]=0
$$
Take the derivative of above equation w.r.t. $x$ and we can get the same result. 



### Equality Constrained Problem in Deep Declarative Network

In DDN, the equality constraints may not be linear and there may exists multiple equality constraints. 

Consider functions $f : \mathbb{R}^n \times \mathbb{R}^m \rarr \mathbb{R}$ and $h: \mathbb{R}^n \times \mathbb{R}^m \rarr \mathbb{R}^p$. Let
$$
\begin{array}{ll}{y(x) \in \operatorname{argmin}_{u \in \mathbb{R}^m}} & f(x, u) \\ \text { subject to } & h_i(x,u) = 0, i=1, \dots, p.\end{array}
$$
Assume that $y(x)$ exists, that $f$ and $h = [h_1, \dots, h_p]^T$ are second-order differentiable in the neighborhood of $x, y(x)$, and that $\text{rank}(\text{D}_Y h(x,y))=p$. Then for $H$ non-singular
$$
\text{D}y(x) = H^{-1}A^T(AH^{-1}A^T)^{-1}(AH^{-2}B-C)-H^{-1}B
$$
where 

$A=\mathrm{D}_{Y} h(x, y) \in \mathbb{R}^{p \times m}$, 

$B=\mathrm{D}_{X Y}^{2} f(x, y)-\sum_{i=1}^{p} \lambda_{i} \mathrm{D}_{X Y}^{2} h_{i}(x, y) \in \mathbb{R}^{m \times n}$

$C=D_{X} h(x, y) \in \mathbb{R}^{p \times n}$

$H=\mathrm{D}_{Y Y}^{2} f(x, y)-\sum_{i=1}^{p} \lambda_{i} \mathrm{D}_{Y Y}^{2} h_{i}(x, y) \in \mathbb{R}^{m \times m}$

and $\lambda \in \mathbb{R}^p$ satisfies $\lambda ^T A = D_Yf(x, y)$ .



----

### Inequality Constrained Optimization Problems

$$
\begin{array}{ll}\operatorname{minimize}_{\boldsymbol{y} \in \mathbb{R}} & f_{0}(x, \boldsymbol{y}) \\ \text { subject to } & f_{i}(x, \boldsymbol{y}) \leq 0 \quad i=1, \ldots, m\end{array}
$$

Considering such an inequality constrained problem. Let $\boldsymbol{g}(x) \in \mathbb{R}^n$ be an optimal solution, we are going to find $\boldsymbol{g}^{\prime}(x)$. 

Some background information:

**interior-point method.** Any convex optimization problem can be transformed into minimizing (or maximizing) a linear function over a convex set by converting to the epigraph form. 

**Barrier function.** A barrier function is a continuous function whose value on a point increases to infinity as the point approaches the boundary of the feasible region of an optimization problem. Such functions are used to replace inequality constraints by a penalizing term in the objective function that is easier to handle.

**Primal-dual interior-point method for non-linear inequality constrained problem.** It's easy to demonstrate for constrained nonlinear optimization. For simplicity, consider the all-inequality version of a nonlinear optimization problem:
$$
 \begin{array}{ll}\operatorname{minimize} & f(x) \\ \text { subject to } & c_{i}(x) \geq 0 \quad i=1, \ldots, m\end{array}
$$
where $x\in\mathbb{R}^n$ , $f: \mathbb{R}^n \rarr \mathbb{R}$, $c_i: \mathbb{R}^n \rarr \mathbb{R}$

The logarithmic barrier function of above is:
$$
B(x, \mu)=f(x)-\mu \sum_{i=1}^{m} \log \left(c_{i}(x)\right)
$$
where $\mu$ is a small positive scalar, sometimes called the "barrier parameter". As $\mu$ converges to zero the minimum of $B(x, \mu)$ should converge to a solution of this problem. 

Gradient of the barrier function:
$$
g_{b}=g-\mu \sum_{i=1}^{m} \frac{1}{c_{i}(x)} \nabla c_{i}(x)
$$
where $g$ is the gradient of the original function and $\nabla c_{i}$ is the gradient of $c_i$. 



Okay, back to the inequality constrained problem:
$$
\begin{array}{ll}\operatorname{minimize}_{\boldsymbol{y} \in \mathbb{R}} & f_{0}(x, \boldsymbol{y}) \\ \text { subject to } & f_{i}(x, \boldsymbol{y}) \leq 0 \quad i=1, \ldots, m\end{array}
$$
Introducing the log-barrier function: $\phi(x, \boldsymbol{y})=\sum_{i=1}^{m} \log \left(-f_{i}(x, \boldsymbol{y})\right)$, we can approximate the above problem as
$$
\operatorname{minimize}_{\boldsymbol{y}} \quad t f_{0}(x, \boldsymbol{y})-\sum_{i=1}^{m} \log \left(-f_{i}(x, \boldsymbol{y})\right)
$$
where $t > 0$ is a scaling factor that controls the approximation. 

Applying the gradient and Hessian of the log-barrier function, the gradient of an inequality constrained argmin function can be approximated as
$$
\boldsymbol{g}^{\prime}(x) \approx-\left(t f_{Y Y}(x, \boldsymbol{g}(x))-\phi_{Y Y}(x, \boldsymbol{g}(x))\right)^{-1}\left(t f_{X Y}(x, \boldsymbol{g}(x))-\phi_{X Y}(x, \boldsymbol{g}(x))\right)
$$
And if functions $f_i$ are not depend on $x$, above expression can be simplified by setting $\phi_{XY}(x, \boldsymbol{y})$ to zero. 



----

### Bi-level Optimization

**Example:**

Considering a 3 classes problem over 2-d space. There will always be a direction in which the likelihood tends to one as the magnitude   of the maximum-likelihood feature vector $\boldsymbol{x}$ tends to infinity. Therefore, the constraint is that the maximum-likelihood feature vectors are constrained to the unit ball centered at the origin. It can e formalised as
$$
\begin{array}{ll}\operatorname{minimize}_{\Theta} & \frac{1}{2} \sum_{i=1}^{m}\left\|\boldsymbol{g}_{i}(\Theta)-\boldsymbol{t}_{i}\right\|^{2} \\ \text { subject to } & \boldsymbol{g}_{i}(\Theta)=\underset{\boldsymbol{x}:\|\boldsymbol{x}\|_{2} \leq 1}{\operatorname{argmax}} \log \ell_{i}(\boldsymbol{x} ; \Theta)\end{array}
$$
Solving this by gradient descent gives updates:
$$
\theta^{(t+1)} \leftarrow \theta^{(t)}-\eta \sum_{i=1}^{m}\left(\boldsymbol{g}_{i}(\theta)-\boldsymbol{t}_{i}\right)^{T} \boldsymbol{g}_{i}^{\prime}(\theta)
$$
for any $\theta \in \Theta=\left\{\left(a_{i j}, b_{i}\right)\right\}_{i=1}^{m}$. Here $\eta$ is the step size. 

In such constraint, both initial parameters and final optimized parameters where we have set. the target locations to be evenly spaced around the unit circle. 





### Conclusion

The results give exact gradients but:

- require that function being optimized be smooth
- involve computing a Hessian matrix inverse, which could be expensive for large-scale problems

In practice, the methods can be applied even on non-smooth functions by approximating the function or perturbing the current solution to a nearby differentiable point. For large-scale problems, the Hessian matrix can be approximated by a diagonal matrix and still give a descent direction as stochastic gradient descent. 

For future research:

- Given changing slowly parameters for any first-order gradient update would be worth investigating whether warm-start technique swould be effectivey for speeding up gradient calculations.
- For large-scale problems, finding a descent direction rather than the direction of steepest descent is necessary. It's still unclear how such a direction culd be found without first computing the true gradient.
- The results reported herein are abased on the optial solution to the lower-level problem. It would be interesting to explore whether non exact solutions could still lead to descent directions.

I'm going to explain details of my research work based on these future works in the later article. 





