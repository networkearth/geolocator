# geolocator
Hidden Markov Models for Geolocation of PSAT Data

## An Example

Check out [Flatland](examples/flatland//Example.ipynb)!

## Overview of the Model

Geolocators are typically based on hidden markov models (HMMs). For a full treatment see [Pederson, 2010](https://henrikmadsen.org/wp-content/uploads/2014/10/Ph.D._thesis_-_2010_-_Hidden_Markov_modelling_of_movement_data_from_fish.pdf). The reason they are considered "hidden" is that the underlying movement of the fish is said to be "hidden" behind measurement error. Therefore the actual process is hidden while the measurements are not. 

In general then there are two pieces to a hidden markov model:

1. Some model of transition (or movement) probabilities.
2. Some model of measurements (and their associated errors).

For tractability all analysis of these models is typically discretized both in space and time. Therefore there is some spatial grid and some series of timesteps over which the model is evaluated. 

Therefore given some $x_k$ (grid cell) that an animal is currently at and some other grid cell $x_{k+1}$ that it may be at at timestep $k+1$, (1) from the above is given by $P(X_{k+1}=x_{k+1} | X_k = x_k)$ and (2) is given by $P(Z_k = z_k | X_k = x_k)$. These are the transition and measurement probabilities respectively. 

One potential view on geolocation is to try and discover the probability of the animal occupying a specific cell at a specific timestep given all of the data $\vec{Z}$, $P(X_k=x_k | \vec{Z})$. Doing this requires a few steps:


First we must prep by:
1. We must build the "world", i.e. the spatio-temporal grid along with its environmental data needed for the computation of $P(Z_k = z_k | X_k = x_k)$. 
2. We must then compute the matrix transition matrix $T$ where $T_{i,j}=P(X_{k+1}=x_i | X_{k} = x_j)$ (i.e. the columns represent the source cells and the rows represent the destination cells). 
3. For each timestep $k$ we must compute the column vector $m_k=[P(Z_k = z_k | x_i)]$ which is just the likelihood that $z_k$ is measured in each cell $x_i$. 


Then if $\phi(t_{k}, z_k)$ (a column vector with an element for each $x_i$) is going to represent the probability of having moved to $x_i$ given $z_k$ we recursively compute:

$$\phi(t_{k+1}, z_k)=T\phi(t_k, z_k)$$
$$\phi(t_{k}, z_k)=\frac{m_k\bullet \phi(t_k, z_{k-1})}{\sum_{x_i}m_k\bullet \phi(t_k, z_{k-1})}$$

Where the $\bullet$ represents elementwise multiplication. This recursion starts with $\phi(t_1, z_1)$ which has a 1.0 wherever the animal was released. 

If you follow the recursion however you'll note that the location the tag was eventually picked up is not included. Therefore there is a reverse pass to "smooth" the above and add in our final location. Letting $\phi(t_k, \vec{Z})$ be the column vector of $P(X_k=x_i | \vec{Z})$ we simply recurse on:

$$\phi(t_k, \vec{Z})=\phi(t_k, z_k)\bullet\left( T^T \frac{\phi(t_{k+1},\vec{Z})}{\phi(t_{k+1}, z_k)}\right)$$

## Practically Speaking

Couple of notes on practical implementations:

1. The transfer matrix is a rather expensive object and has to be computed for each kind of behavior you want to model. However once built for a specific world it never has to be built again, so $T$ can easily be used across different animals. 
2. Given the transfer matrix is often parametrized there's some amount of optimizing those parameters that has to go on. 
3. Given the measurement and transfer matrices are really just based on distributions the code can be setup to allow you to plug and play with different distributions pretty easily. The rest of the process never changes. 
4. Most of the actual implementation challenges will come from preparing the world and inserting the tag data into the world. the rest is just some math. 
5. You can find a bunch of real life measurement distributions at [HMMoce](https://github.com/camrinbraun/HMMoce).
