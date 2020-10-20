# Question 1 - Potential-Based Planner and Navigation Functions

For both methods, we will be attempting to path-plan in a planar-sphere world. The obstacles are circles inside of a circular workspace, which greatly simplifies the problem.

## Potential-Based Planner

First, we will discuss path-planning using potential field methods. This uses the composition of an attractive potential and repulsive potential to create a potential function that has a global minimum at the goal position. The attractive potential is typically a quadratic function or the combination of a conic and quadratic function. In this case, we use the latter. Each obstacle in the space has a repulsive potential that is 0 when far from the obstacle and infinity at the obstacle's boundary. By composing these potential functions, we get a potential that is large near/in obstacles and follows the attractive potential everywhere else.

| Potential Surface | Potential Contours |
| --- | --- |
| ![](examples/Potential%20Field/Eta=1%20Surface.png?raw=true)| ![](examples/Potential%20Field/Eta=1%20Contour.png?raw=true) |

We can change how obstacles affect the potential by varying the parameter eta which effective scales the repulsive potential for each obstacle.

| Eta = 0.1 | Eta = 0.5 |
| --- | --- |
| ![](examples/Potential%20Field/Eta=1e-1%20Contour.png?raw=true) | ![](examples/Potential%20Field/Eta=5e-1%20Contour.png?raw=true) |

| Eta = 1 | Eta = 2 |
| --- | --- |
| ![](examples/Potential%20Field/Eta=1%20Contour.png?raw=true) | ![](examples/Potential%20Field/Eta=2%20Contour.png?raw=true)|

These potential functions are continuous and differentiable, so we can calculate a path using gradient descent. Since the global minimum of the attractive potential function is at the goal, in the presence of no obstacles we would always path-plan successfully to the goal. However, in the presence of obstacles, local minima arise that can cause gradient descent to get trapped so that it never reaches the goal.

![](examples/Potential%20Field/4%20paths%20on%20potential%20field%20contour.png?raw=true)

In this case, only one of the fours paths, purple, is able to avoid local minima and reach the goal. Navigation functions are able to solve this problem.

## Navigation Function

Navigation functions solve the problem of entrapment in local minima by removing them. On the surface level, the resulting surface looks very similar. However, navigation functions have attractive properties such as being smooth, continuous, and bounded to \[0, 1\]. Furthermore, they have unique global minimums that are isolated (i.e. they are Morse functions). 

| Navigation Function Surface | Navigation Function Contours |
| --- | --- |
| ![](examples/Navigation%20Function/K=3%20Surface.png?raw=true)| ![](examples/Navigation%20Function/K=3%20Contour.png?raw=true) |

We can tune the shape of the navigation function by varying the parameter k. Increasing k flattens the navigation function at points close to and far away from the goal while making the transition areas steeper.

| K=3 | K=4 | K=5 |
| --- | --- | --- |
| ![](examples/Navigation%20Function/K=3%20Contour.png?raw=true)| ![](examples/Navigation%20Function/K=4%20Contour.png?raw=true) | ![](examples/Navigation%20Function/K=5%20Contour.png?raw=true) |

We can use this function to path-plan using the same gradient descent algorithm as the potential-based planner. However, this time, there is a problem. As k gets larger, the areas we need to path-plan in have very small gradients. We can scale the gradient in our algorithm before taking a step, but we would need some intelligent scheme to scale it by a large value in the flat regions and a small value in the steep regions. Instead of doing this, we can scale the **direction** of the gradient by a fixed set step size. However, if this step size is too large to meet the tolerance for reaching the goal, we can overshoot. So, we also scale the gradient by a term in the range \[0, 1\] that decreases by an inverse square law from start to finish.

# Question 2 - Brushfire Grid and Generalized Voronoi Diagram

In the first three examples, we demonstrate the wave-front algorithm for generating the voronoi regions and the brushfire grid. The GVD is then simply created from the boundaries separating the Voronoi regions.

In the last example, we'll use the configuration of Example 3 to path-plan, using the GVD as a roadmap through the space.

## Example 1

Configuration: one square obstacle in the center of the workspace.

| Voronoi Region Generation | Brushfire Grid | Generalized Voronoi Diagram
| --- | --- | --- |
| ![](examples/Example1/VoronoiRegionGeneration.gif?raw=true) | ![](examples/Example1/BrushfireDistances.png?raw=true) | ![](examples/Example1/VoronoiBoundary.png?raw=true)

## Example 2

Configuration: two triangular obstacles in opposite corners of the workspace.

| Voronoi Region Generation | Brushfire Grid | Generalized Voronoi Diagram
| --- | --- | --- |
| ![](examples/Example2/VoronoiRegionGeneration.gif?raw=true) | ![](examples/Example2/BrushfireDistances.png?raw=true) | ![](examples/Example2/VoronoiBoundary.png?raw=true)

## Example 3

Configuration: two overlapping square obstacles.

| Voronoi Region Generation | Brushfire Grid | Generalized Voronoi Diagram
| --- | --- | --- |
| ![](examples/Example3/VoronoiRegionGeneration.gif?raw=true) | ![](examples/Example3/BrushfireDistances.png?raw=true) | ![](examples/Example3/VoronoiBoundary.png?raw=true)

## Example 4

Task: path-plan to a goal from three points using the GVD in Example 3 as a roadmap.

| Path on Brushfire Grid | Path on GVD | Path in Real Space |
| --- | --- | --- |
| ![](examples/Example4/PathOnBrushfire.png?raw=true) | ![](examples/Example4/PathOnGVD.png?raw=true) | ![](examples/Example4/PathOnRealSpace.png?raw=true)
