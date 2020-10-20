# Question 1 - Potential-Based Planner and Navigation Functions

For both methods, we will be attempting to path-plan in a planar-sphere world. The obstacles are circles inside of a circular workspace, which greatly simplifies the problem.

## Potential-Based Planner

| Eta = 0.1 | Eta = 0.5 |
| --- | --- |
| ![](examples/Potential%20Field/Eta=1e-1%20Contour.png?raw=true) | ![](examples/Potential%20Field/Eta=5e-1%20Contour.png?raw=true) |

| Eta = 1 | Eta = 2 |
| --- | --- |
| ![](examples/Potential%20Field/Eta=1%20Contour.png?raw=true) | ![](examples/Potential%20Field/Eta=2%20Contour.png?raw=true)|

First, we will discuss discuss path-planning using potential field methods. 

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

