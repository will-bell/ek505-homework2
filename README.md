ek505-homework2

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

